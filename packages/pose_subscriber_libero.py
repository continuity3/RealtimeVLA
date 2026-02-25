#!/usr/bin/env python3
"""
Minimal TEST version for OpenPI LIBERO policy.

Goal:
- Make OpenPI LIBERO policy inference RUN successfully
- No physical meaning, only pipeline verification
"""

import argparse
import sys
import time
import numpy as np

from openpi_client import websocket_client_policy

# OpenCV 用于可视化
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️  cv2 not available. Install with: pip install opencv-python")

# RealSense 相机支持
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("⚠️  pyrealsense2 not available. Install with: pip install pyrealsense2")

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import Float64MultiArray
    from sensor_msgs.msg import JointState
    ROS2_AVAILABLE = True
except Exception:
    rclpy = None
    Node = None
    PoseStamped = None
    Float64MultiArray = None
    JointState = None
    ROS2_AVAILABLE = False


# =========================
# Configuration
# =========================

IMG_SIZE = 224

# LIBERO expects 8-dim joint state
state = [0.0] * 8

# Default task instruction (can be overridden via --prompt argument)
task_instruction = "Pick up the blue square and place it in the blue tray."


# =========================
# RealSense Camera Manager
# =========================

class RealSenseCameraManager:
    """管理 RealSense 相机（LIBERO 版本）"""
    
    def __init__(self, camera_serial=None, width=640, height=480, fps=30):
        """
        初始化 RealSense 相机管理器
        
        Args:
            camera_serial: 相机序列号，如果为 None，将自动检测第一个可用相机
            width: 图像宽度
            height: 图像高度
            fps: 帧率
        """
        if not REALSENSE_AVAILABLE:
            raise RuntimeError("pyrealsense2 is not available. Please install it.")
        
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        
        # 检测可用相机
        ctx = rs.context()
        devices = ctx.query_devices()
        
        print(f"Found {len(devices)} RealSense device(s):")
        available_serials = []
        for dev in devices:
            serial = dev.get_info(rs.camera_info.serial_number)
            name = dev.get_info(rs.camera_info.name)
            print(f"  - {name} (Serial: {serial})")
            available_serials.append(serial)
        
        if len(devices) == 0:
            raise RuntimeError("No RealSense cameras found!")
        
        # 如果没有指定序列号，使用第一个可用相机
        if camera_serial is None:
            camera_serial = available_serials[0]
        
        # 初始化相机
        try:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(camera_serial)
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            
            pipeline.start(config)
            self.pipeline = pipeline
            print(f"✅ Started camera (Serial: {camera_serial})")
        except Exception as e:
            print(f"⚠️  Failed to start camera (Serial: {camera_serial}): {e}")
            raise
    
    def get_image(self):
        """
        获取图像
        
        Returns:
            numpy array in RGB format (H, W, 3), uint8, 或 None 如果相机不可用
        """
        if self.pipeline is None:
            return None
        
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            color_frame = frames.get_color_frame()
            
            if color_frame:
                # 转换为 numpy 数组 (BGR format)
                img = np.asanyarray(color_frame.get_data())
                # BGR -> RGB
                img = img[:, :, ::-1]
                return img
        except Exception as e:
            print(f"⚠️  Error reading from camera: {e}")
        
        return None
    
    def stop(self):
        """停止相机"""
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
                print("✅ Stopped camera")
            except Exception as e:
                print(f"⚠️  Error stopping camera: {e}")


# =========================
# ROS2 Node
# =========================

if ROS2_AVAILABLE:
    class PoseSubscriber(Node):
        def __init__(self, pose_topic: str, action_topic: str, joint_states_topic: str = "/joint_states"):
            super().__init__('pose_subscriber_libero')
            self.get_logger().info(f'Subscribing to pose topic: {pose_topic}')
            self.get_logger().info(f'Subscribing to joint_states topic: {joint_states_topic}')
            self.get_logger().info(f'Publishing actions to topic: {action_topic}')
            
            # 订阅位姿话题
            self.create_subscription(
                PoseStamped,
                pose_topic,
                self.pose_callback,
                10,
            )
            
            # 订阅关节状态话题
            self.create_subscription(
                JointState,
                joint_states_topic,
                self.joint_states_callback,
                10,
            )
            
            # 发布动作话题
            self.action_publisher = self.create_publisher(
                Float64MultiArray,
                action_topic,
                10
            )
            
            self.latest_action = None
            self.latest_state = np.array([0.0] * 8, dtype=np.float32)  # LIBERO 需要 8 维

        def pose_callback(self, msg: PoseStamped):
            # Only log, not used for control yet
            p = msg.pose.position
            self.get_logger().info(
                f"Pose: ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})"
            )
        
        def joint_states_callback(self, msg: JointState):
            """处理关节状态消息"""
            if len(msg.position) >= 14:
                # 提取第 7-13 维（索引 7-12，共 7 个值）
                joint_positions = np.array(msg.position[7:14], dtype=np.float32)
                # 添加夹爪维度（值为 0）
                self.latest_state = np.concatenate([joint_positions, np.array([0.0], dtype=np.float32)])
                self.get_logger().debug(f'Updated state: {self.latest_state}')
            else:
                self.get_logger().warn(f'JointState has {len(msg.position)} positions, expected at least 14')
        
        def publish_action(self, action: np.ndarray):
            """发布动作为 Float64MultiArray 消息"""
            if action is None:
                return
            
            # 确保是 1D 数组（如果是 action chunk，取第一个动作）
            if action.ndim > 1:
                action = action[0]  # 取第一个时间步的动作
            
            # 转换为 float64 列表
            action_list = action.astype(np.float64).tolist()
            
            # 创建 Float64MultiArray 消息
            msg = Float64MultiArray()
            msg.data = action_list
            
            # 发布消息
            self.action_publisher.publish(msg)
            self.latest_action = action
            self.get_logger().debug(f'Published action: {action_list}')
else:
    # 占位符类（当 ROS2 不可用时）
    class PoseSubscriber:
        def __init__(self, pose_topic: str, action_topic: str):
            pass
        
        def publish_action(self, action: np.ndarray):
            """占位符方法"""
            pass

def ensure_hwc_uint8(img: np.ndarray) -> np.ndarray:
    """
    Ensure image is (H, W, 3) uint8 for LIBERO policy.
    LIBERO expects images in HWC format: (224, 224, 3)
    """
    img = np.asarray(img)

    # 去掉 batch 维
    if img.ndim == 4 and img.shape[0] == 1:
        img = img[0]

    # CHW -> HWC (如果当前是 CHW 格式)
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))

    # 灰度 -> RGB
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)

    # 单通道 -> 3 通道
    if img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    # 确保是 (224, 224, 3)
    if img.shape != (IMG_SIZE, IMG_SIZE, 3):
        # 如果尺寸不对，调整大小
        from PIL import Image
        img_pil = Image.fromarray(img)
        img_pil = img_pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        img = np.array(img_pil)

    assert img.shape == (IMG_SIZE, IMG_SIZE, 3), f"Bad image shape: {img.shape}, expected ({IMG_SIZE}, {IMG_SIZE}, 3)"

    return img.astype(np.uint8)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='ROS2 PoseStamped subscriber with OpenPI LIBERO policy inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test mode (no ROS2 required):
  python3 packages/pose_subscriber_libero.py --test-mode
  
  # With ROS2:
  python3 packages/pose_subscriber_libero.py --topic /pose
  
  # Custom server:
  python3 packages/pose_subscriber_libero.py --test-mode --host localhost --port 8001
        """
    )
    parser.add_argument('--topic', '-t', default='/pose', help='ROS2 topic to subscribe to (pose)')
    parser.add_argument('--action-topic', default='/libero/actions', 
                       help='ROS2 topic to publish actions (default: /libero/actions)')
    parser.add_argument('--joint-states-topic', default='/joint_states',
                       help='ROS2 topic to subscribe to for joint states (default: /joint_states)')

    parser.add_argument('--host', default='localhost', help='Policy server host (default: localhost)')
    parser.add_argument('--port', type=int, default=8000, help='Policy server port (default: 8000)')
    parser.add_argument('--test-mode', action='store_true', 
                       help='Run in test mode without ROS2 (for testing policy server connection)')
    parser.add_argument('--publish-actions', action='store_true', default=True,
                       help='Publish actions to ROS2 topic (default: True)')
    parser.add_argument('--use-realsense', action='store_true',
                       help='Use RealSense camera instead of fake images')
    parser.add_argument('--camera-serial', type=str, default=None,
                       help='RealSense serial number (if not specified, uses first available camera)')
    parser.add_argument('--show-camera', action='store_true',
                       help='Show RealSense camera feed in a window (requires opencv-python)')
    args = parser.parse_args(argv)

    # 初始化 RealSense 相机（如果启用）
    camera_manager = None
    if args.use_realsense:
        if not REALSENSE_AVAILABLE:
            print("❌ RealSense not available. Install with: pip install pyrealsense2")
            return 1
        
        try:
            camera_manager = RealSenseCameraManager(camera_serial=args.camera_serial)
            print("✅ RealSense camera initialized")
        except Exception as e:
            print(f"❌ Failed to initialize RealSense camera: {e}")
            return 1
    
    if args.use_realsense and not ROS2_AVAILABLE and not args.test_mode:
        print("⚠️  ROS2 not available, but RealSense is enabled.")
        print("🔄 Automatically switching to test mode (no ROS2 required)")
        args.test_mode = True

    # 测试模式：不需要 ROS2
    if args.test_mode:
        print("🧪 TEST MODE: Running without ROS2 (LIBERO)")
        try:
            client = websocket_client_policy.WebsocketClientPolicy(
                host=args.host,
                port=args.port,
            )
            print("✅ Connected to policy server")
            
            while True:
                # 从 RealSense 获取图像或使用假图像
                if camera_manager:
                    # 从相机读取图像
                    cam_img = camera_manager.get_image()
                    if cam_img is not None:
                        # 显示相机画面（如果启用）
                        if args.show_camera and CV2_AVAILABLE:
                            # 显示原始图像（BGR 格式用于 OpenCV）
                            display_img = cam_img[:, :, ::-1]  # RGB -> BGR for OpenCV
                            cv2.imshow('RealSense Camera Feed', display_img)
                            # 按 'q' 键退出，或等待 1ms（非阻塞）
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                print("🛑 Camera window closed by user")
                                break
                        
                        # 转换为 HWC 格式并调整大小
                        base_img_fixed = ensure_hwc_uint8(cam_img)
                        wrist_img_fixed = base_img_fixed  # 暂时使用相同图像
                    else:
                        # 如果获取失败，使用零图像
                        base_img_fixed = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                        wrist_img_fixed = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                else:
                    # 使用假图像
                    base_img_fixed = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                    wrist_img_fixed = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                
                # 在测试模式下使用默认状态
                current_state = np.array(state, dtype=np.float32)  # 8 维
                
                # LIBERO policy expects images in HWC format with specific keys
                observation = {
                    "observation/state": current_state,  # 8 维
                    "observation/image": base_img_fixed,  # HWC format (224, 224, 3)
                    "observation/wrist_image": wrist_img_fixed,  # HWC format (224, 224, 3)
                    "prompt": task_instruction,
                }
                
                try:
                    result = client.infer(observation)
                    actions = result.get("actions")
                    if actions is not None:
                        print(f"✅ Action chunk: shape={actions.shape}")
                        # LIBERO actions are typically 7-dim, but model may return more
                        if len(actions) > 0 and actions.shape[1] >= 7:
                            print(f"   Actions (first step, first 7 dims): {actions[0, :7]}")
                        elif len(actions) > 0:
                            print(f"   Actions (first step): {actions[0]}")
                    else:
                        print("⚠️  No actions in response")
                except Exception as e:
                    print(f"❌ Inference error: {e}")
                    import traceback
                    traceback.print_exc()
                
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n🛑 Interrupted")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return 1
        finally:
            if args.show_camera and CV2_AVAILABLE:
                cv2.destroyAllWindows()
            if camera_manager:
                camera_manager.stop()
        return 0

    # ROS2 模式
    if rclpy is None:
        print("❌ rclpy not available. Run inside ROS2 environment or use --test-mode")
        sys.exit(1)

    # Connect to OpenPI inference server
    print(f"🔌 Connecting to policy server at {args.host}:{args.port}...")
    try:
        client = websocket_client_policy.WebsocketClientPolicy(
            host=args.host,
            port=args.port,
        )
        print("✅ Connected to policy server")
    except Exception as e:
        print(f"❌ Failed to connect to policy server: {e}")
        print("   Make sure the policy server is running:")
        print("   uv run scripts/serve_policy.py --env LIBERO")
        return 1

    rclpy.init()
    node = PoseSubscriber(args.topic, args.action_topic, args.joint_states_topic)

    if args.use_realsense:
        print("📷 Using RealSense camera for images")
    else:
        print("✅ TEST MODE: Sending fake LIBERO observation")
    if args.publish_actions:
        print(f"📤 Publishing actions to: {args.action_topic}")

    try:
        while True:
            # 从 RealSense 获取图像或使用假图像
            if camera_manager:
                # 从相机读取图像
                cam_img = camera_manager.get_image()
                if cam_img is not None:
                    # 显示相机画面（如果启用）
                    if args.show_camera and CV2_AVAILABLE:
                        # 显示原始图像（BGR 格式用于 OpenCV）
                        display_img = cam_img[:, :, ::-1]  # RGB -> BGR for OpenCV
                        cv2.imshow('RealSense Camera Feed', display_img)
                        # 按 'q' 键退出，或等待 1ms（非阻塞）
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("🛑 Camera window closed by user")
                            break
                    
                    # 转换为 HWC 格式并调整大小
                    base_img_fixed = ensure_hwc_uint8(cam_img)
                    wrist_img_fixed = base_img_fixed  # 暂时使用相同图像
                else:
                    # 如果获取失败，使用零图像
                    base_img_fixed = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                    wrist_img_fixed = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            else:
                # 使用假图像
                base_img_fixed = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                wrist_img_fixed = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

            # 获取最新的关节状态（如果 ROS2 可用）
            if ROS2_AVAILABLE and hasattr(node, 'latest_state'):
                current_state = node.latest_state
            else:
                # 使用默认状态
                current_state = np.array(state, dtype=np.float32)
            
            # LIBERO policy expects images in HWC format with specific keys
            observation = {
                "observation/state": current_state,  # 8 维（从 joint_states 获取或使用默认值）
                "observation/image": base_img_fixed,  # HWC format (224, 224, 3)
                "observation/wrist_image": wrist_img_fixed,  # HWC format (224, 224, 3)
                "prompt": task_instruction,
            }

            try:
                result = client.infer(observation)
                actions = result.get("actions")
                
                if actions is not None:
                    print(f"✅ Action chunk received: shape={actions.shape}")
                    # LIBERO actions are typically 7-dim, but model may return more
                    if len(actions) > 0:
                        if actions.shape[1] >= 7:
                            action_first = actions[0, :7]
                            print(f"   Actions (first step, first 7 dims): {action_first}")
                        else:
                            action_first = actions[0]
                            print(f"   Actions (first step): {action_first}")
                        
                        # 发布动作到 ROS2 话题
                        if args.publish_actions:
                            node.publish_action(actions[0])  # 发布第一个时间步的动作
                    else:
                        print("   Empty action chunk")
                else:
                    print("⚠️  No actions in response")
                    
            except Exception as e:
                print(f"❌ Inference error: {e}")
                import traceback
                traceback.print_exc()

            rclpy.spin_once(node, timeout_sec=0.1)
            time.sleep(0.5)   # 2 Hz (avoid spamming server)

    except KeyboardInterrupt:
        print("\n🛑 Interrupted")

    finally:
        if args.show_camera and CV2_AVAILABLE:
            cv2.destroyAllWindows()
        if camera_manager:
            camera_manager.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

