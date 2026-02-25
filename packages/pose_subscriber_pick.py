#!/usr/bin/env python3
"""
Minimal TEST version for OpenPI LIBERO policy.

Goal:
- Make OpenPI LIBERO policy inference RUN successfully
- No physical meaning, only pipeline verification
"""

import argparse
import pathlib
import sys
import time
import collections
import numpy as np

from openpi_client import websocket_client_policy
import numpy as np
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64, Float64MultiArray

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
    from std_msgs.msg import Float64MultiArray, Float64
    from sensor_msgs.msg import JointState
    ROS2_AVAILABLE = True
except Exception:
    rclpy = None
    Node = None
    PoseStamped = None
    Float64MultiArray = None
    Float64 = None
    JointState = None
    ROS2_AVAILABLE = False


# =========================
# Configuration
# =========================

IMG_SIZE = 224

# LIBERO expects 8-dim joint state
state = [0.0] * 8

# Default task instruction (can be overridden via --prompt argument)
task_instruction = "Pick up the blue square and move it in the  blue plate and return to the original position"


# =========================
# Quaternion to Axis-Angle Conversion
# =========================

def _quat2axisangle(quat):
    """
    Convert quaternion to axis-angle representation.
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    
    Args:
        quat: quaternion [x, y, z, w] or [w, x, y, z]
    
    Returns:
        axis-angle vector [x, y, z]
    """
    # Ensure quat is numpy array
    quat = np.asarray(quat)
    
    # Handle different quaternion formats: [x,y,z,w] vs [w,x,y,z]
    # Assume input is [x, y, z, w] (scipy/ROS format)
    if len(quat) == 4:
        x, y, z, w = quat[0], quat[1], quat[2], quat[3]
    else:
        raise ValueError(f"Quaternion must have 4 elements, got {len(quat)}")
    
    # Clip w component to valid range [-1, 1]
    if w > 1.0:
        w = 1.0
    elif w < -1.0:
        w = -1.0
    
    den = np.sqrt(1.0 - w * w)
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3, dtype=np.float32)
    
    return (np.array([x, y, z]) * 2.0 * math.acos(w)) / den


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
# USB Camera Manager
# =========================

class USBCameraManager:
    """管理 USB 摄像头（用于手腕相机）"""
    
    def __init__(self, device_index=0, width=640, height=480):
        """
        初始化 USB 摄像头管理器
        
        Args:
            device_index: 摄像头设备索引（默认 0，对应 /dev/video0）
            width: 图像宽度
            height: 图像高度
        """
        if not CV2_AVAILABLE:
            raise RuntimeError("cv2 is not available. Please install opencv-python.")
        
        self.cap = cv2.VideoCapture(device_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open USB camera /dev/video{device_index}")
        
        print(f"✅ USB camera opened: /dev/video{device_index} (width={width}, height={height})")
    
    def get_image(self):
        """
        获取图像
        
        Returns:
            numpy array in RGB format (H, W, 3), uint8, 或 None 如果相机不可用
        """
        ret, frame = self.cap.read()
        if not ret:
            return None
        # OpenCV 读出来是 BGR → 转 RGB
        return frame[:, :, ::-1]
    
    def stop(self):
        """停止相机"""
        if self.cap is not None:
            self.cap.release()
            print("✅ Stopped USB camera")


# =========================
# ROS2 Node
# =========================

if ROS2_AVAILABLE:
    class PoseSubscriber(Node):
        def __init__(self, pose_topic: str, action_topic: str,
                    gripper_topic: str = "/gripper/feedback_R"):
            super().__init__('pose_subscriber_libero')

            self.get_logger().info(f'Subscribing to eef_pose topic: {pose_topic}')
            self.get_logger().info(f'Subscribing to gripper topic: {gripper_topic}')
            self.get_logger().info(f'Publishing actions to topic: {action_topic}')

            # --- subscribers ---
            # /eef_pose topic contains 7-dim: [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w]
            self.create_subscription(
                Float64MultiArray,
                pose_topic,
                self.eef_pose_callback,
                10,
            )

            # 订阅 Float64MultiArray，包含 5 维数据，取第一列（索引0）作为夹爪状态值
            # 与 convert_pick_blue_bottle_hdf5_to_lerobot_downsample4x.py 中的逻辑一致
            self.create_subscription(
                Float64MultiArray,
                gripper_topic,
                self.gripper_callback,
                10,
            )

            # --- publisher ---
            self.action_publisher = self.create_publisher(
                Float64MultiArray,
                action_topic,
                10
            )

            # --- state buffers ---
            self.ee_pos = None  # 3-dim position [x, y, z]
            self.ee_rotvec = None  # 3-dim axis-angle [rx, ry, rz]
            self.latest_gripper_value = 0.0  # gripper value

            self.latest_state = np.zeros(8, dtype=np.float32)
            
            # 维护 gripper 的累积状态（用于防止频繁切换）
            self.current_gripper_state = 0.0  # 当前gripper状态：0.0=open, 1.0=close
            self.gripper_state_change_count = 0  # 状态改变的计数

        def eef_pose_callback(self, msg: Float64MultiArray):
            """
            处理 /eef_pose topic 消息
            消息格式：7维 [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w]
            """
            if len(msg.data) < 7:
                self.get_logger().warn(f'/eef_pose has {len(msg.data)} elements, expected 7')
                return
            
            # 提取位置 (前3维)
            self.ee_pos = np.array([msg.data[0], msg.data[1], msg.data[2]], dtype=np.float32)
            
            # 提取四元数 (后4维) [x, y, z, w]
            quat = np.array([msg.data[3], msg.data[4], msg.data[5], msg.data[6]], dtype=np.float32)
            
            # 转换为 axis-angle (旋转向量)
            self.ee_rotvec = _quat2axisangle(quat)
            
            self.get_logger().debug(
                f'EEF Pose: pos=({self.ee_pos[0]:.3f}, {self.ee_pos[1]:.3f}, {self.ee_pos[2]:.3f}), '
                f'rotvec=({self.ee_rotvec[0]:.3f}, {self.ee_rotvec[1]:.3f}, {self.ee_rotvec[2]:.3f})'
            )
            
            # 更新状态
            self._update_state()
        
        def gripper_callback(self, msg: Float64MultiArray):
            """
            处理右夹爪值消息
            消息格式：Float64MultiArray，包含 5 维数据，取第一列（索引0）作为夹爪状态值
            与 convert_pick_blue_bottle_hdf5_to_lerobot_downsample4x.py 中的逻辑一致：
            gripper_feedback_data = gripper_feedback_data[:, 0]  # 取第一列
            """
            if len(msg.data) < 1:
                self.get_logger().warn(f'/gripper/feedback_R has {len(msg.data)} elements, expected at least 1')
                return
            
            # 取第一列（索引0）作为夹爪状态值，与转换脚本保持一致
            self.latest_gripper_value = float(msg.data[0])
            self.get_logger().debug(
                f'Updated gripper value (from first column): {self.latest_gripper_value} '
                f'(full data length: {len(msg.data)})'
            )
            # 更新状态
            self._update_state()
        
        def _update_state(self):
            """
            更新状态：8维
            [ee_pos(3), ee_rotvec(3), gripper_value(1), -gripper_value(1)]
            """
            # 如果位置和旋转向量都可用，则构建完整状态
            if self.ee_pos is not None and self.ee_rotvec is not None:
                # 组合：位置(3) + 旋转向量(3) + 夹爪值(1) + 夹爪值相反数(1) = 8维
                self.latest_state = np.concatenate([
                    self.ee_pos,                    # 3-dim: EEF position
                    self.ee_rotvec,                 # 3-dim: EEF rotation (axis-angle)
                    np.array([self.latest_gripper_value], dtype=np.float32),  # 7th dim: gripper value
                    np.array([-self.latest_gripper_value], dtype=np.float32),  # 8th dim: -gripper value
                ])
                self.get_logger().debug(f'Updated state (8-dim): {self.latest_state}')
            else:
                # 如果数据不完整，使用零值
                if self.ee_pos is None:
                    self.ee_pos = np.zeros(3, dtype=np.float32)
                if self.ee_rotvec is None:
                    self.ee_rotvec = np.zeros(3, dtype=np.float32)
                self.latest_state = np.concatenate([
                    self.ee_pos,
                    self.ee_rotvec,
                    np.array([self.latest_gripper_value], dtype=np.float32),
                    np.array([-self.latest_gripper_value], dtype=np.float32),
                ])
        
        def publish_action(self, action: np.ndarray):
            """发布动作为 Float64MultiArray 消息"""
            if action is None:
                return
            
            # 确保是 1D 数组（如果是 action chunk，取第一个动作）
            if action.ndim > 1:
                action = action[0]  # 取第一个时间步的动作
            
            action_processed = action.copy()
            
            # 处理gripper值：只看第 6 维（索引 6）
            # Action format: [EEF_delta_pos(3), EEF_delta_ori(3), gripper_action(1)] = 7-dim
            if len(action_processed) >= 7:
                # 第 6 维（索引 6）是 gripper 值
                original_gripper = action_processed[6]
                self.get_logger().info(f'🔍 Raw gripper value (dim 6, before processing): {original_gripper:.6f}')
                
                # 改进的gripper判断逻辑：
                # 1. 使用累积状态防止频繁切换
                # 2. 考虑负值情况（负值可能表示close指令）
                # 3. 使用更严格的阈值和滞后（hysteresis）
                
                # 判断模型想要的状态（考虑正负值）
                desired_state = 1.0 if (original_gripper > 0.01 or original_gripper < -0.005) else 0.0
                # 如果绝对值较大（可能是close指令），设为close
                if abs(original_gripper) > 0.01:
                    desired_state = 1.0 if original_gripper > -0.01 else 0.0
                
                # 使用滞后逻辑：如果当前是open，需要更强的信号才能close；反之亦然
                if self.current_gripper_state < 0.5:  # 当前是open
                    # 需要更强的信号才能切换到close
                    if original_gripper > 0.02:  # 明显的正值（close信号）
                        gripper_cmd = 1.0
                        self.current_gripper_state = 1.0
                        self.gripper_state_change_count += 1
                    else:
                        gripper_cmd = 0.0  # 保持open
                else:  # 当前是close
                    # 需要明显的负值或小值才能切换到open
                    if original_gripper < -0.01:  # 明显的负值（open信号）
                        gripper_cmd = 0.0
                        self.current_gripper_state = 0.0
                        self.gripper_state_change_count += 1
                    else:
                        gripper_cmd = 1.0  # 保持close（默认保持关闭状态）
                
                action_processed[6] = gripper_cmd
                
                if gripper_cmd > 0.5:
                    self.get_logger().info(f'✅ Gripper value {original_gripper:.4f} → 1.0 (close), state changes: {self.gripper_state_change_count}')
                else:
                    self.get_logger().info(f'📌 Gripper value {original_gripper:.4f} → 0.0 (open), state changes: {self.gripper_state_change_count}')
            
            # 转换为 float64 列表
            action_list = action_processed.astype(np.float64).tolist()
            
            # 创建 Float64MultiArray 消息
            msg = Float64MultiArray()
            msg.data = action_list
            
            # 发布消息
            self.action_publisher.publish(msg)
            self.latest_action = action_processed
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


def is_using_realsense(using_realsense: bool, camera_manager) -> bool:
    """
    统一判断函数：检查是否正在使用 RealSense 相机作为输入源
    
    Args:
        using_realsense: 显式布尔状态标志
        camera_manager: RealSense 相机管理器实例
    
    Returns:
        bool: True 如果正在使用 RealSense 相机，False 否则
    """
    return using_realsense and camera_manager is not None


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
    parser.add_argument('--topic', '-t', default='/info/eef_right', help='ROS2 topic to subscribe to (eef_pose, 7-dim: pos+quat)')
    parser.add_argument('--action-topic', default='/libero/actions', 
                       help='ROS2 topic to publish actions (default: /libero/actions)')
    parser.add_argument('--gripper-topic', default='/gripper/feedback_R',
                       help='ROS2 topic to subscribe to for right gripper value (default: /gripper/feedback_R)')

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
    parser.add_argument('--use-usb-wrist', action='store_true',
                       help='Use USB camera as wrist camera')
    parser.add_argument('--usb-index', type=int, default=0,
                       help='USB camera index (default: /dev/video0)')
    args = parser.parse_args(argv)

    # 初始化 RealSense 相机（如果启用）
    camera_manager = None
    using_realsense = False
    
    if args.use_realsense:
        if not REALSENSE_AVAILABLE:
            print("❌ RealSense not available. Install with: pip install pyrealsense2")
            return 1
        
        try:
            camera_manager = RealSenseCameraManager(camera_serial=args.camera_serial)
            using_realsense = True
            print("✅ RealSense camera initialized and will be used as input")
        except Exception as e:
            print(f"❌ Failed to initialize RealSense camera: {e}")
            using_realsense = False
            return 1
    
    if args.use_realsense and not ROS2_AVAILABLE and not args.test_mode:
        print("⚠️  ROS2 not available, but RealSense is enabled.")
        print("🔄 Automatically switching to test mode (no ROS2 required)")
        args.test_mode = True

    # 初始化 USB 摄像头（如果启用）
    usb_camera = None
    if args.use_usb_wrist:
        if not CV2_AVAILABLE:
            print("❌ cv2 not available, cannot use USB camera")
            return 1
        try:
            usb_camera = USBCameraManager(device_index=args.usb_index)
            print("✅ USB camera initialized and will be used as wrist camera")
        except Exception as e:
            print(f"❌ Failed to initialize USB camera: {e}")
            return 1

    # 测试模式：不需要 ROS2
    if args.test_mode:
        print("🧪 TEST MODE: Running without ROS2 (LIBERO)")
        try:
            client = websocket_client_policy.WebsocketClientPolicy(
                host=args.host,
                port=args.port,
            )
            print("✅ Connected to policy server")
            
            # 维护 action chunk 队列
            action_plan = collections.deque()
            replan_steps = 5  # 每执行 5 个动作后重新规划（使用新的 chunk）
            
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
                        
                        # base image: RealSense
                        base_img_fixed = ensure_hwc_uint8(cam_img)
                        
                        # wrist image: USB camera
                        if usb_camera:
                            wrist_img = usb_camera.get_image()
                            if wrist_img is not None:
                                wrist_img_fixed = ensure_hwc_uint8(wrist_img)
                            else:
                                wrist_img_fixed = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                        else:
                            wrist_img_fixed = base_img_fixed  # fallback（不推荐）
                    else:
                        # 如果获取失败，使用零图像
                        base_img_fixed = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                        wrist_img_fixed = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                else:
                    # 使用假图像
                    base_img_fixed = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                    # wrist image: USB camera（即使没有 RealSense，也可以使用 USB 作为 wrist）
                    if usb_camera:
                        wrist_img = usb_camera.get_image()
                        if wrist_img is not None:
                            wrist_img_fixed = ensure_hwc_uint8(wrist_img)
                        else:
                            wrist_img_fixed = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                    else:
                        wrist_img_fixed = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                
                # 在测试模式下使用默认状态
                current_state = np.array(state, dtype=np.float32)  # 8 维
                
                # 判断输入源并打印
                if is_using_realsense(using_realsense, camera_manager):
                    print("📷 INPUT SOURCE: RealSense image")
                else:
                    print("🧪 INPUT SOURCE: Fake / zero image")
                    if not using_realsense:
                        print("   ℹ️  Reason: --use-realsense flag not set or initialization failed")
                    elif camera_manager is None:
                        print("   ℹ️  Reason: camera_manager is None")
                
                # 验证双摄像头输入（调试输出）
                print(f"🔍 base mean: {base_img_fixed.mean():.2f}, wrist mean: {wrist_img_fixed.mean():.2f}")
                
                # 如果 action 队列为空，调用推理获取新的 chunk
                if not action_plan:
                    # LIBERO policy expects images in HWC format with specific keys
                    observation = {
                        "observation/state": current_state,  # 8 维
                        "observation/image": base_img_fixed,  # HWC format (224, 224, 3)
                        "observation/wrist_image": wrist_img_fixed,  # HWC format (224, 224, 3)
                        "prompt": task_instruction,
                    }
                    
                    try:
                        print("🔄 Action queue empty, requesting new action chunk...")
                        result = client.infer(observation)
                        action_chunk = result.get("actions")
                        if action_chunk is not None:
                            print(f"✅ Action chunk received: shape={action_chunk.shape}")
                            # 将 chunk 的前 replan_steps 个动作加入队列
                            action_plan.extend(action_chunk[:replan_steps])
                            print(f"   Added {len(action_plan)} actions to queue")
                        else:
                            print("⚠️  No actions in response")
                            continue
                    except Exception as e:
                        print(f"❌ Inference error: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                # 从队列中取出一个动作
                action = action_plan.popleft()
                print(f"📤 Executing action from queue ({len(action_plan)} remaining)")
                print(f"   Action: {action}")
                
                # 检查gripper值（第 6 维，索引 6）
                if len(action) >= 7:
                    gripper_value = action[6]
                    gripper_cmd = 1.0 if gripper_value > 0.02 else 0.0
                    print(f"   🔍 Gripper value (dim 6): {gripper_value:.6f}")
                    print(f"      → Will be set to {gripper_cmd:.1f} ({'close' if gripper_cmd > 0.5 else 'open'})")
                else:
                    print(f"   ⚠️  Action has {len(action)} dims, expected at least 7")
                
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
            if usb_camera:
                usb_camera.stop()
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
    node = PoseSubscriber(args.topic, args.action_topic, args.gripper_topic)

    if args.publish_actions:
        print(f"📤 Publishing actions to: {args.action_topic}")

    # 维护 action chunk 队列
    action_plan = collections.deque()
    replan_steps = 5  # 每执行 5 个动作后重新规划（使用新的 chunk）

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
                    
                    # wrist image: USB camera
                    if usb_camera:
                        wrist_img = usb_camera.get_image()
                        if wrist_img is not None:
                            wrist_img_fixed = ensure_hwc_uint8(wrist_img)
                        else:
                            wrist_img_fixed = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                    else:
                        wrist_img_fixed = base_img_fixed  # fallback（不推荐）
                else:
                    # 如果获取失败，使用零图像
                    base_img_fixed = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                    wrist_img_fixed = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            else:
                # 使用假图像
                base_img_fixed = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                # wrist image: USB camera（即使没有 RealSense，也可以使用 USB 作为 wrist）
                if usb_camera:
                    wrist_img = usb_camera.get_image()
                    if wrist_img is not None:
                        wrist_img_fixed = ensure_hwc_uint8(wrist_img)
                    else:
                        wrist_img_fixed = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                else:
                    wrist_img_fixed = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

            # 获取最新的关节状态（如果 ROS2 可用）
            if ROS2_AVAILABLE and hasattr(node, 'latest_state'):
                current_state = node.latest_state
            else:
                # 使用默认状态
                current_state = np.array(state, dtype=np.float32)
            
            # 判断输入源并打印
            if is_using_realsense(using_realsense, camera_manager):
                print("📷 INPUT SOURCE: RealSense image")
            else:
                print("🧪 INPUT SOURCE: Fake / zero image")
                if not using_realsense:
                    print("   ℹ️  Reason: --use-realsense flag not set or initialization failed")
                elif camera_manager is None:
                    print("   ℹ️  Reason: camera_manager is None")
            
            # 验证双摄像头输入（调试输出）
            print(f"🔍 base mean: {base_img_fixed.mean():.2f}, wrist mean: {wrist_img_fixed.mean():.2f}")
            
            # 如果 action 队列为空，调用推理获取新的 chunk
            if not action_plan:
                # LIBERO policy expects images in HWC format with specific keys
                observation = {
                    "observation/state": current_state,  # 8 维（从 joint_states 获取或使用默认值）
                    "observation/image": base_img_fixed,  # HWC format (224, 224, 3)
                    "observation/wrist_image": wrist_img_fixed,  # HWC format (224, 224, 3)
                    "prompt": task_instruction,
                }

                try:
                    print("🔄 Action queue empty, requesting new action chunk...")
                    result = client.infer(observation)
                    action_chunk = result.get("actions")
                    
                    if action_chunk is not None:
                        print(f"✅ Action chunk received: shape={action_chunk.shape}")
                        # 将 chunk 的前 replan_steps 个动作加入队列
                        action_plan.extend(action_chunk[:replan_steps])
                        print(f"   Added {len(action_plan)} actions to queue")
                    else:
                        print("⚠️  No actions in response")
                        rclpy.spin_once(node, timeout_sec=0.1)
                        continue
                        
                except Exception as e:
                    print(f"❌ Inference error: {e}")
                    import traceback
                    traceback.print_exc()
                    rclpy.spin_once(node, timeout_sec=0.1)
                    continue
            
            # 从队列中取出一个动作
            action = action_plan.popleft()
            print(f"📤 Executing action from queue ({len(action_plan)} remaining)")
            print(f"   Action: {action}")
            
            # 检查gripper值（第 6 维，索引 6）
            if len(action) >= 7:
                gripper_value = action[6]
                # 使用与 publish_action 相同的逻辑
                if node.current_gripper_state < 0.5:  # 当前是open
                    gripper_cmd = 1.0 if gripper_value > 0.02 else 0.0
                    if gripper_cmd > 0.5:
                        node.current_gripper_state = 1.0
                        node.gripper_state_change_count += 1
                else:  # 当前是close
                    gripper_cmd = 0.0 if gripper_value < -0.01 else 1.0
                    if gripper_cmd < 0.5:
                        node.current_gripper_state = 0.0
                        node.gripper_state_change_count += 1
                print(f"   🔍 Gripper value (dim 6): {gripper_value:.6f}")
                print(f"      → Will be set to {gripper_cmd:.1f} ({'close' if gripper_cmd > 0.5 else 'open'}), state changes: {node.gripper_state_change_count}")
            else:
                print(f"   ⚠️  Action has {len(action)} dims, expected at least 7")
            
            # 发布动作到 ROS2 话题
            if args.publish_actions:
                node.publish_action(action)

            rclpy.spin_once(node, timeout_sec=0.1)
            time.sleep(0.2)   # 5 Hz (每 0.2 秒执行一个动作)

    except KeyboardInterrupt:
        print("\n🛑 Interrupted")

    finally:
        if args.show_camera and CV2_AVAILABLE:
            cv2.destroyAllWindows()
        if camera_manager:
            camera_manager.stop()
        if usb_camera:
            usb_camera.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

