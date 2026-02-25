#!/usr/bin/env python3
"""
Minimal TEST version for OpenPI LIBERO policy.

Goal:
- Make OpenPI LIBERO policy inference RUN successfully
- No physical meaning, only pipeline verification
"""

import argparse
import json
import pathlib
import sys
import time
import numpy as np
from datetime import datetime

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
task_instruction = "Pick up the blue square"

# 反归一化相关全局变量
action_norm_stats = None
use_quantile_norm = False


# =========================
# 反归一化功能
# =========================

def load_norm_stats(norm_stats_path: str | pathlib.Path):
    """
    加载归一化统计信息（不依赖 openpi 包）
    
    Args:
        norm_stats_path: norm_stats.json 文件的路径
    """
    global action_norm_stats, use_quantile_norm
    
    norm_stats_path = pathlib.Path(norm_stats_path)
    if not norm_stats_path.exists():
        print(f"⚠️  Norm stats file not found: {norm_stats_path}")
        print("    Actions will not be unnormalized.")
        return False
    
    try:
        with open(norm_stats_path, 'r') as f:
            data = json.load(f)
        
        # 解析 JSON 结构：{"norm_stats": {"actions": {...}, "state": {...}}}
        if "norm_stats" in data:
            norm_stats_dict = data["norm_stats"]
        else:
            norm_stats_dict = data
        
        if "actions" not in norm_stats_dict:
            print(f"⚠️  No 'actions' key in norm stats file")
            return False
        
        action_stats = norm_stats_dict["actions"]
        action_norm_stats = {
            "mean": np.array(action_stats.get("mean", [])),
            "std": np.array(action_stats.get("std", [])),
            "q01": np.array(action_stats.get("q01", [])) if action_stats.get("q01") is not None else None,
            "q99": np.array(action_stats.get("q99", [])) if action_stats.get("q99") is not None else None,
        }
        
        # 判断是否使用分位数归一化（如果有 q01 和 q99，通常使用分位数归一化）
        use_quantile_norm = action_norm_stats["q01"] is not None and action_norm_stats["q99"] is not None
        
        print(f"✅ Loaded norm stats from: {norm_stats_path}")
        print(f"   Action stats shape - mean: {action_norm_stats['mean'].shape}, std: {action_norm_stats['std'].shape}")
        if use_quantile_norm:
            print(f"   Using quantile normalization (Q01, Q99)")
            if action_norm_stats["q01"] is not None:
                print(f"   Actions Q01 shape: {action_norm_stats['q01'].shape}, values: {action_norm_stats['q01']}")
            if action_norm_stats["q99"] is not None:
                print(f"   Actions Q99 shape: {action_norm_stats['q99'].shape}, values: {action_norm_stats['q99']}")
        else:
            print(f"   Using z-score normalization (mean, std)")
            print(f"   Actions Mean shape: {action_norm_stats['mean'].shape}, values: {action_norm_stats['mean']}")
            print(f"   Actions Std shape: {action_norm_stats['std'].shape}, values: {action_norm_stats['std']}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to load norm stats: {e}")
        import traceback
        traceback.print_exc()
        return False


def unnormalize_action(normalized_action: np.ndarray) -> np.ndarray:
    """
    反归一化 action（不依赖 openpi 包）
    
    Args:
        normalized_action: 归一化后的 action (可以是 1D 或 2D 数组)
    
    Returns:
        反归一化后的 action
    """
    global action_norm_stats, use_quantile_norm
    
    if action_norm_stats is None:
        # 如果没有加载统计信息，直接返回原值
        print("⚠️  Warning: action_norm_stats is None, returning normalized action as-is")
        return normalized_action
    
    # 确保是 numpy 数组
    action = np.asarray(normalized_action)
    original_shape = action.shape
    
    # 如果是 2D，先处理为 1D（取第一个时间步）
    if action.ndim > 1:
        action = action.reshape(-1, action.shape[-1])
        is_2d = True
    else:
        action = action.reshape(1, -1)
        is_2d = False
    
    action_dim = action.shape[-1]
    
    if use_quantile_norm:
        # 分位数反归一化: (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        q01 = action_norm_stats["q01"]
        q99 = action_norm_stats["q99"]
        
        # 确保 q01 和 q99 是 1D 数组
        if q01.ndim > 1:
            q01 = q01.flatten()
        if q99.ndim > 1:
            q99 = q99.flatten()
        
        # 处理维度不匹配的情况
        if q01.shape[0] < action_dim:
            # 如果统计信息维度小于 action 维度，只对前面的维度进行反归一化
            unnormalized = np.zeros_like(action)
            dim = q01.shape[0]
            q01_sel = q01[:dim]
            q99_sel = q99[:dim]
            unnormalized[..., :dim] = (action[..., :dim] + 1.0) / 2.0 * (q99_sel - q01_sel + 1e-6) + q01_sel
            unnormalized[..., dim:] = action[..., dim:]  # 后面的维度保持不变
        else:
            # 截取到匹配的维度
            q01_sel = q01[:action_dim]
            q99_sel = q99[:action_dim]
            unnormalized = (action + 1.0) / 2.0 * (q99_sel - q01_sel + 1e-6) + q01_sel
    else:
        # Z-score 反归一化: x * (std + 1e-6) + mean
        mean = action_norm_stats["mean"]
        std = action_norm_stats["std"]
        
        # 确保 mean 和 std 是 1D 数组
        if mean.ndim > 1:
            mean = mean.flatten()
        if std.ndim > 1:
            std = std.flatten()
        
        # 处理维度不匹配的情况
        if mean.shape[0] < action_dim:
            # 如果统计信息维度小于 action 维度，只对前面的维度进行反归一化
            unnormalized = np.zeros_like(action)
            dim = mean.shape[0]
            unnormalized[..., :dim] = action[..., :dim] * (std[:dim] + 1e-6) + mean[:dim]
            unnormalized[..., dim:] = action[..., dim:]  # 后面的维度保持不变
        else:
            # 截取到匹配的维度
            mean_sel = mean[:action_dim]
            std_sel = std[:action_dim]
            unnormalized = action * (std_sel + 1e-6) + mean_sel
    
    # 恢复原始形状
    if is_2d:
        return unnormalized.reshape(original_shape)
    else:
        return unnormalized.reshape(original_shape)


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
        def __init__(self, pose_topic: str, action_topic: str, joint_states_topic: str = "/joint_states", gripper_topic: str = "/gripper/feedback_R"):
            super().__init__('pose_subscriber_libero')
            self.get_logger().info(f'Subscribing to pose topic: {pose_topic}')
            self.get_logger().info(f'Subscribing to joint_states topic: {joint_states_topic}')
            self.get_logger().info(f'Subscribing to gripper topic: {gripper_topic}')
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
            
            # 订阅右夹爪话题（用于读取真实的夹爪值）
            self.create_subscription(
                Float64,
                gripper_topic,
                self.gripper_callback,
                10,
            )
            
            # 发布动作话题
            self.action_publisher = self.create_publisher(
                Float64MultiArray,
                action_topic,
                10
            )
            
            self.latest_action = None
            self.latest_joint_positions = None  # 7个右臂关节位置
            self.latest_gripper_value = 0.0  # 右夹爪值（初始为0，等待接收）
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
                # 提取第 7-13 维（索引 7-12，共 7 个值）- 右臂关节位置
                joint_positions = np.array(msg.position[7:14], dtype=np.float32)
                self.latest_joint_positions = joint_positions
                # 使用最新的夹爪值（如果已接收）或使用0
                self._update_state()
            else:
                self.get_logger().warn(f'JointState has {len(msg.position)} positions, expected at least 14')
        
        def gripper_callback(self, msg: Float64):
            """处理右夹爪值消息"""
            # 夹爪值范围通常是 [0.0, 1.0] (0=全开, 1=全闭)
            self.latest_gripper_value = float(msg.data)
            self.get_logger().debug(f'Updated gripper value: {self.latest_gripper_value}')
            # 更新状态
            self._update_state()
        
        def _update_state(self):
            """更新状态（7个关节位置 + 1个夹爪值）"""
            if self.latest_joint_positions is not None:
                # 组合右臂关节位置和右夹爪值
                self.latest_state = np.concatenate([
                    self.latest_joint_positions,
                    np.array([self.latest_gripper_value], dtype=np.float32)
                ])
                self.get_logger().debug(f'Updated state: {self.latest_state}')
        
        def publish_action(self, action: np.ndarray):
            """发布动作为 Float64MultiArray 消息（已反归一化）"""
            if action is None:
                return
            
            # 先进行反归一化
            action = unnormalize_action(action)
            
            # 确保是 1D 数组（如果是 action chunk，取第一个动作）
            if action.ndim > 1:
                action = action[0]  # 取第一个时间步的动作
            
            # 处理gripper值：如果最后一维大于0.02，自动变为1（让机械臂闭合）
            action_processed = action.copy()
            if len(action_processed) >= 8:
                # 最后一维是gripper值（索引7）
                original_gripper = action_processed[7]
                self.get_logger().info(f'🔍 Raw gripper value (before processing): {original_gripper:.6f}')
                
                # 如果gripper值大于0.02，设置为1.0（闭合）
                # 注意：如果gripper值是负数，可能是归一化后的值，需要检查是否需要取绝对值
                if original_gripper > 0.02:
                    action_processed[7] = 1.0
                    self.get_logger().info(f'✅ Gripper value {original_gripper:.4f} > 0.02, set to 1.0 (close)')
                elif original_gripper < -0.02:
                    # 如果gripper值是负数且绝对值大于0.02，可能是归一化后的"闭合"信号
                    # 这种情况下，我们也可以设置为1.0
                    action_processed[7] = 1.0
                    self.get_logger().info(f'✅ Gripper value {original_gripper:.4f} < -0.02 (negative close signal), set to 1.0 (close)')
                else:
                    # 保持原值（通常是0或接近0，表示打开）
                    action_processed[7] = 0.0  # 确保打开状态为0
                    self.get_logger().info(f'📌 Gripper value {original_gripper:.4f} in [-0.02, 0.02], set to 0.0 (open)')
            
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

def warmup_inference(client: websocket_client_policy.WebsocketClientPolicy):
    """
    预编译（预热）推理，避免第一次真实推理时的延迟导致的超时
    
    Args:
        client: WebSocket 客户端
    """
    print("🔥 Warming up inference server (pre-compiling)...")
    try:
        # 创建 dummy observation（与真实 observation 格式相同）
        dummy_state = np.array(state, dtype=np.float32)  # 8 维
        dummy_image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        dummy_wrist_image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        
        dummy_observation = {
            "observation/state": dummy_state,
            "observation/image": dummy_image,
            "observation/wrist_image": dummy_wrist_image,
            "prompt": task_instruction,
        }
        
        # 执行一次推理（预编译）
        result = client.infer(dummy_observation)
        print("✅ Warmup inference completed successfully")
        if result.get("actions") is not None:
            print(f"   Warmup action shape: {result['actions'].shape}")
        return True
    except Exception as e:
        print(f"⚠️  Warmup inference failed: {e}")
        print("   Continuing anyway, but first real inference may be slow...")
        return False


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
    parser.add_argument('--record', type=str, default=None,
                       help='Record actions and images to a file (specify output directory, e.g., "data/recordings")')
    parser.add_argument('--norm-stats', type=str, default=None,
                       help='Path to norm_stats.json file for action unnormalization. Example: "assets/pi05_pick_blue_bottle_libero_downsample4x/your_hf_username/pick_blue_bottle_libero_downsample4x/norm_stats.json"')
    parser.add_argument('--auto-find-norm-stats', action='store_true',
                       help='Automatically search for norm_stats.json in common locations (assets/, checkpoints/)')
    args = parser.parse_args(argv)
    
    # 加载归一化统计信息
    norm_stats_loaded = False
    if args.norm_stats:
        norm_stats_loaded = load_norm_stats(args.norm_stats)
    elif args.auto_find_norm_stats:
        # 自动查找 norm_stats.json
        search_paths = [
            pathlib.Path("assets") / "pi05_pick_blue_bottle_libero_downsample4x" / "your_hf_username" / "pick_blue_bottle_libero_downsample4x" / "norm_stats.json",
            pathlib.Path("assets") / "pi05_libero" / "your_hf_username" / "pick_blue_bottle_libero_downsample4x" / "norm_stats.json",
            pathlib.Path("checkpoints") / "pi05_pick_blue_bottle_libero_downsample4x" / "*" / "assets" / "*" / "norm_stats.json",
        ]
        for search_path in search_paths:
            # 处理通配符
            if "*" in str(search_path):
                import glob
                matches = glob.glob(str(search_path))
                for match in matches:
                    if pathlib.Path(match).exists():
                        norm_stats_loaded = load_norm_stats(match)
                        if norm_stats_loaded:
                            break
            else:
                if search_path.exists():
                    norm_stats_loaded = load_norm_stats(search_path)
                    if norm_stats_loaded:
                        break
        if not norm_stats_loaded:
            print("⚠️  Could not auto-find norm_stats.json, please specify --norm-stats")
    
    if not norm_stats_loaded:
        print("ℹ️  Actions will not be unnormalized (no norm stats loaded)")

    # 初始化记录功能
    global recording_enabled, recording_dir, recording_file, step_count
    if args.record:
        recording_enabled = True
        recording_dir = pathlib.Path(args.record)
        recording_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recording_file = recording_dir / f"actions_{timestamp}.txt"
        print(f"📝 Recording enabled: {recording_file}")
        # 写入文件头
        with open(recording_file, 'w') as f:
            f.write(f"# Action and Image Recording\n")
            f.write(f"# Started at: {datetime.now().isoformat()}\n")
            f.write(f"# Format: step, action_shape, action_values, image_shape, image_mean, image_std\n")
            f.write(f"# {'='*80}\n\n")
        step_count = 0

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
            
            # 预编译（预热）推理
            warmup_inference(client)
            
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
                        if len(actions) > 0:
                            action_first_normalized = actions[0]  # 获取第一个时间步的完整action（归一化后）
                            print(f"   Actions (normalized, first step): {action_first_normalized}")
                            print(f"   Action dimension: {len(action_first_normalized)}")
                            
                            # 反归一化 action
                            action_first_unnormalized = unnormalize_action(action_first_normalized)
                            print(f"   Actions (unnormalized, first step): {action_first_unnormalized}")
                            if action_norm_stats is not None:
                                print(f"   🔍 Unnormalization applied: use_quantile={use_quantile_norm}")
                            else:
                                print(f"   ⚠️  Unnormalization NOT applied (action_norm_stats is None)")
                            
                            # 特别检查gripper值（如果有第8维）
                            if len(action_first_unnormalized) >= 8:
                                gripper_value = action_first_unnormalized[7]
                                print(f"   🔍 Gripper value (unnormalized, dim 7): {gripper_value:.6f}")
                                if gripper_value > 0.02:
                                    print(f"      → Will be set to 1.0 (close)")
                                else:
                                    print(f"      → Will remain as {gripper_value:.6f} (open or small value)")
                            elif len(action_first_unnormalized) == 7:
                                print(f"   ⚠️  Action only has 7 dims (no gripper dimension)")
                            else:
                                print(f"   ⚠️  Unexpected action dimension: {len(action_first_unnormalized)}")
                        
                        # 记录动作和图片信息（测试模式）
                        if recording_enabled and recording_file:
                            step_count += 1
                            action_shape = actions.shape
                            action_values_normalized = action_first_normalized.tolist() if len(actions) > 0 else []
                            action_values_unnormalized = action_first_unnormalized.tolist() if len(actions) > 0 else []
                            image_shape = base_img_fixed.shape
                            image_mean = float(np.mean(base_img_fixed))
                            image_std = float(np.std(base_img_fixed))
                            
                            with open(recording_file, 'a') as f:
                                f.write(f"Step {step_count}:\n")
                                f.write(f"  Action shape: {action_shape}\n")
                                f.write(f"  Action (normalized): {action_values_normalized}\n")
                                f.write(f"  Action (unnormalized): {action_values_unnormalized}\n")
                                f.write(f"  Image shape: {image_shape}\n")
                                f.write(f"  Image mean: {image_mean:.2f}, std: {image_std:.2f}\n")
                                f.write(f"\n")
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

    # 预编译（预热）推理
    warmup_inference(client)

    rclpy.init()
    node = PoseSubscriber(args.topic, args.action_topic, args.joint_states_topic, args.gripper_topic)

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
                        action_first_normalized = actions[0]  # 获取第一个时间步的完整action（归一化后）
                        print(f"   Actions (normalized, first step): {action_first_normalized}")
                        print(f"   Action dimension: {len(action_first_normalized)}")
                        
                        # 反归一化 action
                        action_first_unnormalized = unnormalize_action(action_first_normalized)
                        print(f"   Actions (unnormalized, first step): {action_first_unnormalized}")
                        if action_norm_stats is not None:
                            print(f"   🔍 Unnormalization applied: use_quantile={use_quantile_norm}")
                        else:
                            print(f"   ⚠️  Unnormalization NOT applied (action_norm_stats is None)")
                        
                        # 特别检查gripper值（如果有第8维）
                        if len(action_first_unnormalized) >= 8:
                            gripper_value = action_first_unnormalized[7]
                            print(f"   🔍 Gripper value (unnormalized, dim 7): {gripper_value:.6f}")
                            if gripper_value > 0.02:
                                print(f"      → Will be set to 1.0 (close)")
                            else:
                                print(f"      → Will remain as {gripper_value:.6f} (open or small value)")
                        elif len(action_first_unnormalized) == 7:
                            print(f"   ⚠️  Action only has 7 dims (no gripper dimension)")
                        else:
                            print(f"   ⚠️  Unexpected action dimension: {len(action_first_unnormalized)}")
                        
                        # 记录动作和图片信息（ROS2 模式）
                        if recording_enabled and recording_file:
                            step_count += 1
                            action_shape = actions.shape
                            action_values_normalized = action_first_normalized.tolist()
                            action_values_unnormalized = action_first_unnormalized.tolist()
                            image_shape = base_img_fixed.shape
                            image_mean = float(np.mean(base_img_fixed))
                            image_std = float(np.std(base_img_fixed))
                            
                            with open(recording_file, 'a') as f:
                                f.write(f"Step {step_count}:\n")
                                f.write(f"  Action shape: {action_shape}\n")
                                f.write(f"  Action (normalized): {action_values_normalized}\n")
                                f.write(f"  Action (unnormalized): {action_values_unnormalized}\n")
                                f.write(f"  Image shape: {image_shape}\n")
                                f.write(f"  Image mean: {image_mean:.2f}, std: {image_std:.2f}\n")
                                f.write(f"\n")
                        
                        # 发布动作到 ROS2 话题（会在 publish_action 内部进行反归一化）
                        if args.publish_actions:
                            node.publish_action(actions)  # 传入完整的 actions（publish_action 会处理）
                    else:
                        print("   Empty action chunk")
                else:
                    print("⚠️  No actions in response")
                    
            except Exception as e:
                print(f"❌ Inference error: {e}")
                import traceback
                traceback.print_exc()

            rclpy.spin_once(node, timeout_sec=0.1)
            time.sleep(2)   # 2 Hz (avoid spamming server)

    except KeyboardInterrupt:
        print("\n🛑 Interrupted")

    finally:
        if args.show_camera and CV2_AVAILABLE:
            cv2.destroyAllWindows()
        if camera_manager:
            camera_manager.stop()
        # if recording_enabled and recording_file:
        #     with open(recording_file, 'a') as f:
        #         f.write(f"# {'='*80}\n")
        #         f.write(f"# Recording ended at: {datetime.now().isoformat()}\n")
        #         f.write(f"# Total steps: {step_count}\n")
        #     print(f"📝 Recording saved: {recording_file} ({step_count} steps)")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

