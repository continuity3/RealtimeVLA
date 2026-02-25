#!/usr/bin/env python3
"""
Real-time inference script for LIBERO robot following the official example structure.

This script connects to a policy server and performs real-time inference with:
- RealSense camera for images
- ROS2 for joint states
- Action recording and analysis
- Input/output normalization logging
"""

import argparse
import collections
import dataclasses
import logging
import pathlib
import sys
import time
from typing import Optional

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro

# RealSense 相机支持
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

# ROS2 支持
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float64MultiArray
    from sensor_msgs.msg import JointState
    ROS2_AVAILABLE = True
except ImportError:
    rclpy = None
    Node = None
    Float64MultiArray = None
    JointState = None
    ROS2_AVAILABLE = False


IMG_SIZE = 224
STATE_DIM = 8  # 7 joint positions + 1 gripper


@dataclasses.dataclass
class Args:
    """Arguments for real-time inference."""
    
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "localhost"
    port: int = 8000
    
    #################################################################################################################
    # Task parameters
    #################################################################################################################
    prompt: str = "Pick up the blue square and place it in the blue tray."
    
    #################################################################################################################
    # ROS2 parameters
    #################################################################################################################
    joint_states_topic: str = "/joint_states"
    action_topic: str = "/libero/actions"
    publish_actions: bool = True
    
    #################################################################################################################
    # Recording parameters
    #################################################################################################################
    record_dir: Optional[str] = "data/realtime_inference"  # Directory to save recorded data
    save_images: bool = True  # Save images during recording
    save_normalization: bool = True  # Save normalization statistics
    
    #################################################################################################################
    # Camera parameters
    #################################################################################################################
    use_realsense: bool = True
    camera_serial: Optional[str] = None
    show_camera: bool = False  # Show camera feed in a window
    
    #################################################################################################################
    # Inference parameters
    #################################################################################################################
    inference_rate: float = 2.0  # Hz (inference frequency)
    test_mode: bool = False  # Test mode without ROS2


class RealSenseCamera:
    """RealSense camera manager."""
    
    def __init__(self, camera_serial: Optional[str] = None, width: int = 640, height: int = 480, fps: int = 30):
        if not REALSENSE_AVAILABLE:
            raise RuntimeError("pyrealsense2 is not available. Install with: pip install pyrealsense2")
        
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        
        # Import rs here (already imported in __init__ check)
        import pyrealsense2 as rs
        
        # Detect available cameras
        ctx = rs.context()
        devices = ctx.query_devices()
        
        logging.info(f"Found {len(devices)} RealSense device(s):")
        available_serials = []
        for dev in devices:
            serial = dev.get_info(rs.camera_info.serial_number)
            name = dev.get_info(rs.camera_info.name)
            logging.info(f"  - {name} (Serial: {serial})")
            available_serials.append(serial)
        
        if len(devices) == 0:
            raise RuntimeError("No RealSense cameras found!")
        
        if camera_serial is None:
            camera_serial = available_serials[0]
        
        # Initialize camera
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(camera_serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        
        pipeline.start(config)
        self.pipeline = pipeline
        logging.info(f"Started camera (Serial: {camera_serial})")
    
    def get_image(self) -> Optional[np.ndarray]:
        """Get image in RGB format (H, W, 3), uint8."""
        if self.pipeline is None:
            return None
        
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            color_frame = frames.get_color_frame()
            
            if color_frame:
                img = np.asanyarray(color_frame.get_data())
                # BGR -> RGB
                img = img[:, :, ::-1]
                return img
        except Exception as e:
            logging.warning(f"Error reading from camera: {e}")
        
        return None
    
    def stop(self):
        """Stop the camera."""
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
                logging.info("Stopped camera")
            except Exception as e:
                logging.warning(f"Error stopping camera: {e}")


class DataRecorder:
    """Records observations, actions, and normalization data."""
    
    def __init__(self, record_dir: str, save_images: bool = True, save_normalization: bool = True):
        self.record_dir = pathlib.Path(record_dir)
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.save_images = save_images
        self.save_normalization = save_normalization
        
        # Storage for recorded data
        self.observations = []
        self.actions = []
        self.raw_observations = []  # Before normalization
        self.raw_actions = []  # After unnormalization
        self.step_count = 0
        
        logging.info(f"Recording data to: {self.record_dir}")
    
    def record_step(self, raw_obs: dict, normalized_obs: dict, raw_action: np.ndarray, normalized_action: np.ndarray):
        """Record a single step."""
        self.step_count += 1
        
        # Store raw observations (before normalization)
        obs_data = {
            "state": raw_obs.get("observation/state", np.zeros(STATE_DIM)),
            "prompt": raw_obs.get("prompt", ""),
        }
        if self.save_images and "observation/image" in raw_obs:
            img_path = self.record_dir / f"image_{self.step_count:06d}.npy"
            np.save(img_path, raw_obs["observation/image"])
            obs_data["image_path"] = str(img_path.relative_to(self.record_dir))
        self.raw_observations.append(obs_data)
        
        # Store normalized observations (for reference)
        self.observations.append({
            "state": normalized_obs.get("state", np.zeros(STATE_DIM)),
        })
        
        # Store actions
        self.raw_actions.append(raw_action.copy())
        self.actions.append(normalized_action.copy())
    
    def save(self):
        """Save all recorded data."""
        logging.info(f"Saving {self.step_count} recorded steps...")
        
        # Save actions
        if len(self.actions) > 0:
            actions_array = np.array(self.actions)
            np.save(self.record_dir / "actions.npy", actions_array)
            logging.info(f"Saved actions: {actions_array.shape}")
            
            # Save raw actions (unnormalized)
            raw_actions_array = np.array(self.raw_actions)
            np.save(self.record_dir / "actions_raw.npy", raw_actions_array)
            logging.info(f"Saved raw actions: {raw_actions_array.shape}")
        
        # Save observations summary
        if len(self.raw_observations) > 0:
            states = np.array([obs["state"] for obs in self.raw_observations])
            np.save(self.record_dir / "states.npy", states)
            logging.info(f"Saved states: {states.shape}")
            
            # Save prompts
            prompts = [obs["prompt"] for obs in self.raw_observations]
            with open(self.record_dir / "prompts.txt", "w") as f:
                for i, prompt in enumerate(prompts):
                    f.write(f"{i}: {prompt}\n")
        
        # Save normalization statistics if requested
        if self.save_normalization and len(self.actions) > 0 and len(self.observations) > 0:
            actions_array = np.array(self.actions)
            states_array = np.array([obs["state"] for obs in self.observations])
            
            norm_stats = {
                "actions": {
                    "mean": np.mean(actions_array, axis=0),
                    "std": np.std(actions_array, axis=0),
                    "min": np.min(actions_array, axis=0),
                    "max": np.max(actions_array, axis=0),
                },
                "states": {
                    "mean": np.mean(states_array, axis=0),
                    "std": np.std(states_array, axis=0),
                    "min": np.min(states_array, axis=0),
                    "max": np.max(states_array, axis=0),
                },
            }
            
            np.save(self.record_dir / "normalization_stats.npy", norm_stats)
            logging.info("Saved normalization statistics")
        
        logging.info(f"✅ All data saved to: {self.record_dir}")


if ROS2_AVAILABLE:
    class ActionPublisher(Node):
        """ROS2 node for publishing actions (same node name as pose_subscriber_libero.py)."""
        
        def __init__(self, action_topic: str, joint_states_topic: str):
            # Use same node name as pose_subscriber_libero.py for compatibility
            super().__init__('pose_subscriber_libero')
            self.action_publisher = self.create_publisher(Float64MultiArray, action_topic, 10)
            self.create_subscription(JointState, joint_states_topic, self.joint_states_callback, 10)
            self.latest_state = np.zeros(STATE_DIM, dtype=np.float32)
            self.latest_action = None
            self.get_logger().info(f'Subscribing to joint_states topic: {joint_states_topic}')
            self.get_logger().info(f'Publishing actions to topic: {action_topic}')
        
        def joint_states_callback(self, msg: JointState):
            """Update joint state (same logic as pose_subscriber_libero.py)."""
            if len(msg.position) >= 14:
                # 提取第 7-13 维（索引 7-12，共 7 个值）
                joint_positions = np.array(msg.position[7:14], dtype=np.float32)
                # 添加夹爪维度（值为 0）
                self.latest_state = np.concatenate([joint_positions, np.array([0.0], dtype=np.float32)])
            else:
                self.get_logger().warn(f'JointState has {len(msg.position)} positions, expected at least 14')
        
        def publish_action(self, action: np.ndarray):
            """Publish action (same logic as pose_subscriber_libero.py)."""
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


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Preprocess image to match training format."""
    img = image_tools.convert_to_uint8(img)
    img = image_tools.resize_with_pad(img, IMG_SIZE, IMG_SIZE)
    return img


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO)
    
    # Initialize camera
    camera = None
    if args.use_realsense:
        # Try to import pyrealsense2 again (in case it's available in the runtime environment)
        try:
            import pyrealsense2 as rs
            runtime_realsense_available = True
        except ImportError:
            runtime_realsense_available = False
        
        if not runtime_realsense_available and not REALSENSE_AVAILABLE:
            logging.warning("RealSense not available. Install with: pip install pyrealsense2")
            logging.warning("Continuing without camera (using dummy images for testing)")
            args.use_realsense = False
        else:
            try:
                camera = RealSenseCamera(camera_serial=args.camera_serial)
            except Exception as e:
                logging.warning(f"Failed to initialize camera: {e}")
                logging.warning("Continuing without camera (using dummy images for testing)")
                camera = None
                args.use_realsense = False
    
    # Initialize ROS2 node
    node = None
    if not args.test_mode:
        if not ROS2_AVAILABLE:
            logging.warning("ROS2 not available. Automatically switching to test mode")
            args.test_mode = True
        else:
            rclpy.init()
            if args.publish_actions:
                node = ActionPublisher(args.action_topic, args.joint_states_topic)
    
    # Initialize data recorder
    recorder = DataRecorder(
        record_dir=args.record_dir,
        save_images=args.save_images,
        save_normalization=args.save_normalization,
    )
    
    # Connect to policy server
    logging.info(f"Connecting to policy server at {args.host}:{args.port}...")
    try:
        client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
        metadata = client.get_server_metadata()
        logging.info(f"✅ Connected to server. Metadata: {metadata}")
    except Exception as e:
        logging.error(f"Failed to connect to server: {e}")
        return
    
    # Inference loop
    logging.info(f"Starting inference with prompt: {args.prompt}")
    logging.info(f"Inference rate: {args.inference_rate} Hz")
    
    step_time = 1.0 / args.inference_rate
    action_plan = collections.deque()
    
    try:
        while True:
            loop_start = time.time()
            
            # Get image from camera
            if camera:
                img = camera.get_image()
                if img is None:
                    logging.warning("Failed to get image from camera")
                    time.sleep(0.1)
                    continue
                
                # Show camera feed if requested
                if args.show_camera:
                    try:
                        import cv2
                        cv2.imshow('Camera Feed', img[:, :, ::-1])  # RGB -> BGR
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    except ImportError:
                        pass
                
                # Preprocess images
                base_img = preprocess_image(img)
                wrist_img = base_img  # Use same image for wrist (or implement separate camera)
            else:
                # Use dummy images for testing
                base_img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                wrist_img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            
            # Get current state
            if node and hasattr(node, 'latest_state'):
                current_state = node.latest_state.copy()
            else:
                current_state = np.zeros(STATE_DIM, dtype=np.float32)
            
            # Prepare observation
            raw_observation = {
                "observation/image": base_img,
                "observation/wrist_image": wrist_img,
                "observation/state": current_state,
                "prompt": args.prompt,
            }
            
            # Get action chunk if needed
            if not action_plan:
                try:
                    result = client.infer(raw_observation)
                    action_chunk = result.get("actions")
                    
                    if action_chunk is not None and len(action_chunk) > 0:
                        # Use print for better visibility (like pose_subscriber_libero.py)
                        print(f"✅ Action chunk received: shape={action_chunk.shape}")
                        if action_chunk.shape[1] >= 7:
                            print(f"   Actions (first step, first 7 dims): {action_chunk[0, :7]}")
                        else:
                            print(f"   Actions (first step): {action_chunk[0]}")
                        sys.stdout.flush()  # Ensure output is printed immediately
                        
                        action_plan.extend(action_chunk)
                        
                        # Record this inference step
                        recorder.record_step(
                            raw_obs=raw_observation,
                            normalized_obs=result,  # Note: result may not contain normalized obs
                            raw_action=action_chunk[0],
                            normalized_action=action_chunk[0],  # Actions are already unnormalized
                        )
                    else:
                        logging.warning("No actions in response")
                        time.sleep(step_time)
                        continue
                except Exception as e:
                    logging.error(f"Inference error: {e}")
                    time.sleep(step_time)
                    continue
            
            # Execute action
            if action_plan:
                action = action_plan.popleft()
                
                # Publish action to ROS2
                if node and args.publish_actions:
                    node.publish_action(action)
                
                logging.debug(f"Action: {action[:3]}... (shape: {action.shape})")
            
            # Spin ROS2 node
            if node:
                rclpy.spin_once(node, timeout_sec=0.01)
            
            # Maintain inference rate
            elapsed = time.time() - loop_start
            if elapsed < step_time:
                time.sleep(step_time - elapsed)
    
    except KeyboardInterrupt:
        logging.info("\n🛑 Interrupted by user")
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
    finally:
        # Save recorded data
        recorder.save()
        
        # Cleanup
        if camera:
            camera.stop()
        if args.show_camera:
            try:
                import cv2
                cv2.destroyAllWindows()
            except ImportError:
                pass
        if node:
            node.destroy_node()
        if ROS2_AVAILABLE:
            rclpy.shutdown()


if __name__ == "__main__":
    tyro.cli(main)

