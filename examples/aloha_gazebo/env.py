"""
Gazebo environment wrapper for ALOHA robot simulation.

This module provides an environment interface for running ALOHA policies
in Gazebo simulation through ROS2.
"""
import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.clock import Clock
    from sensor_msgs.msg import Image
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import Float64MultiArray
    from cv_bridge import CvBridge
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    rclpy = None
    Clock = None
    CvBridge = None


class GazeboAlohaEnvironment(_environment.Environment):
    """An environment for an ALOHA robot in Gazebo simulation via ROS2."""

    def __init__(
        self,
        image_topic_high: str = "/camera_high/image_raw",
        image_topic_low: str = "/camera_low/image_raw",
        image_topic_left_wrist: str = "/camera_left_wrist/image_raw",
        image_topic_right_wrist: str = "/camera_right_wrist/image_raw",
        state_topic: str = "/aloha/joint_states",
        action_topic: str = "/aloha/joint_commands",
        render_height: int = 224,
        render_width: int = 224,
    ) -> None:
        if not ROS2_AVAILABLE:
            raise RuntimeError("ROS2 is required for Gazebo simulation. Please install rclpy and related packages.")
        
        self._render_height = render_height
        self._render_width = render_width
        self._image_topic_high = image_topic_high
        self._image_topic_low = image_topic_low
        self._image_topic_left_wrist = image_topic_left_wrist
        self._image_topic_right_wrist = image_topic_right_wrist
        self._state_topic = state_topic
        self._action_topic = action_topic
        
        # Initialize ROS2
        if not rclpy.ok():
            rclpy.init()
        
        self._node = Node('gazebo_aloha_env')
        self._bridge = CvBridge()
        
        # Storage for latest observations
        self._latest_images = {
            "cam_high": None,
            "cam_low": None,
            "cam_left_wrist": None,
            "cam_right_wrist": None,
        }
        self._latest_state = None
        self._state_received = False
        
        # Subscribers
        self._node.create_subscription(
            Image, self._image_topic_high,
            lambda msg: self._image_callback(msg, "cam_high"), 10
        )
        self._node.create_subscription(
            Image, self._image_topic_low,
            lambda msg: self._image_callback(msg, "cam_low"), 10
        )
        self._node.create_subscription(
            Image, self._image_topic_left_wrist,
            lambda msg: self._image_callback(msg, "cam_left_wrist"), 10
        )
        self._node.create_subscription(
            Image, self._image_topic_right_wrist,
            lambda msg: self._image_callback(msg, "cam_right_wrist"), 10
        )
        self._node.create_subscription(
            Float64MultiArray, self._state_topic,
            self._state_callback, 10
        )
        
        # Publisher for actions
        self._action_pub = self._node.create_publisher(
            Float64MultiArray, self._action_topic, 10
        )
        
        self._episode_started = False

    def _image_callback(self, msg: Image, cam_name: str) -> None:
        """Callback for image messages."""
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self._bridge.imgmsg_to_cv2(msg, "rgb8")
            # Resize and convert to CHW format
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(cv_image, self._render_height, self._render_width)
            )
            # Convert from HWC to CHW
            img = np.transpose(img, (2, 0, 1))
            self._latest_images[cam_name] = img
        except Exception as e:
            self._node.get_logger().warn(f"Error processing image from {cam_name}: {e}")

    def _state_callback(self, msg: Float64MultiArray) -> None:
        """Callback for joint state messages."""
        self._latest_state = np.array(msg.data, dtype=np.float32)
        self._state_received = True

    @override
    def reset(self) -> None:
        """Reset the environment."""
        # Wait for initial observations
        self._node.get_logger().info("Waiting for initial observations...")
        timeout = 5.0  # 5 seconds timeout
        clock = Clock()
        start_time = clock.now()
        
        while (clock.now() - start_time).nanoseconds / 1e9 < timeout:
            rclpy.spin_once(self._node, timeout_sec=0.1)
            if self._state_received and all(img is not None for img in self._latest_images.values()):
                break
        
        if not self._state_received:
            self._node.get_logger().warn("State not received during reset")
        if not all(img is not None for img in self._latest_images.values()):
            missing = [k for k, v in self._latest_images.items() if v is None]
            self._node.get_logger().warn(f"Missing images from: {missing}")
        
        self._episode_started = True

    @override
    def is_episode_complete(self) -> bool:
        """Check if episode is complete."""
        # For now, episodes never complete automatically
        # You can add logic here to detect episode completion
        return False

    @override
    def get_observation(self) -> dict:
        """Get the current observation."""
        if not self._episode_started:
            raise RuntimeError("Episode not started. Call reset() first.")
        
        # Spin to get latest messages
        rclpy.spin_once(self._node, timeout_sec=0.01)
        
        # Use latest received images, or zeros if not available
        images = {}
        for cam_name in ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]:
            if self._latest_images[cam_name] is not None:
                images[cam_name] = self._latest_images[cam_name]
            else:
                # Use zeros if image not available
                images[cam_name] = np.zeros((3, self._render_height, self._render_width), dtype=np.uint8)
        
        # Use latest state or zeros if not available
        state = self._latest_state if self._latest_state is not None else np.zeros(14, dtype=np.float32)
        
        return {
            "state": state,
            "images": images,
        }

    @override
    def apply_action(self, action: dict) -> None:
        """Apply action to the environment."""
        if not self._episode_started:
            raise RuntimeError("Episode not started. Call reset() first.")
        
        # Publish action to Gazebo
        action_msg = Float64MultiArray()
        action_msg.data = action["actions"].flatten().tolist()
        self._action_pub.publish(action_msg)
        
        # Spin to process callbacks
        rclpy.spin_once(self._node, timeout_sec=0.01)

    def shutdown(self) -> None:
        """Shutdown the ROS2 node."""
        if self._node is not None:
            self._node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

