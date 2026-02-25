#!/usr/bin/env python3
"""
ROS2 Client for OpenPI LIBERO policy (FINAL VERSION)

Assumptions:
- Policy server returns UNNORMALIZED physical actions
- Action format is strictly 7-dim:
  [dx, dy, dz, rx, ry, rz, gripper]
"""

import argparse
import pathlib
import sys
import time
import numpy as np
from datetime import datetime

from openpi_client import websocket_client_policy

# -------------------------
# Optional dependencies
# -------------------------

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float64MultiArray, Float64
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import PoseStamped
    ROS2_AVAILABLE = True
except Exception:
    rclpy = None
    ROS2_AVAILABLE = False


# -------------------------
# Configuration
# -------------------------

IMG_SIZE = 224
DEFAULT_STATE = np.zeros(8, dtype=np.float32)
TASK_INSTRUCTION = "Pick up the blue square"


# -------------------------
# ROS2 Node
# -------------------------

if ROS2_AVAILABLE:
    class PoseSubscriber(Node):
        def __init__(
            self,
            pose_topic: str,
            action_topic: str,
            joint_states_topic: str,
            gripper_topic: str,
        ):
            super().__init__("pose_subscriber_libero")

            self.get_logger().info(f"Publishing actions to: {action_topic}")

            self.action_publisher = self.create_publisher(
                Float64MultiArray, action_topic, 10
            )

            self.latest_joint_positions = None
            self.latest_gripper_value = 0.0
            self.latest_state = DEFAULT_STATE.copy()

            self.create_subscription(
                JointState, joint_states_topic, self.joint_cb, 10
            )
            self.create_subscription(
                Float64, gripper_topic, self.gripper_cb, 10
            )
            self.create_subscription(
                PoseStamped, pose_topic, self.pose_cb, 10
            )

        def pose_cb(self, msg: PoseStamped):
            pass  # only for debug

        def joint_cb(self, msg: JointState):
            if len(msg.position) >= 14:
                self.latest_joint_positions = np.array(
                    msg.position[7:14], dtype=np.float32
                )
                self._update_state()

        def gripper_cb(self, msg: Float64):
            self.latest_gripper_value = float(msg.data)
            self._update_state()

        def _update_state(self):
            if self.latest_joint_positions is not None:
                self.latest_state = np.concatenate(
                    [
                        self.latest_joint_positions,
                        np.array([self.latest_gripper_value], dtype=np.float32),
                    ]
                )

        # -------- FINAL publish_action --------
        def publish_action(self, action: np.ndarray):
            if action is None:
                return

            action = np.asarray(action).reshape(-1)

            if action.shape[0] != 7:
                self.get_logger().warn(
                    f"Expected 7-dim action, got {action.shape}"
                )
                return

            action_out = action.copy()

            raw_gripper = float(action_out[6])
            self.get_logger().info(
                f"🔍 Raw gripper (unnormalized): {raw_gripper:.4f}"
            )

            if raw_gripper > 0.5:
                action_out[6] = 1.0
                self.get_logger().info("✅ Gripper CLOSE")
            else:
                action_out[6] = 0.0
                self.get_logger().info("📌 Gripper OPEN")

            msg = Float64MultiArray()
            msg.data = action_out.astype(np.float64).tolist()
            self.action_publisher.publish(msg)

else:
    class PoseSubscriber:
        def __init__(self, *args, **kwargs):
            pass

        def publish_action(self, action: np.ndarray):
            pass


# -------------------------
# Main loop
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--topic", default="/pose")
    parser.add_argument("--action-topic", default="/libero/actions")
    parser.add_argument("--joint-states-topic", default="/joint_states")
    parser.add_argument("--gripper-topic", default="/gripper/feedback_R")
    args = parser.parse_args()

    client = websocket_client_policy.WebsocketClientPolicy(
        host=args.host, port=args.port
    )

    if not ROS2_AVAILABLE:
        print("❌ ROS2 not available")
        return

    rclpy.init()
    node = PoseSubscriber(
        args.topic,
        args.action_topic,
        args.joint_states_topic,
        args.gripper_topic,
    )

    try:
        while rclpy.ok():
            obs = {
                "observation/state": node.latest_state,
                "observation/image": np.zeros((224, 224, 3), np.uint8),
                "observation/wrist_image": np.zeros((224, 224, 3), np.uint8),
                "prompt": TASK_INSTRUCTION,
            }

            result = client.infer(obs)
            actions = result.get("actions")

            if actions is not None and len(actions) > 0:
                node.publish_action(actions[0])

            rclpy.spin_once(node, timeout_sec=0.1)
            time.sleep(1.0)

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
