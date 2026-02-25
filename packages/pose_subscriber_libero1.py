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

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped
except Exception:
    rclpy = None


# =========================
# Dummy inputs (TEST ONLY)
# =========================

IMG_SIZE = 224

# Fake images (uint8, HWC format for LIBERO policy)
# LIBERO expects images in HWC format: (224, 224, 3)
base_img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
wrist_img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

# LIBERO expects 8-dim joint state
state = [0.0] * 8

task_instruction = "Pick up the object."


# =========================
# ROS2 Node
# =========================

class PoseSubscriber(Node):
    def __init__(self, topic: str):
        super().__init__('pose_subscriber_libero')
        self.get_logger().info(f'Subscribing to: {topic}')
        self.create_subscription(
            PoseStamped,
            topic,
            self.callback,
            10,
        )

    def callback(self, msg: PoseStamped):
        # Only log, not used for control yet
        p = msg.pose.position
        self.get_logger().info(
            f"Pose: ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})"
        )

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
    parser.add_argument('--topic', '-t', default='/pose', help='ROS2 topic to subscribe to')

    parser.add_argument('--host', default='localhost', help='Policy server host (default: localhost)')
    parser.add_argument('--port', type=int, default=8000, help='Policy server port (default: 8000)')
    parser.add_argument('--test-mode', action='store_true', 
                       help='Run in test mode without ROS2 (for testing policy server connection)')
    args = parser.parse_args(argv)

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
                base_img_fixed = ensure_hwc_uint8(base_img)
                wrist_img_fixed = ensure_hwc_uint8(wrist_img)
                
                # LIBERO policy expects images in HWC format with specific keys
                observation = {
                    "observation/state": np.array(state, dtype=np.float32),  # 8 维
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
                        print(f"   Actions (first step, first 7 dims): {actions[0, :7] if len(actions) > 0 and actions.shape[1] >= 7 else actions[0]}")
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
    node = PoseSubscriber(args.topic)

    print("✅ TEST MODE: Sending fake LIBERO observation")

    try:
        while True:
            base_img_fixed = ensure_hwc_uint8(base_img)
            wrist_img_fixed = ensure_hwc_uint8(wrist_img)

            # LIBERO policy expects images in HWC format with specific keys
            observation = {
                "observation/state": np.array(state, dtype=np.float32),  # 8 维
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
                            print(f"   Actions (first step, first 7 dims): {actions[0, :7]}")
                        else:
                            print(f"   Actions (first step): {actions[0]}")
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
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
