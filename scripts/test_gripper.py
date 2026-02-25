#!/usr/bin/env python3
"""
测试夹爪开合脚本

功能：
- 前7个关节维度保持为0（机械臂不动）
- 第8个维度（gripper）在0和1之间交替，用于测试夹爪开合

使用方法:
    python scripts/test_gripper.py [--action_topic <topic>] [--interval <seconds>] [--cycles <num>]
"""

import argparse
import sys
import time
import numpy as np

# ROS2 支持
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float64MultiArray
    ROS2_AVAILABLE = True
except ImportError:
    rclpy = None
    Node = None
    Float64MultiArray = None
    ROS2_AVAILABLE = False
    print("⚠️  ROS2 not available. Please install ROS2 and source the setup script.")
    print("   Example: source /opt/ros/humble/setup.bash")


class GripperTester(Node):
    """夹爪测试节点"""
    
    def __init__(self, action_topic: str = "/libero/actions", interval: float = 2.0):
        """
        初始化夹爪测试节点
        
        Args:
            action_topic: 发布action的ROS2话题
            interval: 每次切换的间隔时间（秒）
        """
        super().__init__('gripper_tester')
        self.action_topic = action_topic
        self.interval = interval
        
        # 创建publisher
        self.action_publisher = self.create_publisher(
            Float64MultiArray,
            action_topic,
            10
        )
        
        self.get_logger().info(f'✅ Gripper tester initialized')
        self.get_logger().info(f'   Action topic: {action_topic}')
        self.get_logger().info(f'   Switch interval: {interval:.2f} seconds')
        self.get_logger().info(f'   Action format: [0, 0, 0, 0, 0, 0, 0, <gripper>]')
        self.get_logger().info(f'   Gripper will alternate between 0 (open) and 1 (close)')
    
    def publish_action(self, gripper_value: float):
        """
        发布action
        
        Args:
            gripper_value: 夹爪值 (0=开, 1=闭)
        """
        # 创建action: 前7个维度为0，第8个维度为gripper值
        action = np.zeros(8, dtype=np.float64)
        action[7] = gripper_value  # 第8个维度（索引7）是gripper
        
        # 创建消息
        msg = Float64MultiArray()
        msg.data = action.tolist()
        
        # 发布
        self.action_publisher.publish(msg)
        
        gripper_state = "OPEN" if gripper_value == 0.0 else "CLOSE"
        self.get_logger().info(
            f'📤 Published action: joints=[0,0,0,0,0,0,0], gripper={gripper_value:.1f} ({gripper_state})'
        )
    
    def run_test(self, cycles: int = None):
        """
        运行测试循环
        
        Args:
            cycles: 循环次数，如果为None则无限循环
        """
        self.get_logger().info('🚀 Starting gripper test...')
        self.get_logger().info('   Press Ctrl+C to stop')
        
        if cycles is not None:
            self.get_logger().info(f'   Will run {cycles} cycles (each cycle = open + close)')
        
        try:
            cycle_count = 0
            while True:
                # 检查是否达到指定循环次数
                if cycles is not None and cycle_count >= cycles:
                    self.get_logger().info(f'✅ Completed {cycles} cycles. Stopping.')
                    break
                
                # 打开夹爪 (0)
                self.publish_action(0.0)
                time.sleep(self.interval)
                
                # 关闭夹爪 (1)
                self.publish_action(1.0)
                time.sleep(self.interval)
                
                cycle_count += 1
                if cycles is None:
                    self.get_logger().info(f'   Cycle {cycle_count} completed (continuing...)')
                else:
                    self.get_logger().info(f'   Cycle {cycle_count}/{cycles} completed')
        
        except KeyboardInterrupt:
            self.get_logger().info('\n⚠️  Interrupted by user. Stopping...')
            # 最后发送一次0，确保夹爪打开（安全）
            self.get_logger().info('   Sending final OPEN command for safety...')
            self.publish_action(0.0)
            time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(
        description='Test gripper open/close on real robot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 使用默认设置（无限循环，每2秒切换）
  python scripts/test_gripper.py

  # 指定action话题
  python scripts/test_gripper.py --action_topic /robot/action

  # 指定切换间隔为3秒
  python scripts/test_gripper.py --interval 3.0

  # 运行10个循环后停止
  python scripts/test_gripper.py --cycles 10

  # 组合使用
  python scripts/test_gripper.py --action_topic /action --interval 1.5 --cycles 5
        """
    )
    
    parser.add_argument(
        '--action_topic',
        type=str,
        default='/libero/actions',
        help='ROS2 topic to publish actions to (default: /libero/actions)'
    )
    
    parser.add_argument(
        '--interval',
        type=float,
        default=2.0,
        help='Time interval between gripper state changes in seconds (default: 2.0)'
    )
    
    parser.add_argument(
        '--cycles',
        type=int,
        default=None,
        help='Number of open/close cycles to run (default: infinite)'
    )
    
    args = parser.parse_args()
    
    # 检查ROS2是否可用
    if not ROS2_AVAILABLE:
        print("❌ ROS2 is not available. Cannot run gripper test.")
        print("\nPlease ensure:")
        print("  1. ROS2 is installed")
        print("  2. ROS2 environment is sourced (e.g., source /opt/ros/humble/setup.bash)")
        print("  3. Required ROS2 packages are installed")
        sys.exit(1)
    
    # 初始化ROS2
    rclpy.init()
    
    try:
        # 创建节点
        node = GripperTester(
            action_topic=args.action_topic,
            interval=args.interval
        )
        
        # 等待一下，确保publisher已连接
        time.sleep(0.5)
        
        # 运行测试
        node.run_test(cycles=args.cycles)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # 清理
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

