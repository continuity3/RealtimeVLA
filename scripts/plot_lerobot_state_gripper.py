#!/usr/bin/env python3
"""
从转换完成的LeRobot数据集中提取state的gripper值并绘制折线图

提取训练数据集中state的最后一维（gripper值），这些值来自right_gripper_values

使用方法:
    python scripts/plot_lerobot_state_gripper.py --repo-id your_hf_username/pick_blue_bottle_libero_downsample4x --num-episodes 5
"""

import argparse
import sys
import os

import numpy as np

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    print("⚠️  lerobot not available. Install with: pip install lerobot")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not available. Install with: pip install matplotlib")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def extract_state_gripper_from_lerobot(
    repo_id: str,
    num_episodes: int = 5,
    threshold: float = 0.8,
    output_path: str = None,
):
    """
    从LeRobot数据集中提取state的gripper值（最后一维）并绘制折线图
    
    Args:
        repo_id: LeRobot数据集repo_id
        num_episodes: 要绘制的episode数量
        threshold: gripper值的阈值（用于标记）
        output_path: 输出图片路径（如果为None，则显示图片）
    """
    if not LEROBOT_AVAILABLE:
        print("❌ lerobot不可用，无法加载数据集")
        return
    
    if not MATPLOTLIB_AVAILABLE:
        print("❌ matplotlib不可用，无法绘图")
        return
    
    print(f"📂 加载数据集: {repo_id}")
    
    try:
        dataset = LeRobotDataset(repo_id)
    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        return
    
    # 获取episode数量
    num_episodes_in_dataset = len(dataset)
    num_episodes_to_plot = min(num_episodes, num_episodes_in_dataset)
    
    print(f"📊 数据集包含 {num_episodes_in_dataset} 个episodes，将绘制前 {num_episodes_to_plot} 个")
    
    # 收集数据
    state_gripper_data = []  # 每个episode的state gripper值
    episode_lengths = []
    
    for ep_idx in range(num_episodes_to_plot):
        episode_state_gripper = []
        
        try:
            # 获取episode信息
            if hasattr(dataset, 'episode_data_index'):
                episode_index = dataset.episode_data_index
                if isinstance(episode_index, dict):
                    if 'from' in episode_index and 'to' in episode_index:
                        from_indices = episode_index['from']
                        to_indices = episode_index['to']
                        
                        # 转换为numpy数组（如果是tensor）
                        if TORCH_AVAILABLE and isinstance(from_indices, torch.Tensor):
                            from_indices = from_indices.cpu().numpy()
                        if TORCH_AVAILABLE and isinstance(to_indices, torch.Tensor):
                            to_indices = to_indices.cpu().numpy()
                        
                        if ep_idx < len(from_indices):
                            start_idx = int(from_indices[ep_idx])
                            end_idx = int(to_indices[ep_idx])
                            
                            print(f"  Episode {ep_idx}: frames {start_idx} to {end_idx} (length: {end_idx - start_idx})")
                            
                            # 遍历episode的所有帧
                            for frame_idx in range(start_idx, end_idx):
                                try:
                                    frame = dataset[frame_idx]
                                    if isinstance(frame, dict):
                                        # 提取state的最后一维（gripper值）
                                        # state应该是8维：[7个关节位置, 1个gripper值]
                                        for key in ["state", "observation.state", "observation/state"]:
                                            if key in frame:
                                                state = frame[key]
                                                # 转换为numpy数组
                                                if TORCH_AVAILABLE and isinstance(state, torch.Tensor):
                                                    state = state.cpu().numpy()
                                                else:
                                                    state = np.array(state)
                                                
                                                # state应该是8维，最后一维是gripper值
                                                if state.ndim == 1 and len(state) >= 8:
                                                    gripper_value = float(state[7])  # 索引7是最后一维
                                                    episode_state_gripper.append(gripper_value)
                                                break
                                except Exception as e:
                                    # 跳过无法访问的帧
                                    continue
                        else:
                            print(f"  ⚠️  Episode {ep_idx}: 索引超出范围")
                    else:
                        print(f"  ⚠️  Episode {ep_idx}: episode_data_index格式不正确")
                else:
                    print(f"  ⚠️  Episode {ep_idx}: episode_data_index不是字典")
            else:
                print(f"  ⚠️  Episode {ep_idx}: 无法获取episode_data_index")
                
        except Exception as e:
            print(f"  ⚠️  访问episode {ep_idx} 数据时出错: {e}")
            import traceback
            traceback.print_exc()
        
        state_gripper_data.append(episode_state_gripper)
        
        # 记录episode长度
        if episode_state_gripper:
            episode_lengths.append(len(episode_state_gripper))
            print(f"  ✅ Episode {ep_idx}: 提取了 {len(episode_state_gripper)} 个state gripper值")
        else:
            episode_lengths.append(0)
            print(f"  ⚠️  Episode {ep_idx}: 未找到gripper数据")
    
    # 创建子图
    fig, axes = plt.subplots(num_episodes_to_plot, 1, figsize=(12, 3 * num_episodes_to_plot))
    
    if num_episodes_to_plot == 1:
        axes = [axes]
    
    fig.suptitle(
        f'State Gripper Values from LeRobot Dataset (First {num_episodes_to_plot} episodes, Threshold={threshold})',
        fontsize=14,
        fontweight='bold'
    )
    
    for ep_idx in range(num_episodes_to_plot):
        ax = axes[ep_idx]
        gripper_data = state_gripper_data[ep_idx]
        
        if len(gripper_data) > 0:
            gripper_data = np.array(gripper_data)
            valid_mask = ~np.isnan(gripper_data)
            valid_data = gripper_data[valid_mask]
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_data) > 0:
                # 绘制折线图
                ax.plot(valid_indices, valid_data, 'b-', linewidth=1.5, label='State Gripper Value', alpha=0.7)
                
                # 标记大于阈值的点
                above_threshold_mask = valid_data > threshold
                if np.any(above_threshold_mask):
                    ax.scatter(
                        valid_indices[above_threshold_mask],
                        valid_data[above_threshold_mask],
                        c='red',
                        s=30,
                        marker='o',
                        label=f'> {threshold}',
                        zorder=5
                    )
                
                # 添加阈值线
                ax.axhline(y=threshold, color='r', linestyle='--', linewidth=1, alpha=0.5, label=f'Threshold={threshold}')
                
                # 设置标签和标题
                ax.set_xlabel('Time Step', fontsize=10)
                ax.set_ylabel('State Gripper Value (0=open, 1=closed)', fontsize=10)
                ax.set_title(
                    f'Episode {ep_idx} - State Gripper (from right_gripper_values)\n'
                    f'Total: {len(valid_data)}, >{threshold}: {np.sum(above_threshold_mask)} '
                    f'({np.sum(above_threshold_mask)/len(valid_data)*100:.2f}%)',
                    fontsize=10,
                    fontweight='bold'
                )
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right', fontsize=8)
                
                # 设置y轴范围
                y_min = min(0, np.min(valid_data) * 0.1)
                y_max = max(1.0, np.max(valid_data) * 1.1)
                ax.set_ylim(y_min, y_max)
            else:
                ax.text(0.5, 0.5, f'Episode {ep_idx} - State Gripper\nNo valid data',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Episode {ep_idx} - State Gripper', fontsize=10)
        else:
            ax.text(0.5, 0.5, f'Episode {ep_idx} - State Gripper\nNo data',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Episode {ep_idx} - State Gripper', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Image saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    # 打印统计信息
    print("\n" + "=" * 80)
    print("📊 统计信息:")
    print("=" * 80)
    for ep_idx in range(num_episodes_to_plot):
        if state_gripper_data[ep_idx]:
            state_data = np.array(state_gripper_data[ep_idx])
            valid_state = state_data[~np.isnan(state_data)]
            if len(valid_state) > 0:
                print(f"\nEpisode {ep_idx}:")
                print(f"  State Gripper (from right_gripper_values):")
                print(f"    - 总样本数: {len(valid_state)}")
                print(f"    - 最小值: {np.min(valid_state):.6f}")
                print(f"    - 最大值: {np.max(valid_state):.6f}")
                print(f"    - 平均值: {np.mean(valid_state):.6f}")
                print(f"    - 中位数: {np.median(valid_state):.6f}")
                print(f"    - 标准差: {np.std(valid_state):.6f}")
                print(f"    - >{threshold}: {np.sum(valid_state > threshold)}/{len(valid_state)} ({np.sum(valid_state > threshold)/len(valid_state)*100:.2f}%)")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="从转换完成的LeRobot数据集中提取state的gripper值并绘制折线图",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 绘制前5个episode的state gripper值
  python scripts/plot_lerobot_state_gripper.py --repo-id your_hf_username/pick_blue_bottle_libero_downsample4x --num-episodes 5

  # 保存图片
  python scripts/plot_lerobot_state_gripper.py --repo-id your_hf_username/pick_blue_bottle_libero_downsample4x --num-episodes 5 --output state_gripper_plots.png
        """
    )
    
    parser.add_argument(
        '--repo-id',
        type=str,
        required=True,
        help='LeRobot数据集repo_id'
    )
    
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=5,
        help='要绘制的episode数量（默认：5）'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.8,
        help='gripper值的阈值（默认：0.8）'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='图片输出路径（如果指定，则保存图片；否则显示）'
    )
    
    args = parser.parse_args()
    
    extract_state_gripper_from_lerobot(
        repo_id=args.repo_id,
        num_episodes=args.num_episodes,
        threshold=args.threshold,
        output_path=args.output,
    )


if __name__ == '__main__':
    main()















