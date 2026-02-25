#!/usr/bin/env python3
"""
从训练数据集中提取gripper值并绘制折线图

查看训练数据（LeRobot格式）中gripper值的变化趋势

使用方法:
    python scripts/plot_training_data_gripper.py --repo-id your_hf_username/pick_blue_bottle_libero_downsample4x --num-episodes 5
"""

import argparse
import sys

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

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


def plot_training_data_gripper(
    repo_id: str,
    num_episodes: int = 5,
    threshold: float = 0.8,
    output_path: str = None,
    plot_state: bool = True,
    plot_action: bool = True,
):
    """
    从训练数据集中提取gripper值并绘制折线图
    
    Args:
        repo_id: LeRobot数据集repo_id
        num_episodes: 要绘制的episode数量
        threshold: gripper值的阈值（用于标记）
        output_path: 输出图片路径（如果为None，则显示图片）
        plot_state: 是否绘制state中的gripper值
        plot_action: 是否绘制action中的gripper值
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
    action_gripper_data = []  # 每个episode的action gripper值
    episode_lengths = []
    
    for ep_idx in range(num_episodes_to_plot):
        try:
            episode_state_gripper = []
            episode_action_gripper = []
            
            # 获取episode的所有帧
            try:
                # 获取episode信息
                if hasattr(dataset, 'episode_data_index'):
                    episode_index = dataset.episode_data_index
                    if isinstance(episode_index, dict):
                        # episode_data_index 是一个字典，包含 'from' 和 'to' 键
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
                                            # 提取state gripper
                                            for key in ["state", "observation.state", "observation/state"]:
                                                if key in frame:
                                                    state = frame[key]
                                                    if TORCH_AVAILABLE and isinstance(state, torch.Tensor):
                                                        state = state.cpu().numpy()
                                                    else:
                                                        state = np.array(state)
                                                    if state.ndim == 1 and len(state) >= 8:
                                                        episode_state_gripper.append(float(state[7]))
                                                    break
                                            
                                            # 提取action gripper
                                            for key in ["actions", "action"]:
                                                if key in frame:
                                                    action = frame[key]
                                                    if TORCH_AVAILABLE and isinstance(action, torch.Tensor):
                                                        action = action.cpu().numpy()
                                                    else:
                                                        action = np.array(action)
                                                    if action.ndim == 1 and len(action) >= 8:
                                                        episode_action_gripper.append(float(action[7]))
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
                    # 如果没有episode_data_index，尝试通过episode_index字段查找
                    print(f"  ⚠️  Episode {ep_idx}: 无法获取episode_data_index，尝试通过episode_index字段查找")
                    # 遍历数据集查找属于该episode的所有帧
                    for frame_idx in range(len(dataset)):
                        try:
                            frame = dataset[frame_idx]
                            if isinstance(frame, dict) and 'episode_index' in frame:
                                if frame['episode_index'] == ep_idx:
                                    # 提取state gripper
                                    for key in ["state", "observation.state", "observation/state"]:
                                        if key in frame:
                                            state = frame[key]
                                            if TORCH_AVAILABLE and isinstance(state, torch.Tensor):
                                                state = state.cpu().numpy()
                                            else:
                                                state = np.array(state)
                                            if state.ndim == 1 and len(state) >= 8:
                                                episode_state_gripper.append(float(state[7]))
                                            break
                                    
                                    # 提取action gripper
                                    for key in ["actions", "action"]:
                                        if key in frame:
                                            action = frame[key]
                                            if TORCH_AVAILABLE and isinstance(action, torch.Tensor):
                                                action = action.cpu().numpy()
                                            else:
                                                action = np.array(action)
                                            if action.ndim == 1 and len(action) >= 8:
                                                episode_action_gripper.append(float(action[7]))
                                            break
                        except Exception:
                            continue
                
            except Exception as e:
                print(f"  ⚠️  访问episode {ep_idx} 数据时出错: {e}")
                import traceback
                traceback.print_exc()
            
            state_gripper_data.append(episode_state_gripper)
            action_gripper_data.append(episode_action_gripper)
            
            # 记录episode长度
            if episode_state_gripper:
                episode_lengths.append(len(episode_state_gripper))
                print(f"  ✅ Episode {ep_idx}: 提取了 {len(episode_state_gripper)} 个state gripper值")
            elif episode_action_gripper:
                episode_lengths.append(len(episode_action_gripper))
                print(f"  ✅ Episode {ep_idx}: 提取了 {len(episode_action_gripper)} 个action gripper值")
            else:
                episode_lengths.append(0)
                print(f"  ⚠️  Episode {ep_idx}: 未找到gripper数据")
                
        except Exception as e:
            print(f"⚠️  处理episode {ep_idx} 时出错: {e}")
            import traceback
            traceback.print_exc()
            state_gripper_data.append([])
            action_gripper_data.append([])
            episode_lengths.append(0)
    
    # 确定要绘制的内容
    num_plots = 0
    if plot_state:
        num_plots += num_episodes_to_plot
    if plot_action:
        num_plots += num_episodes_to_plot
    
    if num_plots == 0:
        print("❌ 没有可绘制的内容")
        return
    
    # 创建子图
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 3 * num_plots))
    
    # 如果只有一个子图，axes不是数组，需要转换为数组
    if num_plots == 1:
        axes = [axes]
    
    fig.suptitle(
        f'Training Data Gripper Values (First {num_episodes_to_plot} episodes, Threshold={threshold})',
        fontsize=14,
        fontweight='bold'
    )
    
    plot_idx = 0
    
    # 绘制state中的gripper值
    if plot_state:
        for ep_idx in range(num_episodes_to_plot):
            ax = axes[plot_idx]
            gripper_data = state_gripper_data[ep_idx]
            
            if len(gripper_data) > 0:
                gripper_data = np.array(gripper_data)
                valid_mask = ~np.isnan(gripper_data)
                valid_data = gripper_data[valid_mask]
                valid_indices = np.where(valid_mask)[0]
                
                if len(valid_data) > 0:
                    # 绘制折线图
                    ax.plot(valid_indices, valid_data, 'b-', linewidth=1.5, label='State Gripper', alpha=0.7)
                    
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
                    ax.set_ylabel('State Gripper Value', fontsize=10)
                    ax.set_title(
                        f'Episode {ep_idx} - State Gripper\n'
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
            
            plot_idx += 1
    
    # 绘制action中的gripper值
    if plot_action:
        for ep_idx in range(num_episodes_to_plot):
            ax = axes[plot_idx]
            gripper_data = action_gripper_data[ep_idx]
            
            if len(gripper_data) > 0:
                gripper_data = np.array(gripper_data)
                valid_mask = ~np.isnan(gripper_data)
                valid_data = gripper_data[valid_mask]
                valid_indices = np.where(valid_mask)[0]
                
                if len(valid_data) > 0:
                    # 绘制折线图
                    ax.plot(valid_indices, valid_data, 'g-', linewidth=1.5, label='Action Gripper', alpha=0.7)
                    
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
                    ax.set_ylabel('Action Gripper Value', fontsize=10)
                    ax.set_title(
                        f'Episode {ep_idx} - Action Gripper\n'
                        f'Total: {len(valid_data)}, >{threshold}: {np.sum(above_threshold_mask)} '
                        f'({np.sum(above_threshold_mask)/len(valid_data)*100:.2f}%)',
                        fontsize=10,
                        fontweight='bold'
                    )
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='upper right', fontsize=8)
                    
                    # 设置y轴范围（action可能是速度，范围可能不同）
                    y_min = np.min(valid_data) * 1.1
                    y_max = np.max(valid_data) * 1.1
                    ax.set_ylim(y_min, y_max)
                else:
                    ax.text(0.5, 0.5, f'Episode {ep_idx} - Action Gripper\nNo valid data',
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'Episode {ep_idx} - Action Gripper', fontsize=10)
            else:
                ax.text(0.5, 0.5, f'Episode {ep_idx} - Action Gripper\nNo data',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Episode {ep_idx} - Action Gripper', fontsize=10)
            
            plot_idx += 1
    
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
        print(f"\nEpisode {ep_idx}:")
        if state_gripper_data[ep_idx]:
            state_data = np.array(state_gripper_data[ep_idx])
            valid_state = state_data[~np.isnan(state_data)]
            if len(valid_state) > 0:
                print(f"  State Gripper: min={np.min(valid_state):.6f}, max={np.max(valid_state):.6f}, "
                      f"mean={np.mean(valid_state):.6f}, >{threshold}: {np.sum(valid_state > threshold)}/{len(valid_state)}")
        if action_gripper_data[ep_idx]:
            action_data = np.array(action_gripper_data[ep_idx])
            valid_action = action_data[~np.isnan(action_data)]
            if len(valid_action) > 0:
                print(f"  Action Gripper: min={np.min(valid_action):.6f}, max={np.max(valid_action):.6f}, "
                      f"mean={np.mean(valid_action):.6f}, >{threshold}: {np.sum(valid_action > threshold)}/{len(valid_action)}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="从训练数据集中提取gripper值并绘制折线图",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 绘制前5个episode的state和action gripper值
  python scripts/plot_training_data_gripper.py --repo-id your_hf_username/pick_blue_bottle_libero_downsample4x --num-episodes 5

  # 只绘制state gripper值
  python scripts/plot_training_data_gripper.py --repo-id your_hf_username/pick_blue_bottle_libero_downsample4x --num-episodes 5 --no-action

  # 只绘制action gripper值
  python scripts/plot_training_data_gripper.py --repo-id your_hf_username/pick_blue_bottle_libero_downsample4x --num-episodes 5 --no-state
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
    
    parser.add_argument(
        '--no-state',
        action='store_true',
        help='不绘制state中的gripper值'
    )
    
    parser.add_argument(
        '--no-action',
        action='store_true',
        help='不绘制action中的gripper值'
    )
    
    args = parser.parse_args()
    
    plot_training_data_gripper(
        repo_id=args.repo_id,
        num_episodes=args.num_episodes,
        threshold=args.threshold,
        output_path=args.output,
        plot_state=not args.no_state,
        plot_action=not args.no_action,
    )


if __name__ == '__main__':
    main()

