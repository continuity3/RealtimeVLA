#!/usr/bin/env python3
"""
测试脚本：从HDF5文件读取数据，去掉NaN值后重建state，并绘制gripper折线图

这个脚本会：
1. 读取HDF5文件中的joint_states和gripper数据
2. 过滤掉NaN值，确保数据对齐
3. 重建state（7个关节位置 + 1个gripper值）
4. 提取gripper值并绘制折线图

使用方法:
    python scripts/test_state_gripper_no_nan.py <hdf5_file_path> [output_image.png]
"""

import argparse
import sys

import h5py
import numpy as np

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not available. Install with: pip install matplotlib")


def load_and_rebuild_state(hdf5_path: str):
    """
    从HDF5文件加载数据，去掉NaN值后重建state
    
    Args:
        hdf5_path: HDF5文件路径
    
    Returns:
        states: 重建后的state数组 (T, 8) - [7个关节位置, 1个gripper值]
        gripper_values: gripper值数组 (T,)
        valid_indices: 有效数据的索引
    """
    with h5py.File(hdf5_path, 'r') as f:
        # 读取关节状态
        if "_joint_states" not in f["topics"]:
            raise KeyError("找不到 _joint_states topic")
        
        joint_states = f["topics/_joint_states"]
        positions = joint_states["position"][:]  # (T, 14)
        velocities = joint_states["velocity"][:]  # (T, 14)
        
        # 读取gripper数据
        right_gripper_values = None
        if "_control_gripperValueR" in f["topics"]:
            gripper_topic = f["topics/_control_gripperValueR"]
            if "data" in gripper_topic:
                gripper_data = gripper_topic["data"][:]  # (T,)
                right_gripper_values = gripper_data
            else:
                print("  ⚠️  夹爪话题中没有 'data' 键")
        else:
            print("  ⚠️  未找到 _control_gripperValueR 话题")
        
        if right_gripper_values is None:
            raise ValueError("无法读取gripper数据")
        
        # 确保数据长度一致
        min_length = min(len(positions), len(right_gripper_values))
        positions = positions[:min_length]
        velocities = velocities[:min_length]
        right_gripper_values = right_gripper_values[:min_length]
        
        print(f"📊 原始数据长度: {min_length}")
        
        # 提取右臂关节（列 7-13，对应 Joint1_R 到 Joint7_R）
        right_positions = positions[:, 7:14]  # (T, 7) - 右臂关节位置
        
        # 找到所有有效的时间步（joint位置和gripper值都不是NaN）
        # 对于joint位置，检查所有7个关节是否都是有效值
        joint_valid = ~np.isnan(right_positions).any(axis=1)  # (T,)
        gripper_valid = ~np.isnan(right_gripper_values)  # (T,)
        
        # 两者都有效的时间步
        valid_mask = joint_valid & gripper_valid
        
        valid_indices = np.where(valid_mask)[0]
        
        print(f"📊 有效数据统计:")
        print(f"   Joint有效: {np.sum(joint_valid)}/{len(joint_valid)} ({np.sum(joint_valid)/len(joint_valid)*100:.2f}%)")
        print(f"   Gripper有效: {np.sum(gripper_valid)}/{len(gripper_valid)} ({np.sum(gripper_valid)/len(gripper_valid)*100:.2f}%)")
        print(f"   两者都有效: {len(valid_indices)}/{min_length} ({len(valid_indices)/min_length*100:.2f}%)")
        
        if len(valid_indices) == 0:
            raise ValueError("没有有效的数据步骤（所有数据都包含NaN）")
        
        # 提取有效数据
        valid_right_positions = right_positions[valid_indices]  # (N, 7)
        valid_gripper_values = right_gripper_values[valid_indices]  # (N,)
        
        # 重建state：组合右臂关节位置和gripper值
        states = np.concatenate([
            valid_right_positions,
            valid_gripper_values[:, None]
        ], axis=1)  # (N, 8)
        
        print(f"✅ 重建state完成: shape={states.shape}")
        print(f"   State范围:")
        print(f"     Joint位置: [{np.min(valid_right_positions):.4f}, {np.max(valid_right_positions):.4f}]")
        print(f"     Gripper值: [{np.min(valid_gripper_values):.6f}, {np.max(valid_gripper_values):.6f}]")
        
        return states, valid_gripper_values, valid_indices


def plot_gripper_values(gripper_values: np.ndarray, valid_indices: np.ndarray, hdf5_path: str, threshold: float = 0.8, output_path: str = None):
    """
    绘制gripper值折线图
    
    Args:
        gripper_values: gripper值数组
        valid_indices: 有效数据的原始索引
        hdf5_path: HDF5文件路径（用于标题）
        threshold: 阈值（用于标记）
        output_path: 输出图片路径
    """
    if not MATPLOTLIB_AVAILABLE:
        print("❌ matplotlib不可用，无法绘图")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # 使用原始索引作为x轴（显示在原始数据中的位置）
    x_indices = valid_indices
    
    # 绘制折线图
    ax.plot(x_indices, gripper_values, 'b-', linewidth=1.5, label='Gripper Value (No NaN)', alpha=0.7)
    
    # 标记大于阈值的点
    above_threshold_mask = gripper_values > threshold
    if np.any(above_threshold_mask):
        ax.scatter(
            x_indices[above_threshold_mask],
            gripper_values[above_threshold_mask],
            c='red',
            s=30,
            marker='o',
            label=f'> {threshold} (closed)',
            zorder=5
        )
    
    # 添加阈值线
    ax.axhline(y=threshold, color='r', linestyle='--', linewidth=1, alpha=0.5, label=f'Threshold={threshold}')
    
    # 设置标签和标题
    ax.set_xlabel('Original Time Step Index (NaN values removed)', fontsize=12)
    ax.set_ylabel('Gripper Value (0=open, 1=closed)', fontsize=12)
    ax.set_title(
        f'State Gripper Values (No NaN) - {hdf5_path.split("/")[-1]}\n'
        f'Total valid samples: {len(gripper_values)}, >{threshold}: {np.sum(above_threshold_mask)} ({np.sum(above_threshold_mask)/len(gripper_values)*100:.2f}%)',
        fontsize=12,
        fontweight='bold'
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    # 设置y轴范围
    y_min = min(0, np.min(gripper_values) * 0.1)
    y_max = max(1.0, np.max(gripper_values) * 1.1)
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Image saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="从HDF5文件读取数据，去掉NaN值后重建state，并绘制gripper折线图",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 测试单个文件并显示图片
  python scripts/test_state_gripper_no_nan.py pick_blue_bottle/rosbag2_2026_01_09-21_25_15/rosbag2_2026_01_09-21_25_15_0.h5

  # 测试并保存图片
  python scripts/test_state_gripper_no_nan.py pick_blue_bottle/rosbag2_2026_01_09-21_25_15/rosbag2_2026_01_09-21_25_15_0.h5 state_gripper_no_nan.png
        """
    )
    
    parser.add_argument(
        'hdf5_file',
        type=str,
        help='HDF5文件路径'
    )
    
    parser.add_argument(
        'output_image',
        type=str,
        nargs='?',
        default=None,
        help='输出图片路径（可选，如果不指定则显示图片）'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.8,
        help='gripper值的阈值（默认：0.8）'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"📂 处理HDF5文件: {args.hdf5_file}")
    print("=" * 80)
    print()
    
    try:
        # 加载数据并重建state
        states, gripper_values, valid_indices = load_and_rebuild_state(args.hdf5_file)
        
        print()
        print("=" * 80)
        print("📊 Gripper值统计（去掉NaN后）:")
        print("=" * 80)
        print(f"   总样本数: {len(gripper_values)}")
        print(f"   最小值: {np.min(gripper_values):.6f}")
        print(f"   最大值: {np.max(gripper_values):.6f}")
        print(f"   平均值: {np.mean(gripper_values):.6f}")
        print(f"   中位数: {np.median(gripper_values):.6f}")
        print(f"   标准差: {np.std(gripper_values):.6f}")
        print(f"   大于{args.threshold}的数量: {np.sum(gripper_values > args.threshold)} ({np.sum(gripper_values > args.threshold)/len(gripper_values)*100:.2f}%)")
        print("=" * 80)
        print()
        
        # 绘制折线图
        plot_gripper_values(gripper_values, valid_indices, args.hdf5_file, args.threshold, args.output_image)
        
        print()
        print("✅ 处理完成！")
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

