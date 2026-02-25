#!/usr/bin/env python3
"""
从HDF5文件中提取gripper值（用于state的最后一维）

提取训练数据中使用的gripper值，这些值会作为state的最后一维

使用方法:
    python scripts/extract_gripper_from_hdf5.py <hdf5_file_path>
    python scripts/extract_gripper_from_hdf5.py --directory <directory> --max-files 5
"""

import argparse
import sys
import os
import glob

import h5py
import numpy as np

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not available. Install with: pip install matplotlib")


def extract_gripper_from_hdf5(hdf5_path: str, downsample_factor: int = 4):
    """
    从HDF5文件中提取gripper值（与数据转换脚本相同的逻辑）
    
    Args:
        hdf5_path: HDF5文件路径
        downsample_factor: 下采样因子（默认4，即30fps -> 7.5fps）
    
    Returns:
        gripper_values: 提取的gripper值数组，或None如果失败
    """
    try:
        with h5py.File(hdf5_path, 'r') as f:
            # 检查是否存在gripper话题
            if "_control_gripperValueR" not in f.get("topics", {}):
                print(f"  ⚠️  未找到 _control_gripperValueR 话题")
                return None
            
            gripper_topic = f["topics/_control_gripperValueR"]
            if "data" not in gripper_topic:
                print(f"  ⚠️  夹爪话题中没有 'data' 键")
                return None
            
            # 读取原始gripper数据
            gripper_data = gripper_topic["data"][:]  # (T,)
            print(f"  📊 原始gripper数据: {len(gripper_data)} 个值")
            print(f"     范围: [{np.min(gripper_data):.6f}, {np.max(gripper_data):.6f}]")
            print(f"     平均值: {np.mean(gripper_data):.6f}")
            
            # 过滤NaN值（用0填充，与数据转换脚本一致）
            gripper_data = np.nan_to_num(gripper_data, nan=0.0)
            
            # 下采样（与数据转换脚本一致）
            downsampled_indices = np.arange(0, len(gripper_data), downsample_factor)
            right_gripper_values = gripper_data[downsampled_indices]
            
            print(f"  ✅ 下采样后gripper数据: {len(right_gripper_values)} 个值（下采样 {downsample_factor}x）")
            print(f"     范围: [{np.min(right_gripper_values):.6f}, {np.max(right_gripper_values):.6f}]")
            print(f"     平均值: {np.mean(right_gripper_values):.6f}")
            
            # 统计大于0.8的值（闭合状态）
            above_08 = np.sum(right_gripper_values > 0.8)
            print(f"     大于0.8的数量: {above_08} ({above_08/len(right_gripper_values)*100:.2f}%)")
            
            return right_gripper_values
            
    except Exception as e:
        print(f"  ❌ 读取HDF5文件失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_gripper_values(gripper_values_list: list, filenames: list, output_path: str = None):
    """
    绘制多个文件的gripper值
    
    Args:
        gripper_values_list: gripper值列表（每个元素是一个数组）
        filenames: 对应的文件名列表
        output_path: 输出图片路径
    """
    if not MATPLOTLIB_AVAILABLE:
        print("❌ matplotlib不可用，无法绘图")
        return
    
    num_files = len(gripper_values_list)
    if num_files == 0:
        print("❌ 没有数据可绘制")
        return
    
    fig, axes = plt.subplots(num_files, 1, figsize=(12, 3 * num_files))
    
    if num_files == 1:
        axes = [axes]
    
    fig.suptitle('Gripper Values from HDF5 Files (State Last Dimension)', fontsize=14, fontweight='bold')
    
    for idx, (gripper_values, filename) in enumerate(zip(gripper_values_list, filenames)):
        ax = axes[idx]
        
        if gripper_values is not None and len(gripper_values) > 0:
            # 绘制折线图
            ax.plot(gripper_values, 'b-', linewidth=1.5, label='Gripper Value', alpha=0.7)
            
            # 标记大于0.8的点
            above_08_mask = gripper_values > 0.8
            if np.any(above_08_mask):
                ax.scatter(
                    np.where(above_08_mask)[0],
                    gripper_values[above_08_mask],
                    c='red',
                    s=30,
                    marker='o',
                    label='> 0.8 (closed)',
                    zorder=5
                )
            
            # 添加阈值线
            ax.axhline(y=0.8, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Threshold=0.8')
            
            # 设置标签和标题
            ax.set_xlabel('Time Step (Downsampled)', fontsize=10)
            ax.set_ylabel('Gripper Value (0=open, 1=closed)', fontsize=10)
            ax.set_title(
                f'{os.path.basename(filename)}\n'
                f'Total: {len(gripper_values)}, >0.8: {np.sum(above_08_mask)} ({np.sum(above_08_mask)/len(gripper_values)*100:.2f}%)',
                fontsize=10,
                fontweight='bold'
            )
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)
            ax.set_ylim(-0.1, 1.1)
        else:
            ax.text(0.5, 0.5, f'{os.path.basename(filename)}\nNo gripper data',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(os.path.basename(filename), fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Image saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="从HDF5文件中提取gripper值（用于state的最后一维）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 提取单个文件的gripper值
  python scripts/extract_gripper_from_hdf5.py pick_blue_bottle/rosbag2_2026_01_09-21_26_09/rosbag2_2026_01_09-21_26_09_0.h5

  # 批量提取并绘制
  python scripts/extract_gripper_from_hdf5.py --directory pick_blue_bottle --max-files 5 --plot --output gripper_extracted.png
        """
    )
    
    parser.add_argument(
        'hdf5_file',
        type=str,
        nargs='?',
        help='HDF5文件路径（单个文件模式）'
    )
    
    parser.add_argument(
        '--directory',
        type=str,
        help='HDF5文件所在目录（批量模式）'
    )
    
    parser.add_argument(
        '--max-files',
        type=int,
        default=5,
        help='批量模式下最多处理的文件数量（默认：5）'
    )
    
    parser.add_argument(
        '--downsample-factor',
        type=int,
        default=4,
        help='下采样因子（默认：4，即30fps -> 7.5fps）'
    )
    
    parser.add_argument(
        '--plot',
        action='store_true',
        help='绘制gripper值折线图'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='图片输出路径（如果指定，则保存图片；否则显示）'
    )
    
    parser.add_argument(
        '--save-values',
        type=str,
        default=None,
        help='保存gripper值到numpy文件（.npz格式）'
    )
    
    args = parser.parse_args()
    
    if args.directory:
        # 批量模式
        pattern = os.path.join(args.directory, "**", "*.h5")
        h5_files = sorted(glob.glob(pattern, recursive=True))[:args.max_files]
        
        if len(h5_files) == 0:
            print(f"❌ 在目录 {args.directory} 中未找到HDF5文件")
            return
        
        print(f"📂 批量处理目录: {args.directory}")
        print(f"📊 处理前 {len(h5_files)} 个文件\n")
        
        gripper_values_list = []
        filenames = []
        
        for h5_file in h5_files:
            filename = os.path.basename(h5_file)
            print(f"📄 处理文件: {filename}")
            gripper_values = extract_gripper_from_hdf5(h5_file, args.downsample_factor)
            gripper_values_list.append(gripper_values)
            filenames.append(h5_file)
            print()
        
        # 保存值（如果指定）
        if args.save_values:
            np.savez(args.save_values, 
                    gripper_values=gripper_values_list,
                    filenames=filenames)
            print(f"✅ Gripper值已保存到: {args.save_values}")
        
        # 绘图（如果指定）
        if args.plot:
            plot_gripper_values(gripper_values_list, filenames, args.output)
        
    elif args.hdf5_file:
        # 单个文件模式
        print(f"📄 处理文件: {args.hdf5_file}\n")
        gripper_values = extract_gripper_from_hdf5(args.hdf5_file, args.downsample_factor)
        
        if gripper_values is not None:
            print(f"\n📊 统计信息:")
            print(f"   总样本数: {len(gripper_values)}")
            print(f"   最小值: {np.min(gripper_values):.6f}")
            print(f"   最大值: {np.max(gripper_values):.6f}")
            print(f"   平均值: {np.mean(gripper_values):.6f}")
            print(f"   中位数: {np.median(gripper_values):.6f}")
            print(f"   标准差: {np.std(gripper_values):.6f}")
            print(f"   大于0.8的数量: {np.sum(gripper_values > 0.8)} ({np.sum(gripper_values > 0.8)/len(gripper_values)*100:.2f}%)")
            
            # 保存值（如果指定）
            if args.save_values:
                np.savez(args.save_values, gripper_values=gripper_values)
                print(f"\n✅ Gripper值已保存到: {args.save_values}")
            
            # 绘图（如果指定）
            if args.plot:
                plot_gripper_values([gripper_values], [args.hdf5_file], args.output)
        else:
            print("❌ 未能提取gripper值")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()















