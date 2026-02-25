#!/usr/bin/env python3
"""
检查HDF5文件中的夹爪状态

查看HDF5文件中夹爪数据的格式、范围和分布

使用方法:
    python scripts/inspect_hdf5_gripper.py <hdf5_file_path>
"""

import argparse
import sys
import os

import h5py
import numpy as np

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not available. Install with: pip install matplotlib")


def inspect_hdf5_gripper(hdf5_path: str):
    """
    检查HDF5文件中的夹爪状态
    
    Args:
        hdf5_path: HDF5文件路径
    """
    print("=" * 80)
    print(f"📂 检查HDF5文件: {hdf5_path}")
    print("=" * 80)
    print()
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            # 检查文件结构
            print("📋 文件结构:")
            def print_structure(name, obj):
                print(f"  {name}")
            f.visititems(print_structure)
            print()
            
            # 检查夹爪话题
            gripper_topic_path = "topics/_control_gripperValueR"
            if gripper_topic_path in f:
                print(f"✅ 找到夹爪话题: {gripper_topic_path}")
                gripper_topic = f[gripper_topic_path]
                
                # 列出话题下的所有键
                print(f"   话题下的键: {list(gripper_topic.keys())}")
                print()
                
                # 读取夹爪数据
                if "data" in gripper_topic:
                    gripper_data = gripper_topic["data"][:]  # (T,)
                    print(f"📊 夹爪数据统计:")
                    print(f"   形状: {gripper_data.shape}")
                    print(f"   数据类型: {gripper_data.dtype}")
                    print(f"   最小值: {np.min(gripper_data):.6f}")
                    print(f"   最大值: {np.max(gripper_data):.6f}")
                    print(f"   平均值: {np.mean(gripper_data):.6f}")
                    print(f"   中位数: {np.median(gripper_data):.6f}")
                    print(f"   标准差: {np.std(gripper_data):.6f}")
                    print()
                    
                    # 检查NaN和无效值
                    nan_count = np.sum(np.isnan(gripper_data))
                    inf_count = np.sum(np.isinf(gripper_data))
                    print(f"🔍 数据质量:")
                    print(f"   NaN值数量: {nan_count}")
                    print(f"   无穷值数量: {inf_count}")
                    print()
                    
                    # 统计不同值的分布
                    unique_values, counts = np.unique(gripper_data[~np.isnan(gripper_data)], return_counts=True)
                    print(f"📈 值分布（前20个最常见的值）:")
                    sorted_indices = np.argsort(counts)[::-1][:20]
                    for idx in sorted_indices:
                        print(f"   值 {unique_values[idx]:.6f}: {counts[idx]} 次 ({counts[idx]/len(gripper_data)*100:.2f}%)")
                    print()
                    
                    # 检查0和1的数量
                    zero_count = np.sum(gripper_data == 0.0)
                    one_count = np.sum(gripper_data == 1.0)
                    close_to_zero = np.sum(np.abs(gripper_data) < 0.01)
                    close_to_one = np.sum(np.abs(gripper_data - 1.0) < 0.01)
                    
                    print(f"🎯 关键值统计:")
                    print(f"   等于0.0的数量: {zero_count} ({zero_count/len(gripper_data)*100:.2f}%)")
                    print(f"   接近0.0 (<0.01)的数量: {close_to_zero} ({close_to_zero/len(gripper_data)*100:.2f}%)")
                    print(f"   等于1.0的数量: {one_count} ({one_count/len(gripper_data)*100:.2f}%)")
                    print(f"   接近1.0 (>0.99)的数量: {close_to_one} ({close_to_one/len(gripper_data)*100:.2f}%)")
                    print()
                    
                    # 显示前10个和后10个值
                    print(f"📝 前10个值:")
                    for i in range(min(10, len(gripper_data))):
                        print(f"   [{i}]: {gripper_data[i]:.6f}")
                    print()
                    
                    if len(gripper_data) > 10:
                        print(f"📝 后10个值:")
                        for i in range(max(0, len(gripper_data)-10), len(gripper_data)):
                            print(f"   [{i}]: {gripper_data[i]:.6f}")
                    print()
                    
                    # 检查是否有其他相关数据
                    if "data_length" in gripper_topic:
                        data_length = gripper_topic["data_length"][:]
                        print(f"📏 data_length 信息:")
                        print(f"   形状: {data_length.shape}")
                        print(f"   最小值: {np.min(data_length)}")
                        print(f"   最大值: {np.max(data_length)}")
                        print()
                else:
                    print(f"⚠️  话题中没有 'data' 键")
                    print(f"   可用键: {list(gripper_topic.keys())}")
            else:
                print(f"⚠️  未找到夹爪话题: {gripper_topic_path}")
                print()
                print("🔍 可用的topics:")
                if "topics" in f:
                    topics = list(f["topics"].keys())
                    for topic in topics:
                        print(f"   - {topic}")
                else:
                    print("   文件中没有 'topics' 组")
            
            # 检查是否有其他夹爪相关的数据
            print()
            print("🔍 搜索所有可能包含'gripper'的键:")
            def search_gripper(name, obj):
                if 'gripper' in name.lower():
                    print(f"   找到: {name} (类型: {type(obj).__name__})")
            f.visititems(search_gripper)
            
    except FileNotFoundError:
        print(f"❌ 错误: 文件不存在: {hdf5_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print()
    print("=" * 80)
    print("✅ 检查完成")
    print("=" * 80)


def plot_gripper_values(directory: str, num_files: int = 5, threshold: float = 0.8, output_path: str = None):
    """
    绘制gripper值随时间变化的折线图
    
    Args:
        directory: HDF5文件所在目录
        num_files: 要绘制的文件数量
        threshold: gripper值的阈值（用于标记）
        output_path: 输出图片路径（如果为None，则显示图片）
    """
    if not MATPLOTLIB_AVAILABLE:
        print("❌ matplotlib不可用，无法绘图")
        return
    
    import glob
    
    # 查找所有HDF5文件
    pattern = os.path.join(directory, "**", "*.h5")
    h5_files = sorted(glob.glob(pattern, recursive=True))[:num_files]
    
    if len(h5_files) == 0:
        print(f"❌ 在目录 {directory} 中未找到HDF5文件")
        return
    
    print(f"📊 Preparing to plot gripper values for first {len(h5_files)} files...")
    
    # 创建子图：num_files行1列
    fig, axes = plt.subplots(num_files, 1, figsize=(12, 3 * num_files))
    
    # 如果只有一个文件，axes不是数组，需要转换为数组
    if num_files == 1:
        axes = [axes]
    
    fig.suptitle(f'Gripper Value Over Time (First {num_files} files, Threshold={threshold})', fontsize=14, fontweight='bold')
    
    for idx, h5_file in enumerate(h5_files):
        filename = os.path.basename(h5_file)
        ax = axes[idx]
        
        try:
            with h5py.File(h5_file, 'r') as f:
                gripper_topic_path = "topics/_control_gripperValueR"
                if gripper_topic_path in f and "data" in f[gripper_topic_path]:
                    gripper_data = f[gripper_topic_path]["data"][:]
                    
                    # 过滤NaN值
                    valid_mask = ~np.isnan(gripper_data)
                    valid_data = gripper_data[valid_mask]
                    valid_indices = np.where(valid_mask)[0]
                    
                    if len(valid_data) > 0:
                        # 绘制折线图
                        ax.plot(valid_indices, valid_data, 'b-', linewidth=1.5, label='Gripper Value', alpha=0.7)
                        
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
                        ax.set_ylabel('Gripper Value', fontsize=10)
                        ax.set_title(f'{filename}\nTotal: {len(valid_data)}, >{threshold}: {np.sum(above_threshold_mask)} ({np.sum(above_threshold_mask)/len(valid_data)*100:.2f}%)', 
                                    fontsize=10, fontweight='bold')
                        ax.grid(True, alpha=0.3)
                        ax.legend(loc='upper right', fontsize=8)
                        
                        # 设置y轴范围
                        y_min = min(0, np.min(valid_data) * 0.1)
                        y_max = max(1.0, np.max(valid_data) * 1.1)
                        ax.set_ylim(y_min, y_max)
                        
                        # 设置x轴范围
                        ax.set_xlim(-1, len(gripper_data))
                    else:
                        ax.text(0.5, 0.5, f'{filename}\nNo valid data', 
                               ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(filename, fontsize=10)
                else:
                    ax.text(0.5, 0.5, f'{filename}\nNo gripper data found', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(filename, fontsize=10)
        except Exception as e:
            ax.text(0.5, 0.5, f'{filename}\nError: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(filename, fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Image saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def inspect_hdf5_gripper_batch(directory: str, max_files: int = 20, threshold: float = 0.8, recursive: bool = True):
    """
    批量检查目录中前N个HDF5文件中的夹爪状态
    
    Args:
        directory: HDF5文件所在目录
        max_files: 最多检查的文件数量
        threshold: gripper值的阈值（大于此值视为闭合）
        recursive: 是否递归搜索子目录
    """
    import glob
    import os
    
    # 查找所有HDF5文件
    if recursive:
        # 递归搜索所有子目录中的HDF5文件
        pattern = os.path.join(directory, "**", "*.h5")
        h5_files = sorted(glob.glob(pattern, recursive=True))
    else:
        # 只搜索当前目录
        h5_files = sorted(glob.glob(os.path.join(directory, "*.h5")))
    
    if len(h5_files) == 0:
        print(f"❌ 在目录 {directory} 中未找到HDF5文件")
        return
    
    # 限制文件数量
    h5_files = h5_files[:max_files]
    
    print("=" * 80)
    print(f"📂 批量检查目录: {directory}")
    print(f"📊 检查前 {len(h5_files)} 个文件，gripper值 > {threshold} 的比率")
    print("=" * 80)
    print()
    
    results = []
    
    for h5_file in h5_files:
        filename = os.path.basename(h5_file)
        try:
            with h5py.File(h5_file, 'r') as f:
                gripper_topic_path = "topics/_control_gripperValueR"
                if gripper_topic_path in f and "data" in f[gripper_topic_path]:
                    gripper_data = f[gripper_topic_path]["data"][:]
                    
                    # 过滤NaN值
                    valid_data = gripper_data[~np.isnan(gripper_data)]
                    
                    if len(valid_data) > 0:
                        # 计算大于阈值的比率
                        above_threshold = np.sum(valid_data > threshold)
                        ratio = above_threshold / len(valid_data) * 100
                        
                        # 统计信息
                        min_val = np.min(valid_data)
                        max_val = np.max(valid_data)
                        mean_val = np.mean(valid_data)
                        
                        results.append({
                            'filename': filename,
                            'total': len(valid_data),
                            'above_threshold': above_threshold,
                            'ratio': ratio,
                            'min': min_val,
                            'max': max_val,
                            'mean': mean_val
                        })
                    else:
                        results.append({
                            'filename': filename,
                            'total': 0,
                            'above_threshold': 0,
                            'ratio': 0.0,
                            'min': np.nan,
                            'max': np.nan,
                            'mean': np.nan
                        })
                else:
                    results.append({
                        'filename': filename,
                        'total': 0,
                        'above_threshold': 0,
                        'ratio': 0.0,
                        'min': np.nan,
                        'max': np.nan,
                        'mean': np.nan,
                        'error': 'No gripper data'
                    })
        except Exception as e:
            results.append({
                'filename': filename,
                'error': str(e)
            })
    
    # 打印结果表格
    print(f"{'文件名':<50} {'总数':<8} {'>0.8':<8} {'比率%':<8} {'最小值':<10} {'最大值':<10} {'平均值':<10}")
    print("-" * 120)
    
    for r in results:
        if 'error' in r:
            print(f"{r['filename']:<50} {'ERROR':<8} {r.get('error', 'Unknown')}")
        else:
            print(f"{r['filename']:<50} {r['total']:<8} {r['above_threshold']:<8} {r['ratio']:<8.2f} "
                  f"{r['min']:<10.6f} {r['max']:<10.6f} {r['mean']:<10.6f}")
    
    print()
    print("=" * 80)
    
    # 统计汇总
    valid_results = [r for r in results if 'error' not in r and r['total'] > 0]
    if len(valid_results) > 0:
        total_samples = sum(r['total'] for r in valid_results)
        total_above_threshold = sum(r['above_threshold'] for r in valid_results)
        overall_ratio = total_above_threshold / total_samples * 100 if total_samples > 0 else 0
        
        print(f"📊 汇总统计:")
        print(f"   检查文件数: {len(valid_results)}")
        print(f"   总样本数: {total_samples}")
        print(f"   大于{threshold}的样本数: {total_above_threshold}")
        print(f"   总体比率: {overall_ratio:.2f}%")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="检查HDF5文件中的夹爪状态",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 检查单个HDF5文件
  python scripts/inspect_hdf5_gripper.py pick_blue_bottle_extracted/rosbag2_2026_01_09-21_24_59_0.h5

  # 批量检查目录中前20个文件
  python scripts/inspect_hdf5_gripper.py --directory pick_blue_bottle/rosbag2_2026_01_09-21_26_09 --max-files 20

  # 检查多个文件
  for file in pick_blue_bottle_extracted/*.h5; do
      python scripts/inspect_hdf5_gripper.py "$file"
  done
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
        default=20,
        help='批量模式下最多检查的文件数量（默认：20）'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.8,
        help='gripper值的阈值（默认：0.8）'
    )
    
    parser.add_argument(
        '--recursive',
        action='store_true',
        default=True,
        help='递归搜索子目录中的HDF5文件（默认：True）'
    )
    
    parser.add_argument(
        '--no-recursive',
        dest='recursive',
        action='store_false',
        help='不递归搜索，只搜索指定目录'
    )
    
    parser.add_argument(
        '--plot',
        action='store_true',
        help='绘制gripper值随时间变化的折线图'
    )
    
    parser.add_argument(
        '--plot-files',
        type=int,
        default=5,
        help='绘图模式下要绘制的文件数量（默认：5）'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='图片输出路径（如果指定，则保存图片；否则显示）'
    )
    
    args = parser.parse_args()
    
    if args.plot:
        # 绘图模式
        if not args.directory:
            print("❌ 绘图模式需要指定目录（--directory）")
            sys.exit(1)
        plot_gripper_values(args.directory, args.plot_files, args.threshold, args.output)
    elif args.directory:
        # 批量模式
        inspect_hdf5_gripper_batch(args.directory, args.max_files, args.threshold, args.recursive)
    elif args.hdf5_file:
        # 单个文件模式
        inspect_hdf5_gripper(args.hdf5_file)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()




