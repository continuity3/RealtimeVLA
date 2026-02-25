#!/usr/bin/env python3
"""
检查HDF5文件中是否有gripper/feedback_R数据

使用方法:
    python scripts/check_gripper_feedback_R.py <hdf5_file_path>
"""

import argparse
import h5py
import numpy as np


def check_gripper_feedback_R(hdf5_path: str):
    """
    检查HDF5文件中是否有gripper/feedback_R数据
    
    Args:
        hdf5_path: HDF5文件路径
    """
    print("=" * 80)
    print(f"📂 检查HDF5文件: {hdf5_path}")
    print("=" * 80)
    print()
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            # 可能的路径列表
            possible_paths = [
                "gripper/feedback_R",
                "topics/gripper/feedback_R",
                "topics/_gripper_feedback_R",
                "topics/gripper_feedback_R",
                "/gripper/feedback_R",
            ]
            
            found_paths = []
            
            # 检查每个可能的路径
            print("🔍 检查可能的路径:")
            for path in possible_paths:
                if path in f:
                    found_paths.append(path)
                    print(f"  ✅ 找到: {path}")
                else:
                    print(f"  ❌ 未找到: {path}")
            print()
            
            # 如果找到了，显示详细信息
            if found_paths:
                for path in found_paths:
                    print(f"📊 路径 '{path}' 的详细信息:")
                    obj = f[path]
                    
                    if isinstance(obj, h5py.Group):
                        print(f"   类型: Group")
                        print(f"   子键: {list(obj.keys())}")
                        
                        # 检查是否有data键
                        if "data" in obj:
                            data = obj["data"][:]
                            print(f"   data形状: {data.shape}")
                            print(f"   data类型: {data.dtype}")
                            if len(data) > 0:
                                print(f"   前5个值: {data[:5]}")
                                print(f"   最小值: {np.min(data):.6f}")
                                print(f"   最大值: {np.max(data):.6f}")
                                print(f"   平均值: {np.mean(data):.6f}")
                    elif isinstance(obj, h5py.Dataset):
                        print(f"   类型: Dataset")
                        print(f"   形状: {obj.shape}")
                        print(f"   类型: {obj.dtype}")
                        if obj.size > 0:
                            data = obj[:]
                            print(f"   前5个值: {data[:5] if data.ndim == 1 else data.flat[:5]}")
                            print(f"   最小值: {np.min(data):.6f}")
                            print(f"   最大值: {np.max(data):.6f}")
                            print(f"   平均值: {np.mean(data):.6f}")
                    print()
            else:
                print("⚠️  未找到任何匹配的路径")
                print()
                
                # 列出所有包含'gripper'或'feedback'的键
                print("🔍 搜索所有包含'gripper'或'feedback'的键:")
                gripper_keys = []
                feedback_keys = []
                
                def search_keys(name, obj):
                    name_lower = name.lower()
                    if 'gripper' in name_lower:
                        gripper_keys.append(name)
                    if 'feedback' in name_lower:
                        feedback_keys.append(name)
                
                f.visititems(search_keys)
                
                if gripper_keys:
                    print("  包含'gripper'的键:")
                    for key in sorted(set(gripper_keys))[:20]:  # 最多显示20个
                        print(f"    - {key}")
                else:
                    print("  未找到包含'gripper'的键")
                
                if feedback_keys:
                    print("  包含'feedback'的键:")
                    for key in sorted(set(feedback_keys))[:20]:  # 最多显示20个
                        print(f"    - {key}")
                else:
                    print("  未找到包含'feedback'的键")
                
                print()
                
                # 列出topics下的所有键
                print("📋 topics下的所有键:")
                if "topics" in f:
                    topics = list(f["topics"].keys())
                    for topic in sorted(topics):
                        print(f"    - {topic}")
                else:
                    print("   文件中没有 'topics' 组")
                
    except FileNotFoundError:
        print(f"❌ 错误: 文件不存在: {hdf5_path}")
        return False
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    print("=" * 80)
    if found_paths:
        print("✅ 找到了gripper/feedback_R相关数据")
    else:
        print("⚠️  未找到gripper/feedback_R相关数据")
    print("=" * 80)
    
    return len(found_paths) > 0


def main():
    parser = argparse.ArgumentParser(
        description="检查HDF5文件中是否有gripper/feedback_R数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 检查单个HDF5文件
  python scripts/check_gripper_feedback_R.py pick_blue_bottle_extracted/rosbag2_2026_01_09-21_24_59_0.h5

  # 检查多个文件
  for file in pick_blue_bottle_extracted/*.h5; do
      python scripts/check_gripper_feedback_R.py "$file"
  done
        """
    )
    
    parser.add_argument(
        'hdf5_file',
        type=str,
        help='HDF5文件路径'
    )
    
    args = parser.parse_args()
    
    check_gripper_feedback_R(args.hdf5_file)


if __name__ == '__main__':
    main()

















