#!/usr/bin/env python3
"""
比较 HDF5 文件中的图片数据数量和视频文件的帧数

用法:
    python3 compare_hdf5_video_frames.py <hdf5_file> <video_file>
    
示例:
    python3 compare_hdf5_video_frames.py /home/wyz/realsense_ws/BAG_STORAGE/recorded_bags/bag_20260121-185632/output.h5 /home/wyz/realsense_ws/BAG_STORAGE/ideo/head_20260121-185632.mp4
"""

import argparse
import sys
from pathlib import Path

try:
    import h5py
    import cv2
    import numpy as np
except ImportError as e:
    print(f"❌ 缺少必要的依赖: {e}")
    print("请安装: pip install h5py opencv-python numpy")
    sys.exit(1)


def count_images_in_hdf5(hdf5_path: Path) -> dict:
    """统计 HDF5 文件中的图片数据数量"""
    print(f"\n{'='*80}")
    print(f"📦 读取 HDF5 文件: {hdf5_path}")
    print(f"{'='*80}")
    
    if not hdf5_path.exists():
        print(f"❌ HDF5 文件不存在: {hdf5_path}")
        return {}
    
    image_counts = {}
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            print(f"✅ 文件打开成功")
            print(f"文件大小: {hdf5_path.stat().st_size / (1024*1024):.2f} MB")
            
            # 打印顶层结构
            print(f"\n顶层键: {list(f.keys())}")
            
            # 方法1: 检查 topics 结构（类似 rosbag 转换的格式）
            if 'topics' in f:
                print(f"\n🔍 检查 topics 结构...")
                topics = f['topics']
                print(f"找到 {len(topics)} 个 topics")
                
                # 查找图像相关的 topics
                image_topics = []
                for topic_name in topics.keys():
                    if any(keyword in topic_name.lower() for keyword in ['image', 'camera', 'rgb', 'color']):
                        image_topics.append(topic_name)
                
                print(f"\n找到 {len(image_topics)} 个图像相关的 topics:")
                for topic_name in image_topics:
                    topic_group = topics[topic_name]
                    
                    # 检查是否有 data 字段
                    if 'data' in topic_group:
                        data = topic_group['data']
                        if isinstance(data, h5py.Dataset):
                            shape = data.shape
                            print(f"  📸 {topic_name}:")
                            print(f"     Shape: {shape}")
                            print(f"     Dtype: {data.dtype}")
                            
                            # 判断是否是图像数据
                            if len(shape) >= 2:
                                # 可能是图像数组，第一维通常是帧数
                                num_frames = shape[0]
                                image_counts[topic_name] = {
                                    'count': num_frames,
                                    'shape': shape,
                                    'dtype': str(data.dtype)
                                }
                                print(f"     ✅ 图片数量: {num_frames}")
                            else:
                                print(f"     ⚠️  不是图像数据格式")
                    else:
                        print(f"  ⚠️  {topic_name}: 没有找到 data 字段")
            
            # 方法2: 检查 observations/images 结构（LeRobot 格式）
            if 'observations' in f:
                print(f"\n🔍 检查 observations 结构...")
                obs = f['observations']
                
                if 'images' in obs:
                    print(f"找到 observations/images 结构")
                    images_group = obs['images']
                    for cam_name in images_group.keys():
                        cam_data = images_group[cam_name]
                        if isinstance(cam_data, h5py.Dataset):
                            shape = cam_data.shape
                            num_frames = shape[0] if len(shape) > 0 else 0
                            image_counts[f"observations/images/{cam_name}"] = {
                                'count': num_frames,
                                'shape': shape,
                                'dtype': str(cam_data.dtype)
                            }
                            print(f"  📸 {cam_name}: {num_frames} 帧, shape: {shape}")
            
            # 方法3: 检查 data/demo_X/obs 结构（LIBERO 格式）
            if 'data' in f:
                print(f"\n🔍 检查 data 结构...")
                data_group = f['data']
                demos = [k for k in data_group.keys() if k.startswith('demo_')]
                print(f"找到 {len(demos)} 个演示")
                
                total_images = 0
                for demo_name in sorted(demos):
                    demo_group = data_group[demo_name]
                    if 'obs' in demo_group:
                        obs_group = demo_group['obs']
                        # 查找图像数据
                        for obs_key in obs_group.keys():
                            if any(keyword in obs_key.lower() for keyword in ['rgb', 'image', 'camera']):
                                obs_data = obs_group[obs_key]
                                if isinstance(obs_data, h5py.Dataset):
                                    shape = obs_data.shape
                                    num_frames = shape[0] if len(shape) > 0 else 0
                                    total_images += num_frames
                                    print(f"  📸 {demo_name}/{obs_key}: {num_frames} 帧")
                
                if total_images > 0:
                    image_counts['data/demos_total'] = {
                        'count': total_images,
                        'shape': None,
                        'dtype': None
                    }
            
    except Exception as e:
        print(f"❌ 读取 HDF5 文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return {}
    
    return image_counts


def count_video_frames(video_path: Path) -> int:
    """统计视频文件的帧数"""
    print(f"\n{'='*80}")
    print(f"🎬 读取视频文件: {video_path}")
    print(f"{'='*80}")
    
    if not video_path.exists():
        print(f"❌ 视频文件不存在: {video_path}")
        return -1
    
    try:
        # 使用 OpenCV 读取视频
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"❌ 无法打开视频文件")
            return -1
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        print(f"✅ 视频信息:")
        print(f"   分辨率: {width}x{height}")
        print(f"   帧率: {fps:.2f} FPS")
        print(f"   总帧数: {frame_count}")
        print(f"   时长: {duration:.2f} 秒")
        
        # 验证帧数（通过实际读取）
        print(f"\n🔍 验证帧数（实际读取）...")
        actual_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            actual_count += 1
            if actual_count % 100 == 0:
                print(f"   已读取 {actual_count} 帧...", end='\r')
        
        cap.release()
        print(f"\n✅ 实际读取帧数: {actual_count}")
        
        if frame_count != actual_count:
            print(f"⚠️  警告: 元数据中的帧数 ({frame_count}) 与实际帧数 ({actual_count}) 不一致")
            print(f"   使用实际读取的帧数: {actual_count}")
            return actual_count
        
        return frame_count
        
    except Exception as e:
        print(f"❌ 读取视频文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return -1


def main():
    parser = argparse.ArgumentParser(
        description="比较 HDF5 文件中的图片数据数量和视频文件的帧数"
    )
    parser.add_argument(
        "hdf5_file",
        type=Path,
        help="HDF5 文件路径"
    )
    parser.add_argument(
        "video_file",
        type=Path,
        help="视频文件路径"
    )
    
    args = parser.parse_args()
    
    # 统计 HDF5 中的图片数量
    image_counts = count_images_in_hdf5(args.hdf5_file)
    
    # 统计视频帧数
    video_frame_count = count_video_frames(args.video_file)
    
    # 比较结果
    print(f"\n{'='*80}")
    print(f"📊 比较结果")
    print(f"{'='*80}")
    
    if video_frame_count < 0:
        print(f"❌ 无法读取视频文件")
        return
    
    print(f"\n视频文件帧数: {video_frame_count}")
    
    if not image_counts:
        print(f"\n❌ HDF5 文件中没有找到图片数据")
        print(f"\n📝 分析:")
        print(f"   - 视频文件包含 {video_frame_count} 帧")
        print(f"   - HDF5 文件中没有图像数据，可能的原因:")
        print(f"     1. 图像数据存储在视频文件中，而不是 HDF5 文件中")
        print(f"     2. 转换过程中没有包含图像数据")
        print(f"     3. 图像数据存储在其他文件中")
        print(f"\n💡 建议:")
        print(f"   - 检查是否有其他 HDF5 文件包含图像数据")
        print(f"   - 检查 rosbag 原始文件中是否包含图像 topics")
        print(f"   - 如果图像数据在视频文件中，可能需要单独处理")
        print(f"{'='*80}\n")
        return
    
    print(f"\nHDF5 文件中的图片数据:")
    
    match_found = False
    for key, info in image_counts.items():
        count = info['count']
        match = "✅" if count == video_frame_count else "❌"
        diff = abs(count - video_frame_count)
        
        print(f"  {match} {key}: {count} 帧", end="")
        if count != video_frame_count:
            print(f" (差异: {diff} 帧, {diff/video_frame_count*100:.2f}%)")
        else:
            print()
            match_found = True
    
    print(f"\n{'='*80}")
    if match_found:
        print(f"✅ 找到匹配的数据！HDF5 中的图片数量与视频帧数一致")
    else:
        print(f"⚠️  未找到完全匹配的数据")
        print(f"\n📝 可能的原因:")
        print(f"   - HDF5 和视频的采样率不同")
        print(f"   - 数据采集时间不同步")
        print(f"   - 部分帧丢失或重复")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

