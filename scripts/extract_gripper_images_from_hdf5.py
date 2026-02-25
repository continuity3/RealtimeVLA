#!/usr/bin/env python3
"""
从HDF5文件中提取gripper值大于0.9时对应的图片

使用方法:
    python scripts/extract_gripper_images_from_hdf5.py <hdf5_file> [--output_dir <dir>] [--threshold <value>]
"""

import argparse
import pathlib
import sys

import h5py
import numpy as np
from tqdm import tqdm

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️  cv2 not available. Install with: pip install opencv-python")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️  PIL not available. Install with: pip install Pillow")


def decode_image(img_data: np.ndarray, img_length: int) -> np.ndarray | None:
    """
    解码图像数据
    
    Args:
        img_data: 图像数据数组（扁平化）
        img_length: 图像数据的实际长度
    
    Returns:
        解码后的图像 (H, W, 3) uint8 或 None
    """
    img_bytes = bytes(img_data[:img_length])
    
    # 优先尝试作为 JPEG 解码
    if CV2_AVAILABLE:
        try:
            nparr = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                # OpenCV 返回 BGR，转换为 RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img
        except:
            pass
    
    # 尝试用 PIL
    if PIL_AVAILABLE:
        try:
            from io import BytesIO
            img = Image.open(BytesIO(img_bytes))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return np.array(img)
        except:
            pass
    
    # 如果不是 JPEG，尝试作为原始图像数据
    possible_sizes = [
        (720, 1280, 1),
        (480, 640, 3),
        (480, 854, 3),
        (360, 640, 3),
    ]
    
    for h, w, c in possible_sizes:
        if h * w * c == img_length:
            img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(h, w, c)
            if c == 1:
                img = np.repeat(img, 3, axis=2)
            return img
    
    # 如果都不匹配，尝试直接重塑为 640x480x3
    if img_length >= 640 * 480 * 3:
        img = np.frombuffer(img_bytes[:640*480*3], dtype=np.uint8).reshape(480, 640, 3)
        return img
    
    return None


def extract_gripper_images(hdf5_path: str, output_dir: str, threshold: float = 0.9):
    """
    从HDF5文件中提取gripper值大于threshold时对应的图片
    
    Args:
        hdf5_path: HDF5文件路径
        output_dir: 输出目录
        threshold: gripper阈值（默认0.9）
    """
    print("=" * 80)
    print(f"📂 处理HDF5文件: {hdf5_path}")
    print(f"📁 输出目录: {output_dir}")
    print(f"🎯 Gripper阈值: > {threshold}")
    print("=" * 80)
    print()
    
    # 创建输出目录
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            # 读取gripper数据
            gripper_topic_path = "topics/_control_gripperValueR"
            if gripper_topic_path not in f:
                print(f"❌ 错误: 未找到夹爪话题: {gripper_topic_path}")
                return
            
            gripper_topic = f[gripper_topic_path]
            if "data" not in gripper_topic:
                print(f"❌ 错误: 话题中没有 'data' 键")
                return
            
            gripper_data = gripper_topic["data"][:]  # (T,)
            print(f"✅ 读取夹爪数据: {len(gripper_data)} 个值")
            
            # 找到gripper值大于threshold的索引（排除NaN）
            valid_mask = ~np.isnan(gripper_data)
            gripper_valid = gripper_data[valid_mask]
            indices_valid = np.where(valid_mask)[0]
            
            # 找到大于threshold的索引
            high_gripper_mask = gripper_valid > threshold
            high_gripper_indices = indices_valid[high_gripper_mask]
            
            print(f"📊 统计:")
            print(f"   总数据点: {len(gripper_data)}")
            print(f"   有效数据点: {len(gripper_valid)}")
            print(f"   Gripper > {threshold} 的数据点: {len(high_gripper_indices)}")
            print()
            
            if len(high_gripper_indices) == 0:
                print(f"⚠️  没有找到gripper值大于{threshold}的数据点")
                return
            
            # 读取图像数据
            image_topic_path = "topics/_camera_camera_color_image_raw"
            if image_topic_path not in f:
                print(f"❌ 错误: 未找到图像话题: {image_topic_path}")
                return
            
            image_topic = f[image_topic_path]
            if "data" not in image_topic or "data_length" not in image_topic:
                print(f"❌ 错误: 图像话题中缺少 'data' 或 'data_length' 键")
                return
            
            image_data = image_topic["data"]  # (T, ...)
            image_lengths = image_topic["data_length"][:]  # (T,)
            
            print(f"✅ 读取图像数据: {len(image_data)} 张图像")
            print()
            
            # 提取并保存图片
            saved_count = 0
            failed_count = 0
            
            print(f"💾 开始提取图片...")
            for idx in tqdm(high_gripper_indices, desc="提取图片"):
                try:
                    # 获取gripper值
                    gripper_value = gripper_data[idx]
                    
                    # 读取图像数据
                    img_data = image_data[idx]
                    img_length = int(image_lengths[idx])
                    
                    # 解码图像
                    img = decode_image(img_data, img_length)
                    
                    if img is None:
                        print(f"⚠️  索引 {idx}: 图像解码失败")
                        failed_count += 1
                        continue
                    
                    # 保存图像
                    filename = f"gripper_{gripper_value:.6f}_idx_{idx:05d}.png"
                    filepath = output_path / filename
                    
                    if PIL_AVAILABLE:
                        img_pil = Image.fromarray(img)
                        img_pil.save(filepath)
                    elif CV2_AVAILABLE:
                        # OpenCV需要BGR格式
                        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(filepath), img_bgr)
                    else:
                        print(f"❌ 错误: 没有可用的图像保存库（需要PIL或OpenCV）")
                        return
                    
                    saved_count += 1
                    
                except Exception as e:
                    print(f"⚠️  索引 {idx}: 处理失败 - {e}")
                    failed_count += 1
                    continue
            
            print()
            print("=" * 80)
            print(f"✅ 提取完成!")
            print(f"   成功保存: {saved_count} 张图片")
            print(f"   失败: {failed_count} 张")
            print(f"   输出目录: {output_path}")
            print("=" * 80)
            
    except FileNotFoundError:
        print(f"❌ 错误: 文件不存在: {hdf5_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="从HDF5文件中提取gripper值大于0.9时对应的图片",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认阈值0.9
  python scripts/extract_gripper_images_from_hdf5.py \\
      pick_blue_bottle_extracted/rosbag2_2026_01_09-21_24_59_0.h5 \\
      --output_dir ./gripper_images

  # 自定义阈值
  python scripts/extract_gripper_images_from_hdf5.py \\
      pick_blue_bottle_extracted/rosbag2_2026_01_09-21_24_59_0.h5 \\
      --output_dir ./gripper_images \\
      --threshold 0.95

  # 处理多个文件
  for file in pick_blue_bottle_extracted/*.h5; do
      python scripts/extract_gripper_images_from_hdf5.py \\
          "$file" \\
          --output_dir "./gripper_images/$(basename $file .h5)"
  done
        """
    )
    
    parser.add_argument(
        'hdf5_file',
        type=str,
        help='HDF5文件路径'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./gripper_images',
        help='输出目录（默认: ./gripper_images）'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.9,
        help='Gripper阈值（默认: 0.9）'
    )
    
    args = parser.parse_args()
    
    extract_gripper_images(args.hdf5_file, args.output_dir, args.threshold)


if __name__ == '__main__':
    main()

