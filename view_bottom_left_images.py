"""
使用 rerun 读取和展示 /home/wyz/bottom_left 目录下的图像文件。

Usage:
    python view_bottom_left_images.py
"""

import sys
from pathlib import Path

import numpy as np
import rerun as rr
from PIL import Image


def load_images_from_directory(directory: Path):
    """
    从目录中加载所有图像文件。
    
    Args:
        directory: 图像文件所在的目录路径
        
    Returns:
        list: 图像数组列表，按文件名排序
    """
    image_files = sorted(directory.glob("*.jpg")) + sorted(directory.glob("*.png")) + sorted(directory.glob("*.jpeg"))
    
    if not image_files:
        print(f"❌ 在目录 {directory} 中未找到图像文件")
        return []
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    images = []
    for img_path in image_files:
        try:
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img)
            images.append(img_array)
        except Exception as e:
            print(f"⚠️  无法加载图像 {img_path.name}: {e}")
    
    if images:
        print(f"成功加载 {len(images)} 个图像")
        print(f"图像形状: {images[0].shape}")
    
    return images


def visualize_images_with_rerun(images: list, directory: Path):
    """
    使用 rerun 可视化图像序列。
    
    Args:
        images: 图像数组列表
        directory: 图像目录路径（用于设置标题）
    """
    if not images:
        print("❌ 没有图像可可视化")
        return
    
    # 初始化 rerun
    rr.init(f"图像展示: {directory.name}", spawn=True)
    
    num_frames = len(images)
    print(f"\n可视化 {num_frames} 帧图像...")
    
    # 记录每一帧
    for frame_idx in range(num_frames):
        rr.set_time(sequence="frame", sequence_idx=frame_idx)
        
        # 记录图像
        img = images[frame_idx]
        if img.size > 0 and len(img.shape) >= 2:
            rr.log("image", rr.Image(img))
    
    print(f"\n✅ 可视化完成！共 {num_frames} 帧")
    print("可以在 rerun 窗口中播放、暂停和浏览图像序列")


def main():
    # 图像目录路径
    image_directory = Path("/home/wyz/bottom_left")
    
    if not image_directory.exists():
        print(f"❌ 目录不存在: {image_directory}")
        sys.exit(1)
    
    if not image_directory.is_dir():
        print(f"❌ 路径不是目录: {image_directory}")
        sys.exit(1)
    
    # 加载图像
    images = load_images_from_directory(image_directory)
    
    # 可视化
    visualize_images_with_rerun(images, image_directory)


if __name__ == "__main__":
    main()

