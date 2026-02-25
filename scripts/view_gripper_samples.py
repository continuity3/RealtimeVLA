#!/usr/bin/env python3
"""
查看训练数据中gripper > 0.9时的图片

使用方法:
    python scripts/view_gripper_samples.py <config_name> [--max_samples <num>] [--save_dir <dir>]
"""

import argparse
import pathlib
import sys

import numpy as np
import tyro

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️  PIL not available. Install with: pip install Pillow")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not available. Install with: pip install matplotlib")

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


def parse_image(image):
    """解析图片数据，确保是numpy数组格式 (H, W, C) uint8"""
    img = np.asarray(image)
    
    # 如果是torch tensor，转换为numpy
    if hasattr(img, 'cpu'):
        img = img.cpu().numpy()
    
    # 如果是CHW格式，转换为HWC
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    
    # 如果是float类型，转换为uint8
    if img.dtype == np.float32 or img.dtype == np.float64:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    
    # 确保是uint8
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    
    return img


def save_image(image, path):
    """保存图片"""
    if PIL_AVAILABLE:
        img = Image.fromarray(image)
        img.save(path)
    else:
        print(f"⚠️  Cannot save image (PIL not available): {path}")


def display_image(image, title=""):
    """显示图片"""
    if MATPLOTLIB_AVAILABLE:
        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print(f"⚠️  Cannot display image (matplotlib not available): {title}")


def main(config_name: str, max_samples: int = 10, save_dir: str | None = None, display: bool = False):
    """
    查看训练数据中gripper > 0.9时的图片
    
    Args:
        config_name: 训练配置名称
        max_samples: 最多显示多少个样本
        save_dir: 保存图片的目录（如果为None则不保存）
        display: 是否显示图片（需要matplotlib）
    """
    print(f"📊 Loading config: {config_name}")
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    
    print(f"📦 Creating dataset...")
    dataset = _data_loader.create_torch_dataset(
        data_config, 
        config.model.action_horizon, 
        config.model
    )
    
    # 应用repack transforms（将数据键名转换为模型期望的格式）
    from openpi.training.data_loader import TransformedDataset
    dataset = TransformedDataset(
        dataset,
        data_config.repack_transforms.inputs if data_config.repack_transforms else []
    )
    
    print(f"✅ Dataset created. Total samples: {len(dataset)}")
    print(f"🔍 Searching for samples with gripper > 0.9...")
    print()
    
    # 创建保存目录
    if save_dir:
        save_path = pathlib.Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"💾 Saving images to: {save_path}")
    
    found_samples = 0
    checked_samples = 0
    
    for idx, sample in enumerate(dataset):
        checked_samples += 1
        
        # 获取action（可能是batch格式）
        if isinstance(sample, dict):
            actions = sample.get("actions")
        else:
            actions = sample.actions if hasattr(sample, 'actions') else None
        
        if actions is None:
            continue
        
        # 处理不同的action格式
        if hasattr(actions, 'numpy'):
            actions_np = actions.numpy()
        elif hasattr(actions, 'cpu'):
            actions_np = actions.cpu().numpy()
        else:
            actions_np = np.asarray(actions)
        
        # 如果是batch格式，取第一个
        if actions_np.ndim > 1:
            action = actions_np[0] if actions_np.ndim == 2 else actions_np[0, 0]
        else:
            action = actions_np
        
        # 检查gripper值（第8个维度，索引7）
        if len(action) >= 8:
            gripper_value = action[7]
        elif len(action) == 7:
            # 如果只有7维，可能是没有gripper，跳过
            continue
        else:
            print(f"⚠️  Unexpected action dimension: {len(action)}")
            continue
        
        # 检查gripper是否大于0.9（接近关闭状态）
        if gripper_value > 0.9:
            found_samples += 1
            print(f"✅ Sample {found_samples}/{max_samples} (dataset index: {idx})")
            print(f"   Action: {action}")
            print(f"   Gripper value: {gripper_value:.4f}")
            
            # 获取图片（尝试多种可能的键名）
            base_image = None
            wrist_image = None
            
            if isinstance(sample, dict):
                # 尝试不同的键名
                for key in ["observation/image", "image", "observation"]:
                    if key in sample:
                        val = sample[key]
                        if isinstance(val, dict) and "image" in val:
                            base_image = val["image"]
                            break
                        elif key == "image" or key == "observation/image":
                            base_image = val
                            break
                
                for key in ["observation/wrist_image", "wrist_image"]:
                    if key in sample:
                        val = sample[key]
                        if isinstance(val, dict) and "wrist_image" in val:
                            wrist_image = val["wrist_image"]
                            break
                        elif key == "wrist_image" or key == "observation/wrist_image":
                            wrist_image = val
                            break
            else:
                base_image = getattr(sample, 'image', None)
                wrist_image = getattr(sample, 'wrist_image', None)
            
            # 解析图片
            if base_image is not None:
                base_img = parse_image(base_image)
                print(f"   Base image shape: {base_img.shape}")
                
                if save_dir:
                    save_image(base_img, save_path / f"sample_{found_samples}_base.png")
                
                if display:
                    display_image(base_img, f"Sample {found_samples} - Base Image (Gripper={gripper_value:.2f})")
            
            if wrist_image is not None:
                wrist_img = parse_image(wrist_image)
                print(f"   Wrist image shape: {wrist_img.shape}")
                
                if save_dir:
                    save_image(wrist_img, save_path / f"sample_{found_samples}_wrist.png")
                
                if display:
                    display_image(wrist_img, f"Sample {found_samples} - Wrist Image (Gripper={gripper_value:.2f})")
            
            print()
            
            if found_samples >= max_samples:
                break
        
        # 每检查1000个样本打印一次进度
        if checked_samples % 1000 == 0:
            print(f"   Checked {checked_samples} samples, found {found_samples} with gripper > 0.9...")
    
    print()
    print("=" * 80)
    print(f"📊 Summary:")
    print(f"   Total samples checked: {checked_samples}")
    print(f"   Samples with gripper > 0.9: {found_samples}")
    if save_dir:
        print(f"   Images saved to: {save_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="View training data samples where gripper > 0.9",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 查看10个gripper=1的样本
  python scripts/view_gripper_samples.py pi05_pick_blue_bottle_libero_downsample4x

  # 查看20个样本并保存图片
  python scripts/view_gripper_samples.py pi05_pick_blue_bottle_libero_downsample4x \\
      --max_samples 20 \\
      --save_dir ./gripper_samples

  # 查看并显示图片（需要matplotlib）
  python scripts/view_gripper_samples.py pi05_pick_blue_bottle_libero_downsample4x \\
      --max_samples 5 \\
      --display
        """
    )
    
    parser.add_argument(
        'config_name',
        type=str,
        help='Training config name (e.g., pi05_pick_blue_bottle_libero_downsample4x)'
    )
    
    parser.add_argument(
        '--max_samples',
        type=int,
        default=10,
        help='Maximum number of samples to find and display (default: 10)'
    )
    
    parser.add_argument(
        '--save_dir',
        type=str,
        default=None,
        help='Directory to save images (default: None, do not save)'
    )
    
    parser.add_argument(
        '--display',
        action='store_true',
        help='Display images using matplotlib (requires matplotlib)'
    )
    
    args = parser.parse_args()
    
    main(
        config_name=args.config_name,
        max_samples=args.max_samples,
        save_dir=args.save_dir,
        display=args.display
    )

