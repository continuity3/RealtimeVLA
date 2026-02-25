"""
可视化主视角VIT的attention热力图
用于分析不同任务阶段（初始轨迹、抓取、搬运）的attention分布
"""

import argparse
import pathlib
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2

# 添加项目路径
project_root = pathlib.Path(__file__).parent.parent.absolute()
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def load_model_and_policy(checkpoint_dir: str, config_name: str):
    """加载模型和策略"""
    print(f"正在加载模型: {checkpoint_dir}")
    print(f"配置: {config_name}")
    
    train_config = _config.get_config(config_name)
    checkpoint_path = pathlib.Path(checkpoint_dir)
    
    # 检查是否是PyTorch模型
    weight_path = checkpoint_path / "model.safetensors"
    is_pytorch = weight_path.exists()
    
    if not is_pytorch:
        print("⚠️  警告: 未找到PyTorch模型，将尝试使用JAX模型（可能无法提取attention）")
    
    # 创建策略
    policy = _policy_config.create_trained_policy(
        train_config,
        str(checkpoint_path),
    )
    
    return policy, is_pytorch


def extract_attention_weights_pytorch(model, image_tensor, layer_idx=None):
    """
    从PyTorch模型中提取attention weights
    
    Args:
        model: PyTorch模型
        image_tensor: 输入图像张量 (1, C, H, W)
        layer_idx: 要提取的层索引，None表示提取所有层
    
    Returns:
        attention_weights: 字典，包含各层的attention weights
    """
    attention_weights = {}
    
    # 获取vision tower
    vision_tower = model.paligemma_with_expert.paligemma.vision_tower
    
    # 存储attention weights的hook
    def get_attention_hook(name):
        def hook(module, input, output):
            if hasattr(output, 'attentions') and output.attentions is not None:
                attention_weights[name] = output.attentions
            elif isinstance(output, tuple) and len(output) > 1:
                # 某些情况下attention在tuple的第二个元素
                if isinstance(output[1], torch.Tensor):
                    attention_weights[name] = output[1]
        return hook
    
    # 注册hook到所有attention层
    hooks = []
    for i, layer in enumerate(vision_tower.vision_model.encoder.layers):
        if layer_idx is None or i == layer_idx:
            hook = layer.self_attn.register_forward_hook(get_attention_hook(f"layer_{i}"))
            hooks.append(hook)
    
    # 设置output_attentions=True
    vision_tower.vision_model.config.output_attentions = True
    
    # 前向传播
    with torch.no_grad():
        try:
            outputs = vision_tower(image_tensor, output_attentions=True)
            if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                # 如果模型直接返回了attentions
                for i, attn in enumerate(outputs.attentions):
                    attention_weights[f"layer_{i}"] = attn
        except Exception as e:
            print(f"⚠️  提取attention时出错: {e}")
            # 尝试手动计算attention
            attention_weights = compute_attention_manually(vision_tower, image_tensor, layer_idx)
    
    # 移除hooks
    for hook in hooks:
        hook.remove()
    
    vision_tower.vision_model.config.output_attentions = False
    
    return attention_weights


def compute_attention_manually(vision_tower, image_tensor, layer_idx=None):
    """
    手动计算attention weights（当模型不支持output_attentions时）
    """
    attention_weights = {}
    
    # 获取embeddings
    embeddings = vision_tower.vision_model.embeddings(image_tensor)
    
    # 通过每一层
    hidden_states = embeddings
    for i, layer in enumerate(vision_tower.vision_model.encoder.layers):
        if layer_idx is not None and i != layer_idx:
            hidden_states = layer(hidden_states)[0]
            continue
        
        # 计算self-attention
        residual = hidden_states
        hidden_states = layer.layer_norm1(hidden_states)
        
        # 获取Q, K, V
        query = layer.self_attn.q_proj(hidden_states)
        key = layer.self_attn.k_proj(hidden_states)
        value = layer.self_attn.v_proj(hidden_states)
        
        batch_size, seq_len, embed_dim = query.shape
        num_heads = layer.self_attn.num_heads
        head_dim = embed_dim // num_heads
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        # 计算attention scores
        scale = 1.0 / (head_dim ** 0.5)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        
        # Softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # 平均所有head
        attention_probs = attention_probs.mean(dim=1)  # (batch, seq_len, seq_len)
        
        attention_weights[f"layer_{i}"] = attention_probs
        
        # 继续前向传播
        hidden_states = layer(hidden_states)[0]
    
    return attention_weights


def reshape_attention_to_image(attention_weights, image_size=(224, 224), patch_size=16):
    """
    将attention weights重塑为图像空间
    
    Args:
        attention_weights: (seq_len, seq_len) 或 (batch, seq_len, seq_len)
        image_size: 原始图像大小 (H, W)
        patch_size: patch大小
    
    Returns:
        attention_map: (H, W) 的attention热力图
    """
    if attention_weights.ndim == 3:
        # 取第一个batch，并平均所有query位置
        attention_weights = attention_weights[0].mean(dim=0)  # (seq_len, seq_len)
    elif attention_weights.ndim == 2:
        # 平均所有query位置
        attention_weights = attention_weights.mean(dim=0)  # (seq_len,)
    
    # 计算patch网格大小
    h_patches = image_size[0] // patch_size
    w_patches = image_size[1] // patch_size
    num_patches = h_patches * w_patches
    
    # 如果attention包含CLS token，需要处理
    if attention_weights.shape[0] > num_patches:
        # 假设第一个token是CLS token，跳过它
        attention_weights = attention_weights[1:num_patches+1]
    
    # 重塑为patch网格
    attention_map = attention_weights[:num_patches].reshape(h_patches, w_patches)
    
    # 上采样到原始图像大小
    attention_map = cv2.resize(
        attention_map.cpu().numpy() if isinstance(attention_map, torch.Tensor) else attention_map,
        image_size[::-1],  # (W, H)
        interpolation=cv2.INTER_LINEAR
    )
    
    return attention_map


def visualize_attention_heatmap(
    image: np.ndarray,
    attention_weights: dict,
    output_path: str,
    layer_idx: int = -1,
    alpha: float = 0.6
):
    """
    可视化attention热力图
    
    Args:
        image: 原始图像 (H, W, C)
        attention_weights: 包含各层attention weights的字典
        output_path: 输出路径
        layer_idx: 要可视化的层索引，-1表示最后一层
        alpha: 热力图透明度
    """
    # 选择要可视化的层
    if layer_idx == -1:
        # 使用最后一层
        layer_keys = sorted(attention_weights.keys(), key=lambda x: int(x.split('_')[1]))
        layer_key = layer_keys[-1] if layer_keys else None
    else:
        layer_key = f"layer_{layer_idx}"
    
    if layer_key not in attention_weights:
        print(f"⚠️  未找到层 {layer_key}，可用层: {list(attention_weights.keys())}")
        if attention_weights:
            layer_key = list(attention_weights.keys())[-1]
        else:
            print("❌ 没有可用的attention weights")
            return
    
    attn = attention_weights[layer_key]
    
    # 重塑为图像空间
    if isinstance(image, np.ndarray):
        h, w = image.shape[:2]
    else:
        h, w = 224, 224
    
    attention_map = reshape_attention_to_image(attn, image_size=(h, w), patch_size=16)
    
    # 归一化到[0, 1]
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    
    # 创建热力图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 原始图像
    axes[0].imshow(image)
    axes[0].set_title("原始图像", fontsize=14)
    axes[0].axis('off')
    
    # Attention热力图
    im = axes[1].imshow(attention_map, cmap='jet', interpolation='bilinear')
    axes[1].set_title(f"Attention热力图 (层 {layer_key})", fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    # 叠加图像
    # 将attention map转换为RGB
    attention_colored = plt.cm.jet(attention_map)[:, :, :3]
    # 叠加
    overlay = (1 - alpha) * image / 255.0 + alpha * attention_colored
    axes[2].imshow(overlay)
    axes[2].set_title("叠加可视化", fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 保存热力图到: {output_path}")
    plt.close()


def load_image_from_parquet(parquet_path: str, frame_idx: int = 0):
    """从parquet文件加载图像"""
    import pandas as pd
    from PIL import Image
    from io import BytesIO
    
    df = pd.read_parquet(parquet_path)
    
    if "image" not in df.columns:
        raise ValueError("parquet文件中没有'image'列")
    
    img_data = df["image"].iloc[frame_idx]
    
    # 解码图像
    if isinstance(img_data, dict):
        if "bytes" in img_data:
            img_bytes = img_data["bytes"]
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
        elif "path" in img_data:
            img = Image.open(img_data["path"]).convert("RGB")
    elif isinstance(img_data, bytes):
        img = Image.open(BytesIO(img_data)).convert("RGB")
    elif isinstance(img_data, str):
        img = Image.open(img_data).convert("RGB")
    else:
        raise ValueError(f"无法解析图像数据: {type(img_data)}")
    
    return np.array(img)


def main():
    parser = argparse.ArgumentParser(description="可视化主视角VIT的attention热力图")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="模型checkpoint目录路径"
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="pi05_libero",
        help="配置名称（默认: pi05_libero）"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        help="输入图像路径（或parquet文件路径）"
    )
    parser.add_argument(
        "--parquet_path",
        type=str,
        help="parquet文件路径（如果从parquet加载）"
    )
    parser.add_argument(
        "--frame_idx",
        type=int,
        default=0,
        help="从parquet加载的帧索引（默认: 0）"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="attention_heatmap.png",
        help="输出图像路径（默认: attention_heatmap.png）"
    )
    parser.add_argument(
        "--layer_idx",
        type=int,
        default=-1,
        help="要可视化的层索引，-1表示最后一层（默认: -1）"
    )
    
    args = parser.parse_args()
    
    # 加载模型
    policy, is_pytorch = load_model_and_policy(args.checkpoint_dir, args.config_name)
    
    if not is_pytorch:
        print("❌ 当前只支持PyTorch模型的attention可视化")
        print("   请使用PyTorch checkpoint或等待JAX版本支持")
        return
    
    # 加载图像
    if args.parquet_path:
        image = load_image_from_parquet(args.parquet_path, args.frame_idx)
    elif args.image_path:
        image = np.array(Image.open(args.image_path).convert("RGB"))
    else:
        print("❌ 请提供 --image_path 或 --parquet_path")
        return
    
    # 预处理图像
    # 转换为模型输入格式
    model = policy.model
    if hasattr(model, 'paligemma_with_expert'):
        # 预处理图像
        from openpi.models_pytorch import preprocessing_pytorch as _preprocessing
        
        # 转换为tensor并归一化
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)  # (1, C, H, W)
        
        # 使用模型的预处理
        # 注意：这里需要根据实际模型调整预处理步骤
        if hasattr(model.paligemma_with_expert.paligemma.vision_tower, 'image_processor'):
            processor = model.paligemma_with_expert.paligemma.vision_tower.image_processor
            image_tensor = processor(image_tensor)
        else:
            # 默认预处理：resize到224x224并归一化
            image_tensor = F.interpolate(image_tensor, size=(224, 224), mode='bilinear', align_corners=False)
            # SigLIP的归一化
            mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
            std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
            image_tensor = (image_tensor - mean) / std
    
    # 提取attention weights
    print("正在提取attention weights...")
    attention_weights = extract_attention_weights_pytorch(model, image_tensor, args.layer_idx)
    
    if not attention_weights:
        print("❌ 未能提取到attention weights")
        return
    
    print(f"✅ 成功提取 {len(attention_weights)} 层的attention weights")
    print(f"   可用层: {list(attention_weights.keys())}")
    
    # 可视化
    print("正在生成热力图...")
    visualize_attention_heatmap(
        image,
        attention_weights,
        args.output_path,
        layer_idx=args.layer_idx
    )
    
    print("✅ 完成！")


if __name__ == "__main__":
    main()