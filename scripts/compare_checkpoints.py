#!/usr/bin/env python3
"""
比较不同训练步数的checkpoint使用方式

说明新跑完的权重（20000步）和之前的权重用起来的区别

使用方法:
    python scripts/compare_checkpoints.py
"""

import pathlib

def compare_checkpoints():
    """
    比较不同checkpoint的使用方式
    """
    checkpoint_dir = pathlib.Path("checkpoints/pi05_pick_blue_bottle_libero_downsample4x/downsample4x_right_arm_finetune_30k")
    
    print("=" * 80)
    print("📊 Checkpoint 比较说明")
    print("=" * 80)
    print()
    
    # 查找所有checkpoint
    checkpoints = sorted([int(d.name) for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    
    if not checkpoints:
        print("❌ 未找到任何checkpoint")
        return
    
    print(f"📂 找到的checkpoint步数: {checkpoints}")
    print()
    
    # 检查每个checkpoint的文件
    print("📋 Checkpoint文件结构:")
    print("-" * 80)
    for step in checkpoints[-5:]:  # 只显示最后5个
        ckpt_path = checkpoint_dir / str(step)
        if ckpt_path.exists():
            files = list(ckpt_path.iterdir())
            model_size = None
            optimizer_size = None
            for f in files:
                if f.name == "model.safetensors":
                    model_size = f.stat().st_size / (1024**3)  # GB
                elif f.name == "optimizer.pt":
                    optimizer_size = f.stat().st_size / (1024**3)  # GB
            
            print(f"  Step {step}:")
            print(f"    - model.safetensors: {model_size:.2f} GB" if model_size else "    - model.safetensors: 未找到")
            print(f"    - optimizer.pt: {optimizer_size:.2f} GB" if optimizer_size else "    - optimizer.pt: 未找到")
            print(f"    - 其他文件: {[f.name for f in files if f.name not in ['model.safetensors', 'optimizer.pt', 'metadata.pt']]}")
    print()
    
    # 说明区别
    print("=" * 80)
    print("🔍 不同训练步数checkpoint的区别:")
    print("=" * 80)
    print()
    print("1. **训练步数差异**:")
    print("   - 20000步: 训练进行到2/3的位置")
    print("   - 30000步: 训练完成（目标步数）")
    print("   - 更早的步数（如10000, 15000）: 训练早期/中期")
    print()
    print("2. **模型性能差异**:")
    print("   - 通常来说，训练步数越多，模型在训练数据上的表现越好")
    print("   - 但也要注意过拟合：如果验证集性能不再提升，可能已经过拟合")
    print("   - 20000步 vs 30000步: 可能30000步的模型更接近收敛")
    print()
    print("3. **使用方式**:")
    print("   - 所有checkpoint的使用方式完全相同")
    print("   - 只需要在serve_policy脚本中指定不同的checkpoint路径")
    print("   - 模型结构、输入输出格式都相同")
    print()
    print("4. **如何选择checkpoint**:")
    print("   - 查看wandb日志，找到验证集性能最好的checkpoint")
    print("   - 或者使用最终checkpoint（30000步）")
    print("   - 如果训练还在进行，可以使用最新的checkpoint（20000步）")
    print()
    print("=" * 80)
    print("📝 使用不同checkpoint的方法:")
    print("=" * 80)
    print()
    
    # 生成使用示例
    latest_checkpoint = max(checkpoints)
    print(f"使用最新的checkpoint ({latest_checkpoint}步):")
    print(f"  uv run scripts/serve_policy1.py \\")
    print(f"    --policy.path checkpoints/pi05_pick_blue_bottle_libero_downsample4x/downsample4x_right_arm_finetune_30k/{latest_checkpoint} \\")
    print(f"    --policy.config pi05_pick_blue_bottle_libero_downsample4x")
    print()
    
    if len(checkpoints) > 1:
        previous_checkpoint = checkpoints[-2] if len(checkpoints) > 1 else None
        if previous_checkpoint:
            print(f"使用之前的checkpoint ({previous_checkpoint}步):")
            print(f"  uv run scripts/serve_policy1.py \\")
            print(f"    --policy.path checkpoints/pi05_pick_blue_bottle_libero_downsample4x/downsample4x_right_arm_finetune_30k/{previous_checkpoint} \\")
            print(f"    --policy.config pi05_pick_blue_bottle_libero_downsample4x")
            print()
    
    print("=" * 80)
    print("⚠️  重要提示:")
    print("=" * 80)
    print("1. 所有checkpoint的模型结构相同，只是权重不同")
    print("2. 输入输出格式完全相同（8维state，8维action）")
    print("3. 推理代码不需要修改，只需要改变checkpoint路径")
    print("4. 建议查看wandb日志，选择性能最好的checkpoint")
    print("=" * 80)


if __name__ == "__main__":
    compare_checkpoints()















