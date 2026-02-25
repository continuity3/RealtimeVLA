"""
从 LeRobot parquet 文件中恢复数据。

这个脚本可以从已转换的 LeRobot 数据集中读取数据，并可以：
1. 导出为 numpy 数组
2. 导出为 HDF5 格式
3. 导出图像为单独的文件
4. 可视化数据

Usage:
    # 从 parquet 文件恢复数据
    python recover_data_from_lerobot.py --parquet_path /path/to/episode_000000.parquet --output_dir /path/to/output

    # 从数据集目录恢复所有 episode
    python recover_data_from_lerobot.py --dataset_path /path/to/dataset --output_dir /path/to/output
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tyro
from PIL import Image
from tqdm import tqdm


def decode_image(img_data):
    """解码图像数据，支持多种格式"""
    from io import BytesIO
    
    # 如果是字典（lerobot 格式）
    if isinstance(img_data, dict):
        # 优先使用 bytes
        if "bytes" in img_data and img_data["bytes"] is not None:
            img_bytes = img_data["bytes"]
            if isinstance(img_bytes, bytes):
                # 尝试作为 JPEG/PNG 解码
                try:
                    img = np.array(Image.open(BytesIO(img_bytes)).convert("RGB"))
                    return img
                except:
                    # 如果不是压缩格式，尝试作为原始数组
                    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                    if len(img_array) == 256 * 256 * 3:
                        return img_array.reshape(256, 256, 3)
                    elif len(img_array) == 480 * 640 * 3:
                        return img_array.reshape(480, 640, 3)
        # 如果没有 bytes，尝试使用 path
        if "path" in img_data and img_data["path"] is not None:
            img_path = img_data["path"]
            # 如果是相对路径，需要找到数据集的根目录
            if Path(img_path).exists():
                return np.array(Image.open(img_path).convert("RGB"))
    
    # 如果是字节
    if isinstance(img_data, bytes):
        try:
            return np.array(Image.open(BytesIO(img_data)).convert("RGB"))
        except:
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            if len(img_array) == 256 * 256 * 3:
                return img_array.reshape(256, 256, 3)
            elif len(img_array) == 480 * 640 * 3:
                return img_array.reshape(480, 640, 3)
    
    # 如果是数组
    if isinstance(img_data, (list, np.ndarray)):
        img = np.array(img_data)
        if img.size > 0:
            return img
    
    # 如果是字符串路径
    if isinstance(img_data, str):
        if Path(img_data).exists():
            return np.array(Image.open(img_data).convert("RGB"))
    
    return None


def read_parquet_file(parquet_path: Path) -> dict:
    """
    读取 parquet 文件并返回数据字典。
    
    Returns:
        dict: 包含 images, wrist_images, states, actions, task 的字典
    """
    print(f"正在读取: {parquet_path}")
    
    df = pd.read_parquet(parquet_path)
    print(f"数据行数: {len(df)}")
    print(f"列名: {list(df.columns)}")
    
    data = {}
    
    # 读取图像
    if "image" in df.columns:
        images = []
        for idx, img_data in enumerate(tqdm(df["image"], desc="解码图像", leave=False)):
            img = decode_image(img_data)
            if img is None:
                print(f"⚠️  无法解析图像 {idx}")
                img = np.zeros((256, 256, 3), dtype=np.uint8)
            images.append(img)
        data["images"] = np.array(images)
        print(f"图像形状: {data['images'].shape}")
    else:
        data["images"] = None
    
    # 读取手腕图像
    if "wrist_image" in df.columns:
        wrist_images = []
        for idx, img_data in enumerate(tqdm(df["wrist_image"], desc="解码手腕图像", leave=False)):
            img = decode_image(img_data)
            if img is None:
                print(f"⚠️  无法解析手腕图像 {idx}")
                img = np.zeros((256, 256, 3), dtype=np.uint8)
            wrist_images.append(img)
        data["wrist_images"] = np.array(wrist_images)
        print(f"手腕图像形状: {data['wrist_images'].shape}")
    else:
        data["wrist_images"] = None
    
    # 读取状态
    if "state" in df.columns:
        states = []
        for state in df["state"]:
            if isinstance(state, (list, np.ndarray)):
                states.append(np.array(state))
            else:
                states.append(state)
        data["states"] = np.array(states)
        print(f"状态形状: {data['states'].shape}")
    else:
        data["states"] = None
    
    # 读取动作
    if "actions" in df.columns:
        actions = []
        for action in df["actions"]:
            if isinstance(action, (list, np.ndarray)):
                actions.append(np.array(action))
            else:
                actions.append(action)
        data["actions"] = np.array(actions)
        print(f"动作形状: {data['actions'].shape}")
    else:
        data["actions"] = None
    
    # 读取任务描述
    if "task" in df.columns and len(df) > 0:
        data["task"] = df["task"].iloc[0]
    else:
        data["task"] = ""
    
    print(f"任务: {data['task']}")
    
    return data


def save_data(data: dict, output_dir: Path, episode_name: str = "episode"):
    """
    保存恢复的数据到文件。
    
    Args:
        data: 数据字典
        output_dir: 输出目录
        episode_name: episode 名称（用于文件命名）
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n保存数据到: {output_dir}")
    
    # 保存状态
    if data["states"] is not None:
        states_path = output_dir / f"{episode_name}_states.npy"
        np.save(states_path, data["states"])
        print(f"✅ 状态已保存: {states_path} ({data['states'].shape})")
        
        # 也保存为文本文件（便于查看）
        states_txt_path = output_dir / f"{episode_name}_states.txt"
        np.savetxt(states_txt_path, data["states"], fmt="%.6f")
        print(f"✅ 状态文本已保存: {states_txt_path}")
    
    # 保存动作
    if data["actions"] is not None:
        actions_path = output_dir / f"{episode_name}_actions.npy"
        np.save(actions_path, data["actions"])
        print(f"✅ 动作已保存: {actions_path} ({data['actions'].shape})")
        
        # 也保存为文本文件
        actions_txt_path = output_dir / f"{episode_name}_actions.txt"
        np.savetxt(actions_txt_path, data["actions"], fmt="%.6f")
        print(f"✅ 动作文本已保存: {actions_txt_path}")
    
    # 保存图像
    if data["images"] is not None:
        images_dir = output_dir / f"{episode_name}_images"
        images_dir.mkdir(exist_ok=True)
        for i, img in enumerate(tqdm(data["images"], desc="保存图像", leave=False)):
            img_path = images_dir / f"frame_{i:06d}.png"
            Image.fromarray(img).save(img_path)
        print(f"✅ 图像已保存: {images_dir} ({len(data['images'])} 张)")
    
    # 保存手腕图像
    if data["wrist_images"] is not None:
        wrist_images_dir = output_dir / f"{episode_name}_wrist_images"
        wrist_images_dir.mkdir(exist_ok=True)
        for i, img in enumerate(tqdm(data["wrist_images"], desc="保存手腕图像", leave=False)):
            img_path = wrist_images_dir / f"frame_{i:06d}.png"
            Image.fromarray(img).save(img_path)
        print(f"✅ 手腕图像已保存: {wrist_images_dir} ({len(data['wrist_images'])} 张)")
    
    # 保存任务描述
    if data["task"]:
        task_path = output_dir / f"{episode_name}_task.txt"
        with open(task_path, "w") as f:
            f.write(data["task"])
        print(f"✅ 任务描述已保存: {task_path}")
    
    # 保存元数据
    metadata = {
        "num_frames": len(data["states"]) if data["states"] is not None else 0,
        "state_shape": list(data["states"].shape) if data["states"] is not None else None,
        "action_shape": list(data["actions"].shape) if data["actions"] is not None else None,
        "image_shape": list(data["images"].shape) if data["images"] is not None else None,
        "task": data["task"],
    }
    metadata_path = output_dir / f"{episode_name}_metadata.txt"
    with open(metadata_path, "w") as f:
        f.write("元数据:\n")
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    print(f"✅ 元数据已保存: {metadata_path}")


def recover_from_parquet(
    parquet_path: str,
    output_dir: str,
):
    """
    从单个 parquet 文件恢复数据。
    
    Args:
        parquet_path: parquet 文件路径
        output_dir: 输出目录
    """
    parquet_path = Path(parquet_path).resolve()
    if not parquet_path.exists():
        raise FileNotFoundError(f"文件不存在: {parquet_path}")
    
    # 读取数据
    data = read_parquet_file(parquet_path)
    
    # 保存数据
    episode_name = parquet_path.stem  # 例如 "episode_000000"
    save_data(data, output_dir, episode_name)
    
    print(f"\n✅ 数据恢复完成！")


def recover_from_dataset(
    dataset_path: str,
    output_dir: str,
):
    """
    从数据集目录恢复所有 episode 的数据。
    
    Args:
        dataset_path: 数据集根目录路径
        output_dir: 输出目录
    """
    dataset_path = Path(dataset_path).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"目录不存在: {dataset_path}")
    
    # 查找所有 parquet 文件
    # LeRobot 数据集通常在 data/chunk-*/ 目录下
    parquet_files = sorted(
        list(dataset_path.glob("**/episode_*.parquet"))
    )
    
    if not parquet_files:
        raise FileNotFoundError(f"在 '{dataset_path}' 中找不到任何 parquet 文件")
    
    print(f"找到 {len(parquet_files)} 个 parquet 文件")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 恢复每个 episode
    for parquet_file in tqdm(parquet_files, desc="恢复数据"):
        try:
            data = read_parquet_file(parquet_file)
            episode_name = parquet_file.stem
            save_data(data, output_dir, episode_name)
        except Exception as e:
            print(f"❌ 处理 {parquet_file} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✅ 所有数据恢复完成！共 {len(parquet_files)} 个 episode")


def main(
    parquet_path: Optional[str] = None,
    dataset_path: Optional[str] = None,
    output_dir: str = "./recovered_data",
):
    """
    主函数：从 LeRobot 数据恢复数据。
    
    Args:
        parquet_path: 单个 parquet 文件路径（与 dataset_path 二选一）
        dataset_path: 数据集根目录路径（与 parquet_path 二选一）
        output_dir: 输出目录
    """
    if parquet_path is not None:
        recover_from_parquet(parquet_path, output_dir)
    elif dataset_path is not None:
        recover_from_dataset(dataset_path, output_dir)
    else:
        print("❌ 请提供 parquet_path 或 dataset_path 参数")
        print("\n用法:")
        print("  # 从单个 parquet 文件恢复")
        print("  python recover_data_from_lerobot.py --parquet_path /path/to/episode_000000.parquet --output_dir ./output")
        print("\n  # 从数据集目录恢复所有 episode")
        print("  python recover_data_from_lerobot.py --dataset_path /path/to/dataset --output_dir ./output")
        sys.exit(1)


if __name__ == "__main__":
    tyro.cli(main)

