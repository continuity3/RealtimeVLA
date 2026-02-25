"""
转换 LIBERO HDF5 格式数据到 LeRobot 格式的脚本。

如果你的数据是 LIBERO 风格的 HDF5 格式，可以使用这个脚本。

Usage:
uv run examples/libero/convert_libero_hdf5_to_lerobot.py --data_dir /path/to/your/hdf5/files

如果你的 HDF5 文件结构不同，需要修改下面的读取逻辑。

HDF5 文件结构示例（LIBERO 风格）:
data/
    demo_1/
        obs/
            agentview_rgb: (T, H, W, 3) - 主视角图像
            eye_in_hand_rgb: (T, H, W, 3) - 手腕相机图像
            joint_states: (T, 7) - 关节状态
            gripper_states: (T, 1) - 夹爪状态
        actions: (T, 7) - 动作
        states: (T, ...) - 状态（可选）
    demo_2/
    ...
"""

import shutil
from pathlib import Path

import h5py
import numpy as np
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image
from tqdm import tqdm
import tyro

REPO_NAME = "your_hf_username/libero_hdf5"  # 输出数据集名称


def resize_image(image: np.ndarray, target_size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """调整图像大小到目标尺寸"""
    if image.shape[:2] == target_size:
        return image
    img = Image.fromarray(image)
    img = img.resize(target_size, resample=Image.BICUBIC)
    return np.array(img)


def load_hdf5_episode(hdf5_file: h5py.File, demo_key: str) -> list[dict]:
    """
    从 HDF5 文件中加载一个 episode 的数据。
    
    根据你的 HDF5 结构修改这个函数。
    
    Args:
        hdf5_file: 打开的 HDF5 文件对象
        demo_key: demo 的键名，如果为空字符串，则从根目录读取
    """
    # 如果 demo_key 为空，使用根目录
    if demo_key == "" or demo_key == "root":
        demo_group = hdf5_file
    else:
        demo_group = hdf5_file[demo_key]
    
    # 读取图像数据
    # 根据你的 HDF5 结构修改这些路径
    if "obs/agentview_rgb" in demo_group:
        agentview_images = demo_group["obs/agentview_rgb"][:]  # (T, H, W, 3)
    elif "obs/agentview_rgb" in hdf5_file:
        # 如果图像在根目录
        agentview_images = hdf5_file["obs/agentview_rgb"][:]
    else:
        raise KeyError("找不到 agentview_rgb 图像数据")
    
    if "obs/eye_in_hand_rgb" in demo_group:
        wrist_images = demo_group["obs/eye_in_hand_rgb"][:]  # (T, H, W, 3)
    elif "obs/eye_in_hand_rgb" in hdf5_file:
        wrist_images = hdf5_file["obs/eye_in_hand_rgb"][:]
    else:
        # 如果没有手腕相机，使用主相机
        wrist_images = agentview_images.copy()
    
    # 读取状态数据
    if "obs/joint_states" in demo_group:
        joint_states = demo_group["obs/joint_states"][:]  # (T, 7)
    elif "obs/joint_states" in hdf5_file:
        joint_states = hdf5_file["obs/joint_states"][:]
    else:
        raise KeyError("找不到 joint_states 数据")
    
    # 读取夹爪状态（可选）
    if "obs/gripper_states" in demo_group:
        gripper_states = demo_group["obs/gripper_states"][:]  # (T, 1)
    elif "obs/gripper_states" in hdf5_file:
        gripper_states = hdf5_file["obs/gripper_states"][:]
    else:
        # 如果没有夹爪状态，使用零
        gripper_states = np.zeros((len(joint_states), 1), dtype=np.float32)
    
    # 读取动作数据
    if "actions" in demo_group:
        actions = demo_group["actions"][:]  # (T, 7)
    elif "actions" in hdf5_file:
        actions = hdf5_file["actions"][:]
    else:
        raise KeyError("找不到 actions 数据")
    
    # 读取任务描述（如果有）
    task_description = "Do something"  # 默认任务描述
    if "task" in demo_group.attrs:
        task_description = demo_group.attrs["task"]
    elif "task" in hdf5_file.attrs:
        task_description = hdf5_file.attrs["task"]
    
    # 确保所有数据长度一致
    min_length = min(
        len(agentview_images),
        len(wrist_images),
        len(joint_states),
        len(actions),
    )
    
    # 组合状态（关节 + 夹爪）
    states = np.concatenate([joint_states[:min_length], gripper_states[:min_length]], axis=1)  # (T, 8)
    
    # 转换为步骤列表
    steps = []
    for i in range(min_length):
        steps.append({
            "image": agentview_images[i],
            "wrist_image": wrist_images[i],
            "state": states[i],
            "action": actions[i],
            "task": task_description,
        })
    
    return steps


def main(data_dir: str, *, push_to_hub: bool = False):
    """
    主函数：将 HDF5 格式的 LIBERO 数据转换为 LeRobot 格式
    
    Args:
        data_dir: HDF5 文件所在的目录
        push_to_hub: 是否推送到 Hugging Face Hub
    """
    data_dir = Path(data_dir)
    
    # 清理输出目录
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)
    
    # 创建 LeRobot 数据集
    # OpenPi 假设状态存储在 `state`，动作存储在 `action`
    # LeRobot 假设图像数据的 dtype 是 `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,  # 根据你的数据采集频率调整
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),  # 7 关节 + 1 夹爪
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),  # 根据你的动作维度调整
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
    
    # 查找所有 HDF5 文件
    hdf5_files = list(data_dir.glob("*.hdf5")) + list(data_dir.glob("*.h5"))
    if not hdf5_files:
        raise ValueError(f"在 {data_dir} 中找不到 HDF5 文件")
    
    print(f"找到 {len(hdf5_files)} 个 HDF5 文件")
    
    # 遍历所有 HDF5 文件
    for hdf5_path in tqdm(hdf5_files, desc="处理 HDF5 文件"):
        with h5py.File(hdf5_path, "r") as hdf5_file:
            # 查找所有 demo/episode
            # 根据你的 HDF5 结构修改这部分
            demo_keys = []
            
            # 方式 1: 如果 demos 在 "data" 组下
            if "data" in hdf5_file:
                data_group = hdf5_file["data"]
                demo_keys = [key for key in data_group.keys() if key.startswith("demo_")]
            # 方式 2: 如果 demos 直接在根目录
            else:
                demo_keys = [key for key in hdf5_file.keys() if key.startswith("demo_")]
            
            # 如果没有找到 demo_* 格式，尝试其他格式
            if not demo_keys:
                # 尝试查找所有组（可能是其他命名方式）
                demo_keys = [key for key in hdf5_file.keys() if isinstance(hdf5_file[key], h5py.Group)]
            
            # 如果还是没有，可能整个文件就是一个 episode
            if not demo_keys:
                print(f"警告: {hdf5_path} 中没有找到 demo 组，尝试作为单个 episode 处理")
                demo_keys = ["root"]  # 使用特殊标记
            
            # 处理每个 episode
            for demo_key in demo_keys:
                try:
                    if demo_key == "root":
                        # 整个文件作为一个 episode
                        steps = load_hdf5_episode(hdf5_file, "")
                    else:
                        steps = load_hdf5_episode(hdf5_file, f"data/{demo_key}" if "data" in hdf5_file else demo_key)
                    
                    # 写入 LeRobot 数据集
                    for step in steps:
                        # 调整图像大小（如果需要）
                        image = resize_image(step["image"], (256, 256))
                        wrist_image = resize_image(step["wrist_image"], (256, 256))
                        
                        # 确保图像是 uint8 格式
                        if image.dtype != np.uint8:
                            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
                        if wrist_image.dtype != np.uint8:
                            wrist_image = (wrist_image * 255).astype(np.uint8) if wrist_image.max() <= 1.0 else wrist_image.astype(np.uint8)
                        
                        dataset.add_frame({
                            "image": image,
                            "wrist_image": wrist_image,
                            "state": step["state"].astype(np.float32),
                            "actions": step["action"].astype(np.float32),
                            "task": step["task"],
                        })
                    
                    dataset.save_episode()
                    print(f"✅ 成功转换 {hdf5_path} 中的 {demo_key} ({len(steps)} 步)")
                    
                except Exception as e:
                    print(f"❌ 处理 {hdf5_path} 中的 {demo_key} 时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    print(f"✅ 转换完成！数据集保存在: {output_path}")
    
    # 可选：推送到 Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "hdf5"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print("✅ 已推送到 Hugging Face Hub")


if __name__ == "__main__":
    tyro.cli(main)

