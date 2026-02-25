"""
转换 ROS2 bag 转换的 HDF5 格式数据到 LeRobot 格式的脚本。

这个脚本专门用于处理从 ROS2 bag 转换来的 HDF5 文件。

HDF5 文件结构:
- time: (T,) 时间戳
- topics/_joint_states/:
    - position: (T, 14) 关节位置
    - velocity: (T, 14) 关节速度
    - effort: (T, 14) 关节力矩
- topics/_usb_cam_0_image_raw/:
    - data: (T, H, W, 3) 图像数据

Usage:
uv run examples/libero/convert_ros2bag_hdf5_to_lerobot.py --data_dir /path/to/your/hdf5/files
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

REPO_NAME = "your_hf_username/ros2bag_libero"  # 输出数据集名称


def resize_image(image: np.ndarray, target_size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """调整图像大小到目标尺寸"""
    if image.shape[:2] == target_size:
        return image
    img = Image.fromarray(image)
    img = img.resize(target_size, resample=Image.BICUBIC)
    return np.array(img)


def compute_actions_from_states(positions: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    """
    从关节位置和速度计算动作。
    
    对于 LIBERO，通常使用位置增量（delta）或速度作为动作。
    这里我们使用速度作为动作（如果可用），否则使用位置差分。
    """
    if velocities is not None and len(velocities) > 0:
        # 使用速度作为动作
        return velocities
    else:
        # 使用位置差分作为动作
        actions = np.diff(positions, axis=0, prepend=positions[0:1])
        return actions


def load_ros2bag_hdf5(hdf5_path: Path, task_description: str = "Do something", ignore_valid: bool = False) -> list[dict]:
    """
    从 ROS2 bag HDF5 文件中加载数据。
    
    Args:
        hdf5_path: HDF5 文件路径
        task_description: 任务描述（如果没有在文件中找到）
    
    Returns:
        步骤列表，每个步骤包含 image, wrist_image, state, action, task
    """
    with h5py.File(hdf5_path, "r") as f:
        # 读取关节状态
        if "_joint_states" not in f["topics"]:
            raise KeyError("找不到 _joint_states topic")
        
        joint_states = f["topics/_joint_states"]
        positions = joint_states["position"][:]  # (T, 14)
        velocities = joint_states["velocity"][:]  # (T, 14)
        effort = joint_states["effort"][:]  # (T, 14)
        
        # 读取图像
        if "_usb_cam_0_image_raw" not in f["topics"]:
            raise KeyError("找不到 _usb_cam_0_image_raw topic")
        
        images = f["topics/_usb_cam_0_image_raw/data"][:]  # (T, H, W, 3)
        
        # 读取有效性标记（如果有）
        valid = None
        if not ignore_valid and "valid" in f and "_joint_states" in f["valid"]:
            valid = f["valid/_joint_states"][:]  # (T,)
        
        # 确保所有数据长度一致
        min_length = min(len(positions), len(images))
        if valid is not None and not ignore_valid:
            # 只使用有效的步骤
            valid_indices = np.where(valid[:min_length])[0]
        else:
            valid_indices = np.arange(min_length)
        
        if len(valid_indices) == 0:
            raise ValueError("没有有效的数据步骤")
        
        # 提取有效数据
        positions = positions[valid_indices]
        velocities = velocities[valid_indices]
        images = images[valid_indices]
        
        # 计算动作（使用速度或位置差分）
        # 对于 LIBERO，我们通常只需要左臂或右臂，或者组合
        # 这里我们提取左臂的关节（前7个）和右臂的关节（后7个）
        # 根据你的需求调整
        
        # 选项1: 只使用左臂（前7个关节）
        left_positions = positions[:, :7]  # (T, 7)
        left_velocities = velocities[:, :7]  # (T, 7)
        
        # 选项2: 使用左臂 + 右臂（14个关节，但 LIBERO 通常只需要7-8维）
        # 如果需要，可以组合或选择
        
        # 计算动作（使用速度）
        actions = left_velocities  # (T, 7)
        
        # 组合状态（关节位置 + 夹爪，LIBERO 需要8维）
        # 假设最后一个关节是夹爪，或者添加一个零夹爪维度
        states = np.concatenate([left_positions, np.zeros((len(left_positions), 1))], axis=1)  # (T, 8)
        
        # 转换为步骤列表
        steps = []
        for i in range(len(positions)):
            # 调整图像大小
            image = resize_image(images[i], (256, 256))
            
            # 如果没有手腕相机，使用主相机
            wrist_image = image.copy()
            
            steps.append({
                "image": image,
                "wrist_image": wrist_image,
                "state": states[i].astype(np.float32),
                "action": actions[i].astype(np.float32),
                "task": task_description,
            })
        
        return steps


def main(
    data_dir: str,
    *,
    push_to_hub: bool = False,
    task_description: str = "Do something",
    use_left_arm: bool = True,
    use_right_arm: bool = False,
    ignore_valid: bool = False,
):
    """
    主函数：将 ROS2 bag HDF5 格式数据转换为 LeRobot 格式
    
    Args:
        data_dir: HDF5 文件所在的目录
        push_to_hub: 是否推送到 Hugging Face Hub
        task_description: 任务描述（如果没有在文件中找到）
        use_left_arm: 是否使用左臂（前7个关节）
        use_right_arm: 是否使用右臂（后7个关节）
        ignore_valid: 是否忽略有效性标记，使用所有数据
    """
    data_dir = Path(data_dir)
    
    # 清理输出目录
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)
    
    # 创建 LeRobot 数据集
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
                "shape": (7,),  # 7 维动作
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
        try:
            steps = load_ros2bag_hdf5(hdf5_path, task_description, ignore_valid)
            
            # 写入 LeRobot 数据集
            for step in steps:
                # 确保图像是 uint8 格式
                image = step["image"]
                wrist_image = step["wrist_image"]
                
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
            print(f"✅ 成功转换 {hdf5_path.name} ({len(steps)} 步)")
            
        except Exception as e:
            print(f"❌ 处理 {hdf5_path} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"✅ 转换完成！数据集保存在: {output_path}")
    
    # 可选：推送到 Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "ros2bag", "hdf5"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print("✅ 已推送到 Hugging Face Hub")


if __name__ == "__main__":
    tyro.cli(main)

