"""
转换 pick_blue_bottle 数据集的 HDF5 格式数据到 LeRobot 格式的脚本。

这个脚本专门用于处理 pick_blue_bottle 数据集的 HDF5 文件。

HDF5 文件结构:
- time: (T,) 时间戳
- topics/_joint_states/:
    - position: (T, 14) 关节位置
    - velocity: (T, 14) 关节速度
- topics/_camera_camera_color_image_raw/:
    - data: (T, 921600) 图像数据（扁平化）
    - data_length: (T,) 每个图像的实际长度

Usage:
uv run examples/libero/convert_pick_blue_bottle_hdf5_to_lerobot.py --data_dir /path/to/pick_blue_bottle_extracted
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

REPO_NAME = "your_hf_username/pick_blue_bottle_libero"  # 输出数据集名称


def decode_image(img_data: np.ndarray, img_length: int) -> np.ndarray:
    """
    解码图像数据。
    
    图像数据可能是：
    1. JPEG 压缩格式（需要解码）
    2. 原始 RGB 图像数据（需要重塑）
    
    Args:
        img_data: 图像数据数组（扁平化）
        img_length: 图像数据的实际长度
    
    Returns:
        解码后的图像 (H, W, 3) uint8
    """
    img_bytes = bytes(img_data[:img_length])
    
    # 优先尝试作为 JPEG 解码（最常见）
    try:
        import cv2
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            # OpenCV 返回 BGR，转换为 RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
    except ImportError:
        # 如果没有 OpenCV，尝试用 PIL
        try:
            from io import BytesIO
            img = Image.open(BytesIO(img_bytes))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return np.array(img)
        except:
            pass
    except:
        # OpenCV 解码失败，尝试其他方法
        pass
    
    # 如果不是 JPEG，尝试作为原始图像数据
    # 尝试常见的尺寸
    possible_sizes = [
        (720, 1280, 1),  # 单通道
        (480, 640, 3),   # RGB
        (480, 854, 3),   # RGB
        (360, 640, 3),   # RGB
    ]
    
    for h, w, c in possible_sizes:
        if h * w * c == img_length:
            img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(h, w, c)
            if c == 1:
                # 单通道转 RGB
                img = np.repeat(img, 3, axis=2)
            return img
    
    # 如果都不匹配，尝试直接重塑为 640x480x3（最常见）
    if img_length >= 640 * 480 * 3:
        img = np.frombuffer(img_bytes[:640*480*3], dtype=np.uint8).reshape(480, 640, 3)
        return img
    
    raise ValueError(f"无法解码图像数据，长度: {img_length}")


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


def load_pick_blue_bottle_hdf5(hdf5_path: Path, task_description: str = "Pick blue bottle and place it in blue plate", ignore_valid: bool = False) -> list[dict]:
    """
    从 pick_blue_bottle HDF5 文件中加载数据。
    
    Args:
        hdf5_path: HDF5 文件路径
        task_description: 任务描述
        ignore_valid: 是否忽略有效性标记
    
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
        
        # 读取图像
        if "_camera_camera_color_image_raw" not in f["topics"]:
            raise KeyError("找不到 _camera_camera_color_image_raw topic")
        
        image_topic = f["topics/_camera_camera_color_image_raw"]
        image_data = image_topic["data"][:]  # (T, 921600)
        image_lengths = image_topic["data_length"][:]  # (T,)
        
        # 读取有效性标记（如果有）
        valid = None
        if not ignore_valid and "valid" in f:
            # 优先使用 joint_states 的有效性，如果图像也有效则更好
            if "_joint_states" in f["valid"]:
                valid_joint = f["valid/_joint_states"][:]  # (T,)
            else:
                valid_joint = None
            
            if "_camera_camera_color_image_raw" in f["valid"]:
                valid_image = f["valid/_camera_camera_color_image_raw"][:]  # (T,)
            else:
                valid_image = None
            
            # 如果两者都有效则使用，否则优先保证 joint_states 有效
            if valid_joint is not None and valid_image is not None:
                valid = valid_joint & valid_image
            elif valid_joint is not None:
                valid = valid_joint
            elif valid_image is not None:
                valid = valid_image
        
        # 确保所有数据长度一致
        min_length = min(len(positions), len(image_data))
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
        image_data = image_data[valid_indices]
        image_lengths = image_lengths[valid_indices]
        
        # 解码图像
        print(f"  解码 {len(valid_indices)} 张图像...")
        images = []
        for i in tqdm(range(len(valid_indices)), desc="  解码图像", leave=False):
            try:
                img = decode_image(image_data[i], image_lengths[i])
                images.append(img)
            except Exception as e:
                print(f"  ⚠️  解码图像 {i} 失败: {e}，使用零图像")
                # 使用零图像作为占位符
                images.append(np.zeros((480, 640, 3), dtype=np.uint8))
        
        images = np.array(images)
        
        # 提取左臂关节（前7个关节）用于 LIBERO
        left_positions = positions[:, :7]  # (T, 7)
        left_velocities = velocities[:, :7]  # (T, 7)
        
        # 计算动作（使用速度）
        actions = left_velocities  # (T, 7)
        
        # 组合状态（关节位置 + 夹爪，LIBERO 需要8维）
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
    task_description: str = "Pick blue bottle and place it in blue plate",
    ignore_valid: bool = False,
):
    """
    主函数：将 pick_blue_bottle HDF5 格式数据转换为 LeRobot 格式
    
    Args:
        data_dir: HDF5 文件所在的目录
        push_to_hub: 是否推送到 Hugging Face Hub
        task_description: 任务描述
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
        fps=30,  # 根据用户要求，fps 是 30
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
    hdf5_files = sorted(list(data_dir.glob("*.hdf5")) + list(data_dir.glob("*.h5")))
    if not hdf5_files:
        raise ValueError(f"在 {data_dir} 中找不到 HDF5 文件")
    
    print(f"找到 {len(hdf5_files)} 个 HDF5 文件")
    
    # 遍历所有 HDF5 文件
    total_steps = 0
    for hdf5_path in tqdm(hdf5_files, desc="处理 HDF5 文件"):
        try:
            steps = load_pick_blue_bottle_hdf5(hdf5_path, task_description, ignore_valid)
            
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
            total_steps += len(steps)
            print(f"✅ 成功转换 {hdf5_path.name} ({len(steps)} 步)")
            
        except Exception as e:
            print(f"❌ 处理 {hdf5_path} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"✅ 转换完成！")
    print(f"   - 总文件数: {len(hdf5_files)}")
    print(f"   - 总步数: {total_steps}")
    print(f"   - 数据集保存在: {output_path}")
    
    # 可选：推送到 Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "pick_blue_bottle", "manipulation"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print("✅ 已推送到 Hugging Face Hub")


if __name__ == "__main__":
    tyro.cli(main)

