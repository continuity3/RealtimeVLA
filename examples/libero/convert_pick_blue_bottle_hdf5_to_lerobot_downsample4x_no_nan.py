"""
转换 pick_blue_bottle 数据集的 HDF5 格式数据到 LeRobot 格式（下采样4倍版本，去掉NaN值）。

这个脚本专门用于处理 pick_blue_bottle 数据集的 HDF5 文件，并将数据下采样4倍。
只使用右臂数据（左臂未使用），并包含右夹爪信息。
**关键区别：遇到NaN值直接删除，而不是填充0**

HDF5 文件结构:
- time: (T,) 时间戳
- topics/_joint_states/:
    - position: (T, 14) 关节位置（前7维=左臂，后7维=右臂）
    - velocity: (T, 14) 关节速度（前7维=左臂，后7维=右臂）
- topics/_control_gripperValueR/:
    - data: (T,) 右夹爪值（0=全开，1=全闭）
- topics/_camera_camera_color_image_raw/:
    - data: (T, 921600) 图像数据（扁平化）
    - data_length: (T,) 每个图像的实际长度

输出数据:
- 状态: [7个右臂关节位置, 1个右夹爪值] = 8维
- 动作: [7个右臂关节速度, 1个右夹爪速度] = 8维（包含gripper）
  - 关节动作 = velocity (rad/s)
  - gripper动作 = gripper速度 (变化率)
  - 注意：训练时需要使用 q_next = q_curr + velocity * dt，其中 dt = 1/fps
- **NaN值处理：直接删除包含NaN的时间步**

Usage:
uv run examples/libero/convert_pick_blue_bottle_hdf5_to_lerobot_downsample4x_no_nan.py --data_dir /path/to/pick_blue_bottle_extracted
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

REPO_NAME = "your_hf_username/pick_blue_bottle_libero_downsample4x_no_nan"  # 输出数据集名称（去掉NaN版本）


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


def compute_actions_from_states(positions: np.ndarray, velocities: np.ndarray | None) -> np.ndarray:
    """
    从关节位置和速度计算动作。
    
    对于 LIBERO，通常使用位置增量（delta）或速度作为动作。
    这里我们使用速度作为动作（如果可用且有效），否则使用位置差分。
    
    Args:
        positions: 关节位置数组 (T, 7)
        velocities: 关节速度数组 (T, 7) 或 None
    
    Returns:
        动作数组 (T, 7)
    """
    # 检查速度是否有效（非 None、非空、非全 NaN、非全零）
    if velocities is not None and len(velocities) > 0:
        if not np.isnan(velocities).any() and np.any(np.abs(velocities) > 1e-6):
            # 速度有效，使用速度作为动作
            return velocities
    
    # 速度无效或不可用，使用位置差分作为动作
    # np.diff 计算相邻帧之间的位置差
    # prepend 确保第一帧的动作为 0（或使用第一帧的位置作为初始值）
    actions = np.diff(positions, axis=0, prepend=positions[0:1])
    return actions


def load_pick_blue_bottle_hdf5(hdf5_path: Path, task_description: str = "Pick blue bottle and place it in blue plate", ignore_valid: bool = False, downsample_factor: int = 4) -> list[dict]:
    """
    从 pick_blue_bottle HDF5 文件中加载数据（下采样版本，去掉NaN值）。
    
    Args:
        hdf5_path: HDF5 文件路径
        task_description: 任务描述
        ignore_valid: 是否忽略有效性标记
        downsample_factor: 下采样因子（每N帧取1帧）
    
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
        
        # 读取主相机图像
        if "_camera_camera_color_image_raw" not in f["topics"]:
            raise KeyError("找不到 _camera_camera_color_image_raw topic")
        
        image_topic = f["topics/_camera_camera_color_image_raw"]
        image_data = image_topic["data"][:]  # 可能是 (T, H, W, 3) 或 (T, 921600)
        
        # 检查数据格式：新格式是已解码的图像数组，旧格式是扁平化的字节数据
        if len(image_data.shape) == 4:
            # 新格式：已经是解码后的图像 (T, H, W, 3)
            image_lengths = None
            print("  ✅ 检测到新格式：主相机图像已解码")
        else:
            # 旧格式：扁平化的字节数据，需要解码
            image_lengths = image_topic["data_length"][:]  # (T,)
            print("  ✅ 检测到旧格式：主相机图像需要解码")
        
        # 读取手腕相机图像（如果存在）
        wrist_image_data = None
        wrist_image_lengths = None
        has_wrist_camera = "image_wrist" in f["topics"]
        if has_wrist_camera:
            wrist_topic = f["topics/image_wrist"]
            wrist_image_data = wrist_topic["data"][:]  # 可能是 (T, H, W, 3) 或 (T, ...)
            if len(wrist_image_data.shape) == 4:
                # 新格式：已经是解码后的图像
                wrist_image_lengths = None
                print("  ✅ 检测到手腕相机数据（新格式）")
            else:
                # 旧格式：可能需要解码
                if "data_length" in wrist_topic:
                    wrist_image_lengths = wrist_topic["data_length"][:]
                print("  ✅ 检测到手腕相机数据（旧格式）")
        else:
            print("  ⚠️  未找到 image_wrist topic，将使用主相机图像作为手腕图像")
        
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
            
            # 如果存在手腕相机，也检查其有效性
            valid_wrist = None
            if has_wrist_camera and "image_wrist" in f.get("valid", {}):
                valid_wrist = f["valid/image_wrist"][:]  # (T,)
            
            # 组合有效性：joint_states 必须有效，主相机和手腕相机（如果存在）也应该有效
            valid_list = [v for v in [valid_joint, valid_image, valid_wrist] if v is not None]
            if valid_list:
                valid = valid_list[0]
                for v in valid_list[1:]:
                    valid = valid & v
            elif valid_joint is not None:
                valid = valid_joint
            elif valid_image is not None:
                valid = valid_image
        
        # 确保所有数据长度一致
        min_length = len(positions)
        min_length = min(min_length, len(image_data))
        if wrist_image_data is not None:
            min_length = min(min_length, len(wrist_image_data))
        
        if valid is not None and not ignore_valid:
            # 只使用有效的步骤
            valid_indices = np.where(valid[:min_length])[0]
        else:
            valid_indices = np.arange(min_length)
        
        if len(valid_indices) == 0:
            raise ValueError("没有有效的数据步骤")
        
        # 读取右夹爪数据（在过滤和下采样之前）
        right_gripper_values = None
        if "_control_gripperValueR" in f["topics"]:
            gripper_topic = f["topics/_control_gripperValueR"]
            if "data" in gripper_topic:
                gripper_data = gripper_topic["data"][:]  # (T,)
                # 先过滤有效索引
                gripper_data = gripper_data[valid_indices]
                right_gripper_values = gripper_data
                print(f"  ✅ 读取右夹爪数据: {len(right_gripper_values)} 个值（过滤后）")
            else:
                print("  ⚠️  夹爪话题中没有 'data' 键")
        else:
            print("  ⚠️  未找到 _control_gripperValueR 话题")
        
        # 提取有效数据
        positions = positions[valid_indices]
        velocities = velocities[valid_indices]
        image_data = image_data[valid_indices]
        if image_lengths is not None:
            image_lengths = image_lengths[valid_indices]
        
        # 提取手腕相机数据（如果存在）
        if wrist_image_data is not None:
            wrist_image_data = wrist_image_data[valid_indices]
            if wrist_image_lengths is not None:
                wrist_image_lengths = wrist_image_lengths[valid_indices]
        
        # 下采样：每 downsample_factor 帧取1帧
        downsampled_indices = np.arange(0, len(positions), downsample_factor)
        positions = positions[downsampled_indices]
        velocities = velocities[downsampled_indices]
        image_data = image_data[downsampled_indices]
        if image_lengths is not None:
            image_lengths = image_lengths[downsampled_indices]
        
        # 下采样手腕相机数据（如果存在）
        if wrist_image_data is not None:
            wrist_image_data = wrist_image_data[downsampled_indices]
            if wrist_image_lengths is not None:
                wrist_image_lengths = wrist_image_lengths[downsampled_indices]
        
        # 下采样夹爪数据（与关节数据同步）
        if right_gripper_values is not None:
            right_gripper_values = right_gripper_values[downsampled_indices]
            print(f"  ✅ 下采样后右夹爪数据: {len(right_gripper_values)} 个值，范围 [{np.min(right_gripper_values[~np.isnan(right_gripper_values)]):.4f}, {np.max(right_gripper_values[~np.isnan(right_gripper_values)]):.4f}]")
        else:
            # 如果夹爪数据不可用，创建零数组
            right_gripper_values = np.zeros(len(positions))
            print("  ⚠️  使用零夹爪值（未找到夹爪数据）")
        
        # 提取右臂关节（列 7-13，对应 Joint1_R 到 Joint7_R）
        # 注意：joint_states 有14维：前7维是左臂（Joint1_L 到 Joint7_L），后7维是右臂（Joint1_R 到 Joint7_R）
        right_positions = positions[:, 7:14]  # (T, 7) - 右臂关节位置
        right_velocities = velocities[:, 7:14]  # (T, 7) - 右臂关节速度
        
        # 🔑 关键修改：找到所有有效的时间步（joint位置和gripper值都不是NaN）
        # 对于joint位置，检查所有7个关节是否都是有效值
        joint_valid = ~np.isnan(right_positions).any(axis=1)  # (T,)
        gripper_valid = ~np.isnan(right_gripper_values)  # (T,)
        
        # 两者都有效的时间步
        no_nan_mask = joint_valid & gripper_valid
        
        no_nan_indices = np.where(no_nan_mask)[0]
        
        print(f"  📊 NaN值过滤统计:")
        print(f"     原始数据: {len(right_positions)} 个时间步")
        print(f"     Joint有效: {np.sum(joint_valid)}/{len(joint_valid)} ({np.sum(joint_valid)/len(joint_valid)*100:.2f}%)")
        print(f"     Gripper有效: {np.sum(gripper_valid)}/{len(gripper_valid)} ({np.sum(gripper_valid)/len(gripper_valid)*100:.2f}%)")
        print(f"     两者都有效（去掉NaN后）: {len(no_nan_indices)}/{len(right_positions)} ({len(no_nan_indices)/len(right_positions)*100:.2f}%)")
        
        if len(no_nan_indices) == 0:
            raise ValueError("错误：去掉NaN后没有有效的数据步骤！")
        
        # 🔑 关键修改：只保留没有NaN的时间步
        right_positions = right_positions[no_nan_indices]  # (N, 7)
        right_velocities = right_velocities[no_nan_indices]  # (N, 7)
        right_gripper_values = right_gripper_values[no_nan_indices]  # (N,)
        image_data = image_data[no_nan_indices]
        if image_lengths is not None:
            image_lengths = image_lengths[no_nan_indices]
        
        # 同步过滤手腕相机数据（如果存在）
        if wrist_image_data is not None:
            wrist_image_data = wrist_image_data[no_nan_indices]
            if wrist_image_lengths is not None:
                wrist_image_lengths = wrist_image_lengths[no_nan_indices]
        
        print(f"  ✅ 去掉NaN后保留: {len(no_nan_indices)} 个时间步")
        
        # --- 🛡️ 健壮的动作计算 ---
        # 检查速度是否有效（不是 NaN 且不全为 0）
        velocity_is_valid = not np.isnan(right_velocities).any() and np.any(np.abs(right_velocities) > 1e-6)
        
        # ⚠️ CRITICAL: Actions are velocities (rad/s), NOT delta positions
        # The training script will multiply by dt when using: q_next = q_curr + velocity * dt
        if velocity_is_valid:
            print("  ✅ 使用原始关节速度作为动作 (rad/s)")
            actions = right_velocities
        else:
            print("  ⚠️  原始速度无效或为 NaN，使用位置差分计算速度")
            # 使用位置差分计算速度（需要除以 dt）
            # 数据集下采样4倍：30fps -> 7.5fps，所以 dt = 1/7.5
            dt = 1.0 / 7.5
            position_deltas = compute_actions_from_states(right_positions, right_velocities)
            # 将位置差分转换为速度: velocity = delta_position / dt
            actions = position_deltas / dt
        
        # 确保动作没有 NaN（理论上不应该有，因为已经过滤了）
        if np.isnan(actions).any():
            print("  ⚠️  警告：检测到动作包含 NaN（不应该发生，因为已经过滤了NaN）")
            # 如果还有NaN，删除这些时间步
            action_valid = ~np.isnan(actions).any(axis=1)
            actions = actions[action_valid]
            right_positions = right_positions[action_valid]
            right_gripper_values = right_gripper_values[action_valid]
            image_data = image_data[action_valid]
            if image_lengths is not None:
                image_lengths = image_lengths[action_valid]
            
            # 同步过滤手腕相机数据（如果存在）
            if wrist_image_data is not None:
                wrist_image_data = wrist_image_data[action_valid]
                if wrist_image_lengths is not None:
                    wrist_image_lengths = wrist_image_lengths[action_valid]
            
            print(f"  ✅ 进一步过滤后保留: {len(actions)} 个时间步")
        
        # 计算gripper动作（gripper的速度，即变化率）
        # 对于gripper，我们使用差分来计算速度（需要除以 dt）
        dt = 1.0 / 7.5  # 下采样4倍: 30fps -> 7.5fps
        gripper_position_deltas = np.diff(right_gripper_values, axis=0, prepend=right_gripper_values[0:1])
        gripper_actions = gripper_position_deltas / dt  # 转换为速度
        
        # 确保gripper动作没有NaN（理论上不应该有）
        if np.isnan(gripper_actions).any():
            print("  ⚠️  警告：检测到gripper动作包含 NaN（不应该发生）")
            gripper_actions = np.nan_to_num(gripper_actions, nan=0.0)
        
        # 组合动作（7个关节速度 + 1个gripper速度） = 8维
        actions = np.concatenate([actions, gripper_actions[:, None]], axis=1)  # (N, 8)
        print(f"  ✅ 动作维度: {actions.shape} (7个关节速度 rad/s + 1个gripper速度)")
        
        # 组合状态（右臂关节位置 + 右夹爪，LIBERO 需要8维）
        # 状态: [7个右臂关节位置, 1个右夹爪值]
        states = np.concatenate([right_positions, right_gripper_values[:, None]], axis=1)  # (N, 8)
        print(f"  ✅ 状态维度: {states.shape} (7个关节位置 + 1个gripper值)")
        
        # 解码主相机图像
        print(f"  处理 {len(image_data)} 张主相机图像（去掉NaN后）...")
        if len(image_data.shape) == 4:
            # 新格式：已经是解码后的图像数组 (T, H, W, 3)
            images = image_data.astype(np.uint8)
            print(f"  ✅ 主相机图像已解码，形状: {images.shape}")
        else:
            # 旧格式：需要解码扁平化的字节数据
            images = []
            for i in tqdm(range(len(image_data)), desc="  解码主相机图像", leave=False):
                try:
                    img = decode_image(image_data[i], image_lengths[i])
                    images.append(img)
                except Exception as e:
                    print(f"  ⚠️  解码主相机图像 {i} 失败: {e}，使用零图像")
                    # 使用零图像作为占位符
                    images.append(np.zeros((480, 640, 3), dtype=np.uint8))
            images = np.array(images)
        
        # 解码手腕相机图像（如果存在）
        wrist_images = None
        if wrist_image_data is not None:
            print(f"  处理 {len(wrist_image_data)} 张手腕相机图像（去掉NaN后）...")
            if len(wrist_image_data.shape) == 4:
                # 新格式：已经是解码后的图像数组 (T, H, W, 3)
                wrist_images = wrist_image_data.astype(np.uint8)
                print(f"  ✅ 手腕相机图像已解码，形状: {wrist_images.shape}")
            else:
                # 旧格式：需要解码扁平化的字节数据
                wrist_images = []
                for i in tqdm(range(len(wrist_image_data)), desc="  解码手腕相机图像", leave=False):
                    try:
                        if wrist_image_lengths is not None:
                            img = decode_image(wrist_image_data[i], wrist_image_lengths[i])
                        else:
                            # 如果没有 data_length，尝试直接解码
                            img = decode_image(wrist_image_data[i], len(wrist_image_data[i]))
                        wrist_images.append(img)
                    except Exception as e:
                        print(f"  ⚠️  解码手腕相机图像 {i} 失败: {e}，使用零图像")
                        # 使用零图像作为占位符
                        wrist_images.append(np.zeros((480, 640, 3), dtype=np.uint8))
                wrist_images = np.array(wrist_images)
        
        # 转换为步骤列表
        steps = []
        for i in range(len(right_positions)):
            # 调整主相机图像大小
            image = resize_image(images[i], (256, 256))
            
            # --- 🛡️ 确保图像是 uint8 且不需要额外缩放 ---
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            else:
                image = image.astype(np.uint8)
            
            # 处理手腕相机图像
            if wrist_images is not None:
                # 使用真实的手腕相机图像
                wrist_image = resize_image(wrist_images[i], (256, 256))
                if wrist_image.dtype != np.uint8:
                    if wrist_image.max() <= 1.0:
                        wrist_image = (wrist_image * 255).astype(np.uint8)
                    else:
                        wrist_image = wrist_image.astype(np.uint8)
                else:
                    wrist_image = wrist_image.astype(np.uint8)
            else:
                # 如果没有手腕相机，使用主相机（向后兼容）
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
    downsample_factor: int = 4,
    fps: int = 7.5,  # 原始30fps下采样4倍后为7.5fps
):
    """
    主函数：将 pick_blue_bottle HDF5 格式数据转换为 LeRobot 格式（下采样4倍版本，去掉NaN值）
    
    Args:
        data_dir: HDF5 文件所在的目录
        push_to_hub: 是否推送到 Hugging Face Hub
        task_description: 任务描述
        ignore_valid: 是否忽略有效性标记，使用所有数据
        downsample_factor: 下采样因子（每N帧取1帧）
        fps: 输出数据集的帧率（原始30fps下采样4倍后为7.5fps）
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
        fps=fps,  # 下采样后的帧率
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
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),  # 8 维动作（7个关节速度 + 1个gripper速度）
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
    
    # 查找所有 HDF5 文件
    hdf5_files = sorted(list(data_dir.glob("*.h5")) + list(data_dir.glob("*.hdf5")))
    if not hdf5_files:
        raise FileNotFoundError(f"在目录 '{data_dir}' 中找不到任何 .h5 或 .hdf5 文件")
    
    print(f"找到 {len(hdf5_files)} 个 HDF5 文件")
    print(f"下采样因子: {downsample_factor}x")
    print(f"输出帧率: {fps} fps")
    print(f"⚠️  重要：遇到NaN值将直接删除该时间步（不填充0）")
    print()
    
    # 遍历所有 HDF5 文件
    total_steps = 0
    total_original_steps = 0
    for hdf5_path in tqdm(hdf5_files, desc="处理 HDF5 文件"):
        try:
            # 先统计原始数据量
            with h5py.File(hdf5_path, 'r') as f:
                if "_joint_states" in f["topics"]:
                    original_length = len(f["topics/_joint_states"]["position"])
                    total_original_steps += original_length
            
            steps = load_pick_blue_bottle_hdf5(hdf5_path, task_description, ignore_valid, downsample_factor)
            
            # 写入 LeRobot 数据集
            # 直接使用 step 中的数据，因为 load 函数里已经处理好了
            for step in steps:
                dataset.add_frame({
                    "image": step["image"],           # 已经是处理好的 uint8
                    "wrist_image": step["wrist_image"], # 已经是处理好的 uint8
                    "state": step["state"].astype(np.float32),
                    "actions": step["action"].astype(np.float32),
                    "task": step["task"],
                })
            
            dataset.save_episode()
            total_steps += len(steps)
            print(f"✅ 成功转换 {hdf5_path.name} ({len(steps)} 步，去掉NaN后)")
            print()
            
        except Exception as e:
            print(f"❌ 处理 {hdf5_path} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✅ 转换完成！")
    print(f"   原始数据总步数: {total_original_steps}")
    print(f"   去掉NaN后总步数: {total_steps}")
    print(f"   保留率: {total_steps/total_original_steps*100:.2f}%")
    print(f"数据集保存在: {output_path}")
    
    # 可选：推送到 Hugging Face Hub
    if push_to_hub:
        print("\n推送到 Hugging Face Hub...")
        dataset.push_to_hub(
            tags=["libero", "panda", "downsampled", "no_nan"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print(f"✅ 已推送到 Hub: {REPO_NAME}")


if __name__ == "__main__":
    tyro.cli(main)


