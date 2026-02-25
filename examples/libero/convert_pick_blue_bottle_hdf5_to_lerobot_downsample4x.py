"""
转换 pick_blue_bottle 数据集的 HDF5 格式数据到 LeRobot 格式的脚本（下采样4倍版本，仅使用右臂数据）。

这个脚本专门用于处理 pick_blue_bottle 数据集的 HDF5 文件，并将数据下采样4倍。
只使用右臂数据（左臂未使用），并包含右夹爪信息。

HDF5 文件结构:
- time: (T,) 时间戳
- topics/_joint_states/:
    - position: (T, 14) 关节位置（前7维=左臂，后7维=右臂）
    - velocity: (T, 14) 关节速度（前7维=左臂，后7维=右臂）
- topics/_info_eef_right/:
    - position: (T, 3) 右臂末端执行器位置
    - orientation: (T, 4) 右臂末端执行器方向（四元数）
- topics/_gripper_feedback_R/:
    - data: (T, 5) 右夹爪反馈数据（第一列是夹爪状态值）
- topics/_camera_camera_color_image_raw/:
    - data: (T, 921600) 或 (T, H, W, 3) 主相机图像数据（旧格式扁平化，新格式已解码）
    - data_length: (T,) 每个图像的实际长度（仅旧格式）
- topics/image_wrist/ (可选):
    - data: (T, H, W, 3) 或 (T, ...) 手腕相机图像数据（新格式已解码，旧格式可能需要解码）
    - data_length: (T,) 每个图像的实际长度（仅旧格式，如果存在）

输出数据:
- 状态: [3维EEF位置, 3维EEF方向(轴角), 1维夹爪值, 1维夹爪值相反数] = 8维
- 动作: [7个右臂关节速度, 1个右夹爪速度] = 8维（包含gripper）
  - 关节动作 = velocity (rad/s)
  - gripper动作 = gripper速度 (变化率)
  - 注意：训练时需要使用 q_next = q_curr + velocity * dt，其中 dt = 1/fps

Usage:
uv run examples/libero/convert_pick_blue_bottle_hdf5_to_lerobot_downsample4x.py --data_dir /path/to/pick_blue_bottle_extracted
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
from scipy.spatial.transform import Rotation as R

REPO_NAME = "your_hf_username/pick_blue_bottle_libero_downsample4x"  # 输出数据集名称（下采样4倍版本）


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


def handle_nan_values(data: np.ndarray) -> np.ndarray:
    """
    处理 NaN 值：
    - 第一个数据出现 NaN → 使用第二个数据
    - 最后一个数据出现 NaN → 使用倒数第二个数据
    - 中间出现 NaN → 使用上一时刻的值
    
    Args:
        data: 数据数组，可以是 1D 或 2D
    
    Returns:
        处理后的数据数组
    """
    data_clean = data.copy()
    
    if len(data_clean.shape) == 1:
        # 1D 数组
        for i in range(len(data_clean)):
            if np.isnan(data_clean[i]):
                if i == 0:
                    # 第一个：使用下一个非 NaN 值
                    for j in range(1, len(data_clean)):
                        if not np.isnan(data_clean[j]):
                            data_clean[i] = data_clean[j]
                            break
                elif i == len(data_clean) - 1:
                    # 最后一个：使用上一个非 NaN 值
                    for j in range(len(data_clean) - 2, -1, -1):
                        if not np.isnan(data_clean[j]):
                            data_clean[i] = data_clean[j]
                            break
                else:
                    # 中间：使用前一个非 NaN 值
                    for j in range(i - 1, -1, -1):
                        if not np.isnan(data_clean[j]):
                            data_clean[i] = data_clean[j]
                            break
    else:
        # 2D 数组，逐行处理
        for i in range(len(data_clean)):
            if np.isnan(data_clean[i]).any():
                if i == 0:
                    # 第一个：使用下一个非 NaN 值
                    for j in range(1, len(data_clean)):
                        if not np.isnan(data_clean[j]).any():
                            data_clean[i] = data_clean[j]
                            break
                elif i == len(data_clean) - 1:
                    # 最后一个：使用上一个非 NaN 值
                    for j in range(len(data_clean) - 2, -1, -1):
                        if not np.isnan(data_clean[j]).any():
                            data_clean[i] = data_clean[j]
                            break
                else:
                    # 中间：使用前一个非 NaN 值
                    for j in range(i - 1, -1, -1):
                        if not np.isnan(data_clean[j]).any():
                            data_clean[i] = data_clean[j]
                            break
    
    return data_clean


def axisangle_to_quat(axis_angle):
    """
    axis_angle: (3,)
    return quat: (4,) in xyzw
    """
    return R.from_rotvec(axis_angle).as_quat()


def quat_to_axisangle(quat):
    """
    quat: (4,) xyzw
    return axis-angle: (3,)
    """
    return R.from_quat(quat).as_rotvec()


def relative_axisangle(aa_t, aa_t1):
    """
    Compute relative rotation from t -> t+1 in axis-angle
    """
    q_t = axisangle_to_quat(aa_t)
    q_t1 = axisangle_to_quat(aa_t1)

    # relative rotation: q_rel = q_t1 * inverse(q_t)
    q_rel = R.from_quat(q_t1) * R.from_quat(q_t).inv()
    return q_rel.as_rotvec()


def compute_geom_action(
    ee_pos_t,
    ee_ori_t,
    ee_pos_t1,
    ee_ori_t1,
):
    """
    All inputs are np.ndarray with shape (3,)
    ee_ori_* are axis-angle
    """
    delta_pos = ee_pos_t1 - ee_pos_t
    delta_ori = relative_axisangle(ee_ori_t, ee_ori_t1)

    action_6d = np.concatenate([delta_pos, delta_ori])
    return action_6d


def process_gripper_feedback(gripper_values: np.ndarray) -> np.ndarray:
    """
    处理夹爪反馈值：
    - 小于 0.4 → 0（打开）
    - 大于 0.6 → 1（闭合）
    - 0.4-0.6 之间 → 保持原值或插值
    
    Args:
        gripper_values: 夹爪反馈值数组 (T,)
    
    Returns:
        处理后的夹爪状态数组 (T,)
    """
    gripper_state = gripper_values.copy()
    
    # 处理 NaN
    gripper_state = handle_nan_values(gripper_state)
    
    # 判断夹爪状态
    gripper_binary = np.zeros_like(gripper_state)
    gripper_binary[gripper_state < 0.4] = 0.0
    gripper_binary[gripper_state > 0.6] = 1.0
    # 0.4-0.6 之间：线性插值
    mask_middle = (gripper_state >= 0.4) & (gripper_state <= 0.6)
    gripper_binary[mask_middle] = (gripper_state[mask_middle] - 0.4) / 0.2  # 归一化到 [0, 1]
    
    return gripper_binary


def load_pick_blue_bottle_hdf5(hdf5_path: Path, task_description: str = "Pick blue bottle and place it in blue plate", ignore_valid: bool = False, downsample_factor: int = 4) -> list[dict]:
    """
    从 pick_blue_bottle HDF5 文件中加载数据（下采样版本）。
    
    使用几何 action 计算方式：
    - Action: 6维 EEF action (xyz位移 + 轴角旋转) + 1维夹爪状态 = 7维
    - State: 3维EEF位置 + 3维EEF方向(轴角) + 1维夹爪值 + 1维夹爪值相反数 = 8维
    
    Args:
        hdf5_path: HDF5 文件路径
        task_description: 任务描述
        ignore_valid: 是否忽略有效性标记
        downsample_factor: 下采样因子（每N帧取1帧）
    
    Returns:
        步骤列表，每个步骤包含 image, wrist_image, state, action, task
    """
    with h5py.File(hdf5_path, "r") as f:
        # 读取 _info_eef_right 数据（用于计算几何 action）
        if "_info_eef_right" not in f["topics"]:
            raise KeyError("找不到 _info_eef_right topic")
        
        eef_right_group = f["topics/_info_eef_right"]
        eef_positions = eef_right_group["position"][:]  # (T, 3)
        eef_orientations = eef_right_group["orientation"][:]  # (T, 4) - 四元数 [x, y, z, w]
        
        print(f"  ✅ 读取 _info_eef_right: position {eef_positions.shape}, orientation {eef_orientations.shape}")
        
        # 处理 NaN
        eef_positions = handle_nan_values(eef_positions)
        eef_orientations = handle_nan_values(eef_orientations)
        
        # 转换为轴角
        axis_angles = np.zeros((len(eef_orientations), 3))
        for i in range(len(eef_orientations)):
            quat = eef_orientations[i]  # [x, y, z, w]
            axis_angles[i] = quat_to_axisangle(quat)
        
        print(f"  ✅ 转换为轴角: {axis_angles.shape}")
        
        # 读取关节状态（用于 state）
        if "_joint_states" not in f["topics"]:
            raise KeyError("找不到 _joint_states topic")
        
        joint_states = f["topics/_joint_states"]
        joint_positions = joint_states["position"][:]  # (T, 14)
        
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
        min_length = min(len(eef_positions), len(joint_positions), len(image_data))
        if wrist_image_data is not None:
            min_length = min(min_length, len(wrist_image_data))
        
        if valid is not None and not ignore_valid:
            # 只使用有效的步骤
            valid_indices = np.where(valid[:min_length])[0]
        else:
            valid_indices = np.arange(min_length)
        
        if len(valid_indices) == 0:
            raise ValueError("没有有效的数据步骤")
        
        # 读取右夹爪反馈数据（从 _gripper_feedback_R）
        # 第一列（索引0）就是夹爪状态值
        right_gripper_feedback = None
        if "_gripper_feedback_R" in f["topics"]:
            gripper_feedback_group = f["topics/_gripper_feedback_R"]
            if "data" in gripper_feedback_group:
                gripper_feedback_data = gripper_feedback_group["data"][:]  # (T, 5)
                print(f"  📊 原始夹爪反馈数据 shape: {gripper_feedback_data.shape}")
                # 取第一列（索引0）作为夹爪状态值
                if len(gripper_feedback_data.shape) > 1:
                    gripper_feedback_data = gripper_feedback_data[:, 0]  # 取第一列
                else:
                    gripper_feedback_data = gripper_feedback_data  # 已经是1D
                # 先过滤有效索引
                gripper_feedback_data = gripper_feedback_data[valid_indices]
                right_gripper_feedback = gripper_feedback_data
                print(f"  ✅ 读取右夹爪反馈数据（第一列）: {len(right_gripper_feedback)} 个值（过滤后）")
                print(f"  📊 夹爪值范围: [{np.nanmin(right_gripper_feedback):.6f}, {np.nanmax(right_gripper_feedback):.6f}]")
            else:
                print("  ⚠️  夹爪反馈话题中没有 'data' 键")
                print(f"  📋 可用键: {list(gripper_feedback_group.keys())}")
        else:
            print("  ⚠️  未找到 _gripper_feedback_R 话题")
            print(f"  📋 可用 topics: {list(f['topics'].keys())[:10]}...")  # 只显示前10个
        
        # 提取有效数据
        eef_positions = eef_positions[valid_indices]
        eef_orientations = eef_orientations[valid_indices]
        axis_angles = axis_angles[valid_indices]
        joint_positions = joint_positions[valid_indices]
        image_data = image_data[valid_indices]
        if image_lengths is not None:
            image_lengths = image_lengths[valid_indices]
        
        # 提取手腕相机数据（如果存在）
        if wrist_image_data is not None:
            wrist_image_data = wrist_image_data[valid_indices]
            if wrist_image_lengths is not None:
                wrist_image_lengths = wrist_image_lengths[valid_indices]
        
        # 下采样：每 downsample_factor 帧取1帧
        downsampled_indices = np.arange(0, len(eef_positions), downsample_factor)
        eef_positions = eef_positions[downsampled_indices]
        eef_orientations = eef_orientations[downsampled_indices]
        axis_angles = axis_angles[downsampled_indices]
        joint_positions = joint_positions[downsampled_indices]
        image_data = image_data[downsampled_indices]
        if image_lengths is not None:
            image_lengths = image_lengths[downsampled_indices]
        
        # 下采样手腕相机数据（如果存在）
        if wrist_image_data is not None:
            wrist_image_data = wrist_image_data[downsampled_indices]
            if wrist_image_lengths is not None:
                wrist_image_lengths = wrist_image_lengths[downsampled_indices]
        
        # 下采样夹爪反馈数据（与关节数据同步）
        if right_gripper_feedback is not None:
            right_gripper_feedback = right_gripper_feedback[downsampled_indices]
            print(f"  ✅ 下采样后右夹爪反馈数据: {len(right_gripper_feedback)} 个值")
        else:
            # 如果夹爪数据不可用，创建零数组
            right_gripper_feedback = np.zeros(len(eef_positions))
            print("  ⚠️  使用零夹爪值（未找到夹爪数据）")
        
        # 解码主相机图像
        print(f"  处理 {len(downsampled_indices)} 张主相机图像（下采样 {downsample_factor}x）...")
        if len(image_data.shape) == 4:
            # 新格式：已经是解码后的图像数组 (T, H, W, 3)
            # 注意：HDF5中存储的可能是BGR格式，需要转换为RGB
            images = image_data.astype(np.uint8)
            # 转换BGR到RGB（交换通道顺序）
            images = images[..., ::-1]  # 反转最后一个维度 (BGR -> RGB)
            print(f"  ✅ 主相机图像已解码并转换为RGB，形状: {images.shape}")
        else:
            # 旧格式：需要解码扁平化的字节数据
            images = []
            for i in tqdm(range(len(downsampled_indices)), desc="  解码主相机图像", leave=False):
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
            print(f"  处理 {len(downsampled_indices)} 张手腕相机图像（下采样 {downsample_factor}x）...")
            if len(wrist_image_data.shape) == 4:
                # 新格式：已经是解码后的图像数组 (T, H, W, 3)
                # 注意：HDF5中存储的可能是BGR格式，需要转换为RGB
                wrist_images = wrist_image_data.astype(np.uint8)
                # 转换BGR到RGB（交换通道顺序）
                wrist_images = wrist_images[..., ::-1]  # 反转最后一个维度 (BGR -> RGB)
                print(f"  ✅ 手腕相机图像已解码并转换为RGB，形状: {wrist_images.shape}")
            else:
                # 旧格式：需要解码扁平化的字节数据
                wrist_images = []
                for i in tqdm(range(len(downsampled_indices)), desc="  解码手腕相机图像", leave=False):
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
        
        # 提取右臂关节（列 7-13，对应 Joint1_R 到 Joint7_R）
        # 注意：joint_states 有14维：前7维是左臂（Joint1_L 到 Joint7_L），后7维是右臂（Joint1_R 到 Joint7_R）
        right_joint_positions = joint_positions[:, 7:14]  # (T, 7) - 右臂关节位置
        
        # 处理关节位置的 NaN
        right_joint_positions = handle_nan_values(right_joint_positions)
        print(f"  ✅ 处理后的右臂关节位置: {right_joint_positions.shape}")
        
        # 计算几何 action（从 EEF 位置和方向计算）
        print("  🔄 计算几何 action...")
        eef_actions = np.zeros((len(eef_positions) - 1, 6))  # (T-1, 6)
        
        for i in range(len(eef_positions) - 1):
            ee_pos_t = eef_positions[i]
            ee_ori_t = axis_angles[i]
            ee_pos_t1 = eef_positions[i + 1]
            ee_ori_t1 = axis_angles[i + 1]
            
            action_6d = compute_geom_action(ee_pos_t, ee_ori_t, ee_pos_t1, ee_ori_t1)
            eef_actions[i] = action_6d
        
        # 最后一个 action 应该是全 0（不动）
        last_action = np.zeros((1, 6))
        eef_actions = np.concatenate([eef_actions, last_action], axis=0)  # (T, 6)
        
        print(f"  ✅ EEF action 维度: {eef_actions.shape} (6维: xyz位移 + 轴角旋转)")
        
        # 处理夹爪反馈数据
        # State 的夹爪部分：直接使用原始的第一维数值（只处理 NaN）
        gripper_states_for_state = right_gripper_feedback.copy()
        # 处理 NaN 值
        gripper_states_for_state = handle_nan_values(gripper_states_for_state)
        print(f"  ✅ 夹爪状态（原始值，用于 state）: {gripper_states_for_state.shape}, 范围 [{np.nanmin(gripper_states_for_state):.6f}, {np.nanmax(gripper_states_for_state):.6f}]")
        
        # Action 的夹爪部分：读取后一个时刻的 gripper 值
        # 小于等于 0.4 就是 0，大于 0.4 就是 1
        gripper_actions = np.zeros(len(gripper_states_for_state), dtype=np.float32)
        for i in range(len(gripper_states_for_state) - 1):
            # 读取后一个时刻的 gripper 值
            next_gripper = gripper_states_for_state[i + 1]
            if next_gripper <= 0.4:
                gripper_actions[i] = 0.0
            else:
                gripper_actions[i] = 1.0
        # 最后一个时间步没有后一个时刻，action 为 0（不动）
        gripper_actions[-1] = 0.0
        print(f"  ✅ 夹爪 action: {gripper_actions.shape}, 范围 [{np.min(gripper_actions):.2f}, {np.max(gripper_actions):.2f}]")
        print(f"  📊 夹爪 action 统计: 0 的数量={np.sum(gripper_actions == 0)}, 1 的数量={np.sum(gripper_actions == 1)}")
        
        # 组合 action（6维 EEF action + 1维夹爪 action） = 7维
        actions = np.concatenate([eef_actions, gripper_actions[:, None]], axis=1)  # (T, 7)
        print(f"  ✅ 最终 action 维度: {actions.shape} (6维 EEF + 1维夹爪)")
        
        # 组合 state（3维EEF位置 + 3维EEF方向(轴角) + 1维夹爪值 + 1维夹爪值相反数） = 8维
        # State 的夹爪部分直接使用 _gripper_feedback_R 的第一维原始数值及其相反数
        states = np.concatenate([
            eef_positions,  # (T, 3) - EEF位置
            axis_angles,    # (T, 3) - EEF方向(轴角)
            gripper_states_for_state[:, None],  # (T, 1) - 夹爪值
            -gripper_states_for_state[:, None],  # (T, 1) - 夹爪值相反数
        ], axis=1)  # (T, 8)
        print(f"  ✅ 最终 state 维度: {states.shape} (3维EEF位置 + 3维EEF方向 + 1维夹爪值 + 1维夹爪值相反数)")
        
        # 确保所有数据长度一致
        assert len(actions) == len(states) == len(images), \
            f"数据长度不一致: actions={len(actions)}, states={len(states)}, images={len(images)}"
        
        # 去掉开始和最后的2帧（总共去掉4帧）
        trim_frames = 2
        if len(states) <= trim_frames * 2:
            print(f"  ⚠️  警告：数据长度 ({len(states)}) 不足以去掉 {trim_frames * 2} 帧，跳过修剪")
        else:
            print(f"  ✂️  去掉开始和最后的各 {trim_frames} 帧（总共 {trim_frames * 2} 帧）")
            print(f"     修剪前: {len(states)} 帧")
            # 去掉前2帧和后2帧
            states = states[trim_frames:-trim_frames]
            actions = actions[trim_frames:-trim_frames]
            images = images[trim_frames:-trim_frames]
            if wrist_images is not None:
                wrist_images = wrist_images[trim_frames:-trim_frames]
            print(f"     修剪后: {len(states)} 帧")
        
        # 转换为步骤列表
        steps = []
        for i in range(len(states)):
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
    fps: float = 7.5,  # 原始30fps下采样4倍后为7.5fps
):
    """
    主函数：将 pick_blue_bottle HDF5 格式数据转换为 LeRobot 格式（下采样4倍版本）
    
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
                "shape": (7,),  # 7 维动作（6维 EEF action + 1维夹爪 action）
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
    
    # 查找所有 HDF5 文件（递归搜索子目录）
    hdf5_files = sorted(
        list(data_dir.glob("**/*.h5")) + list(data_dir.glob("**/*.hdf5"))
    )
    if not hdf5_files:
        raise FileNotFoundError(f"在目录 '{data_dir}' 及其子目录中找不到任何 .h5 或 .hdf5 文件")
    
    print(f"找到 {len(hdf5_files)} 个 HDF5 文件")
    print(f"下采样因子: {downsample_factor}x")
    print(f"输出帧率: {fps} fps")
    
    # 遍历所有 HDF5 文件
    total_steps = 0
    for hdf5_path in tqdm(hdf5_files, desc="处理 HDF5 文件"):
        try:
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
            print(f"✅ 成功转换 {hdf5_path.name} ({len(steps)} 步，下采样 {downsample_factor}x)")
            
        except Exception as e:
            print(f"❌ 处理 {hdf5_path} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✅ 转换完成！总共 {total_steps} 步（下采样 {downsample_factor}x）")
    print(f"数据集保存在: {output_path}")
    
    # 可选：推送到 Hugging Face Hub
    if push_to_hub:
        print("\n推送到 Hugging Face Hub...")
        dataset.push_to_hub(
            tags=["libero", "panda", "downsampled"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print(f"✅ 已推送到 Hub: {REPO_NAME}")


if __name__ == "__main__":
    tyro.cli(main)

