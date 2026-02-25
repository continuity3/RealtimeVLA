"""
通用的 HDF5 格式数据到 LeRobot 格式转换脚本（不降采样版本，仅使用右臂数据）。

这个脚本用于处理 LIBERO 数据集的 HDF5 文件，不进行降采样。
只使用右臂数据（左臂未使用），并包含右夹爪信息。

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
- 状态: [3维EEF位置, 3维EEF方向(轴角), 1维夹爪值, 1维夹爪值相反数] = 8维
- 动作: [6维EEF action (xyz位移 + 轴角旋转), 1维夹爪状态] = 7维

Usage:
# 转换单个文件
uv run examples/libero/convert_hdf5_to_lerobot.py --hdf5_path /path/to/data.h5 --output_repo_name your_hf_username/dataset_name

# 转换目录中的所有文件
uv run examples/libero/convert_hdf5_to_lerobot.py --hdf5_path /path/to/data_dir --output_repo_name your_hf_username/dataset_name
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


def load_hdf5_data(
    hdf5_path: Path,
    task_description: str = "Robot manipulation task",
    ignore_valid: bool = False,
) -> list[dict]:
    """
    从 HDF5 文件中加载数据（不降采样版本）。
    
    使用几何 action 计算方式：
    - Action: 6维 EEF action (xyz位移 + 轴角旋转) + 1维夹爪状态 = 7维
    - State: 3维EEF位置 + 3维EEF方向(轴角) + 1维夹爪值 + 1维夹爪值相反数 = 8维
    
    Args:
        hdf5_path: HDF5 文件路径
        task_description: 任务描述
        ignore_valid: 是否忽略有效性标记
    
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
        
        # 读取图像（支持两种格式）
        image_topic = None
        image_data = None
        image_lengths = None
        image_topic_name = None
        
        if "_camera_image" in f["topics"]:
            # 格式1: 已解码的图像数组
            image_topic = f["topics/_camera_image"]
            image_data = image_topic["data"][:]  # (T, H, W, 3) - 已解码
            image_topic_name = "_camera_image"
            print(f"  ✅ 使用已解码图像格式: {image_data.shape}")
        elif "_camera_camera_color_image_raw" in f["topics"]:
            # 格式2: 需要解码的图像数据
            image_topic = f["topics/_camera_camera_color_image_raw"]
            image_data = image_topic["data"][:]  # (T, 921600)
            image_lengths = image_topic["data_length"][:]  # (T,)
            image_topic_name = "_camera_camera_color_image_raw"
            print(f"  ✅ 使用原始图像格式: {image_data.shape}")
        else:
            raise KeyError("找不到图像 topic（尝试了 _camera_image 和 _camera_camera_color_image_raw）")
        
        # 读取有效性标记（如果有）
        valid = None
        if not ignore_valid and "valid" in f:
            # 优先使用 joint_states 的有效性，如果图像也有效则更好
            if "_joint_states" in f["valid"]:
                valid_joint = f["valid/_joint_states"][:]  # (T,)
            else:
                valid_joint = None
            
            # 检查两种图像格式的有效性
            if image_topic_name == "_camera_image" and "_camera_image" in f["valid"]:
                valid_image = f["valid/_camera_image"][:]  # (T,)
            elif image_topic_name == "_camera_camera_color_image_raw" and "_camera_camera_color_image_raw" in f["valid"]:
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
        min_length = min(len(eef_positions), len(joint_positions), len(image_data))
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
        
        # 提取有效数据（不降采样）
        eef_positions = eef_positions[valid_indices]
        eef_orientations = eef_orientations[valid_indices]
        axis_angles = axis_angles[valid_indices]
        joint_positions = joint_positions[valid_indices]
        image_data = image_data[valid_indices]
        if image_lengths is not None:
            image_lengths = image_lengths[valid_indices]
        
        # 处理夹爪反馈数据（与关节数据同步）
        if right_gripper_feedback is not None:
            print(f"  ✅ 右夹爪反馈数据: {len(right_gripper_feedback)} 个值")
        else:
            # 如果夹爪数据不可用，创建零数组
            right_gripper_feedback = np.zeros(len(eef_positions))
            print("  ⚠️  使用零夹爪值（未找到夹爪数据）")
        
        # 处理图像（根据格式决定是否需要解码）
        if image_topic_name == "_camera_image":
            # 已解码格式：直接使用，但需要确保是 uint8 类型
            print(f"  处理 {len(valid_indices)} 张已解码图像（不降采样）...")
            images = image_data
            # 确保是 uint8 类型
            if images.dtype != np.uint8:
                if images.max() <= 1.0:
                    images = (images * 255).astype(np.uint8)
                else:
                    images = images.astype(np.uint8)
        else:
            # 需要解码的格式
            print(f"  解码 {len(valid_indices)} 张图像（不降采样）...")
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
        
        # 转换为步骤列表
        steps = []
        for i in range(len(states)):
            # 调整图像大小
            image = resize_image(images[i], (256, 256))
            
            # 确保图像是 uint8 且不需要额外缩放
            # decode_image 通常返回 uint8，这里做一个防御性转换
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            else:
                # 确保是 uint8 类型
                image = image.astype(np.uint8)
            
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
    hdf5_path: str,
    output_repo_name: str,
    *,
    push_to_hub: bool = False,
    task_description: str = "Robot manipulation task",
    ignore_valid: bool = False,
    fps: float = 30.0,  # 默认30fps（不降采样）
):
    """
    主函数：将 HDF5 格式数据转换为 LeRobot 格式（不降采样版本）
    
    Args:
        hdf5_path: HDF5 文件路径或包含 HDF5 文件的目录路径
        output_repo_name: 输出数据集名称（格式：your_hf_username/dataset_name）
        push_to_hub: 是否推送到 Hugging Face Hub
        task_description: 任务描述
        ignore_valid: 是否忽略有效性标记，使用所有数据
        fps: 输出数据集的帧率（默认30fps，不降采样）
    """
    # 规范化路径（处理双斜杠等问题）
    hdf5_path = Path(str(hdf5_path).replace("//", "/")).resolve()
    
    # 判断是文件还是目录
    if hdf5_path.is_file():
        # 单个文件
        hdf5_files = [hdf5_path]
    elif hdf5_path.is_dir():
        # 目录：查找所有 HDF5 文件（递归搜索子目录）
        hdf5_files = sorted(
            list(hdf5_path.glob("**/*.h5")) + list(hdf5_path.glob("**/*.hdf5"))
        )
        if not hdf5_files:
            raise FileNotFoundError(f"在目录 '{hdf5_path}' 及其子目录中找不到任何 .h5 或 .hdf5 文件")
    else:
        raise FileNotFoundError(f"路径不存在: {hdf5_path}")
    
    print(f"找到 {len(hdf5_files)} 个 HDF5 文件")
    print(f"输出帧率: {fps} fps（不降采样）")
    print(f"输出数据集: {output_repo_name}")
    
    # 清理输出目录
    output_path = HF_LEROBOT_HOME / output_repo_name
    if output_path.exists():
        shutil.rmtree(output_path)
    
    # 创建 LeRobot 数据集
    dataset = LeRobotDataset.create(
        repo_id=output_repo_name,
        robot_type="panda",
        fps=fps,
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
    
    # 遍历所有 HDF5 文件
    total_steps = 0
    for hdf5_file in tqdm(hdf5_files, desc="处理 HDF5 文件"):
        try:
            steps = load_hdf5_data(hdf5_file, task_description, ignore_valid)
            
            # 写入 LeRobot 数据集
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
            print(f"✅ 成功转换 {hdf5_file.name} ({len(steps)} 步)")
            
        except Exception as e:
            print(f"❌ 处理 {hdf5_file} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✅ 转换完成！总共 {total_steps} 步（不降采样）")
    print(f"数据集保存在: {output_path}")
    
    # 可选：推送到 Hugging Face Hub
    if push_to_hub:
        print("\n推送到 Hugging Face Hub...")
        dataset.push_to_hub(
            tags=["libero", "panda"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print(f"✅ 已推送到 Hub: {output_repo_name}")


if __name__ == "__main__":
    tyro.cli(main)

