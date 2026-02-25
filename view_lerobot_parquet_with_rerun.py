"""
使用 rerun 读取和可视化 lerobot parquet 文件。

这个脚本读取 lerobot 格式的 parquet 文件，并使用 rerun 进行可视化：
- 图像（image 和 wrist_image）
- EEF 位置轨迹（3D）
- 状态和动作的时间序列

Usage:
    python view_lerobot_parquet_with_rerun.py /path/to/episode_000000.parquet
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rerun as rr
from scipy.spatial.transform import Rotation as R


def axisangle_to_quat(axis_angle):
    """将轴角转换为四元数 (xyzw)"""
    return R.from_rotvec(axis_angle).as_quat()


def quat_to_axisangle(quat):
    """将四元数 (xyzw) 转换为轴角"""
    return R.from_quat(quat).as_rotvec()


def read_lerobot_parquet(parquet_path: Path):
    """
    读取 lerobot parquet 文件。
    
    Returns:
        dict: 包含 image, wrist_image, state, actions 的字典
    """
    print(f"正在读取 parquet 文件: {parquet_path}")
    
    # 读取 parquet 文件
    df = pd.read_parquet(parquet_path)
    
    print(f"数据形状: {len(df)} 行")
    print(f"列名: {list(df.columns)}")
    
    # 解析数据
    data = {}
    
    # 读取图像
    # lerobot 将图像存储为字典，包含 'bytes' 和 'path' 键
    def decode_image(img_data):
        """解码图像数据，支持多种格式"""
        from PIL import Image
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
                parquet_dir = parquet_path.parent.parent.parent  # 回到数据集根目录
                full_path = parquet_dir / img_path
                if full_path.exists():
                    return np.array(Image.open(full_path).convert("RGB"))
                # 尝试绝对路径
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
            # 尝试相对路径
            parquet_dir = parquet_path.parent.parent.parent
            full_path = parquet_dir / img_data
            if full_path.exists():
                return np.array(Image.open(full_path).convert("RGB"))
        
        return None
    
    if "image" in df.columns:
        images = []
        for idx, img_data in enumerate(df["image"]):
            try:
                img = decode_image(img_data)
                if img is None:
                    print(f"⚠️  无法解析图像 {idx}，类型: {type(img_data)}")
                    img = np.zeros((256, 256, 3), dtype=np.uint8)
                images.append(img)
            except Exception as e:
                print(f"⚠️  解析图像 {idx} 时出错: {e}")
                images.append(np.zeros((256, 256, 3), dtype=np.uint8))
        
        if images:
            # 确保所有图像形状一致
            target_shape = images[0].shape
            images = [img if img.shape == target_shape else np.zeros(target_shape, dtype=np.uint8) for img in images]
            data["image"] = np.array(images)
            print(f"图像形状: {data['image'].shape}")
        else:
            data["image"] = None
    else:
        print("⚠️  未找到 'image' 列")
        data["image"] = None
    
    if "wrist_image" in df.columns:
        wrist_images = []
        for idx, img_data in enumerate(df["wrist_image"]):
            try:
                img = decode_image(img_data)
                if img is None:
                    print(f"⚠️  无法解析手腕图像 {idx}，类型: {type(img_data)}")
                    img = np.zeros((256, 256, 3), dtype=np.uint8)
                wrist_images.append(img)
            except Exception as e:
                print(f"⚠️  解析手腕图像 {idx} 时出错: {e}")
                wrist_images.append(np.zeros((256, 256, 3), dtype=np.uint8))
        
        if wrist_images:
            target_shape = wrist_images[0].shape
            wrist_images = [img if img.shape == target_shape else np.zeros(target_shape, dtype=np.uint8) for img in wrist_images]
            data["wrist_image"] = np.array(wrist_images)
            print(f"手腕图像形状: {data['wrist_image'].shape}")
        else:
            data["wrist_image"] = None
    else:
        print("⚠️  未找到 'wrist_image' 列")
        data["wrist_image"] = None
    
    # 读取状态
    if "state" in df.columns:
        states = []
        for state in df["state"]:
            if isinstance(state, (list, np.ndarray)):
                states.append(np.array(state))
            else:
                states.append(state)
        data["state"] = np.array(states)
        print(f"状态形状: {data['state'].shape}")
        print(f"状态范围: [{np.min(data['state'])}, {np.max(data['state'])}]")
    else:
        print("⚠️  未找到 'state' 列")
        data["state"] = None
    
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
        print(f"动作范围: [{np.min(data['actions'])}, {np.max(data['actions'])}]")
    else:
        print("⚠️  未找到 'actions' 列")
        data["actions"] = None
    
    # 读取任务描述（如果有）
    if "task" in df.columns:
        data["task"] = df["task"].iloc[0] if len(df) > 0 else ""
        print(f"任务: {data['task']}")
    else:
        data["task"] = ""
    
    return data


def visualize_with_rerun(data: dict, parquet_path: Path):
    """
    使用 rerun 可视化数据。
    
    Args:
        data: 从 parquet 文件读取的数据
        parquet_path: parquet 文件路径（用于设置标题）
    """
    # 初始化 rerun
    rr.init("Lerobot 数据可视化", spawn=True)
    
    # 设置时间轴
    num_frames = len(data.get("state", [])) if data.get("state") is not None else 0
    if num_frames == 0:
        print("❌ 没有数据可可视化")
        return
    
    # 解析状态数据
    # state: [3维EEF位置, 3维EEF方向(轴角), 1维夹爪值, 1维夹爪值相反数] = 8维
    states = data["state"]
    eef_positions = states[:, :3]  # (T, 3) - EEF 位置
    eef_axis_angles = states[:, 3:6]  # (T, 3) - EEF 方向（轴角）
    gripper_values = states[:, 6]  # (T,) - 夹爪值
    
    # 解析动作数据
    # actions: [6维 EEF action, 1维夹爪 action] = 7维
    actions = data["actions"]
    eef_actions = actions[:, :6]  # (T, 6) - EEF 动作
    gripper_actions = actions[:, 6]  # (T,) - 夹爪动作
    
    print(f"\n可视化 {num_frames} 帧数据...")
    print(f"EEF 位置范围: x=[{np.min(eef_positions[:, 0]):.3f}, {np.max(eef_positions[:, 0]):.3f}], "
          f"y=[{np.min(eef_positions[:, 1]):.3f}, {np.max(eef_positions[:, 1]):.3f}], "
          f"z=[{np.min(eef_positions[:, 2]):.3f}, {np.max(eef_positions[:, 2]):.3f}]")
    
    # 记录每一帧
    for frame_idx in range(num_frames):
        rr.set_time_sequence("frame", frame_idx)
        
        # 记录图像
        if data["image"] is not None and len(data["image"]) > frame_idx:
            img = data["image"][frame_idx]
            if img.size > 0 and len(img.shape) >= 2:
                rr.log("camera/image", rr.Image(img))
        
        if data["wrist_image"] is not None and len(data["wrist_image"]) > frame_idx:
            wrist_img = data["wrist_image"][frame_idx]
            if wrist_img.size > 0 and len(wrist_img.shape) >= 2:
                rr.log("camera/wrist_image", rr.Image(wrist_img))
        
        # 记录 EEF 位置（3D 点）
        eef_pos = eef_positions[frame_idx]
        rr.log("robot/eef_position", rr.Points3D([eef_pos], colors=[[255, 0, 0]]))
        
        # 记录 EEF 方向（使用轴角转换为四元数，然后记录为箭头）
        axis_angle = eef_axis_angles[frame_idx]
        quat = axisangle_to_quat(axis_angle)
        # 创建一个方向向量（从四元数）
        rot = R.from_quat(quat)
        direction = rot.apply([0, 0, 0.1])  # 指向 z 方向，长度为 0.1
        end_point = eef_pos + direction
        rr.log(
            "robot/eef_orientation",
            rr.Arrows3D(
                origins=[eef_pos],
                vectors=[direction],
                colors=[[0, 255, 0]],
            ),
        )
        
        # 记录夹爪状态
        gripper_val = gripper_values[frame_idx]
        rr.log("robot/gripper_value", rr.Scalars(gripper_val))
        
        # 记录动作
        eef_action = eef_actions[frame_idx]
        rr.log("action/eef_translation", rr.Scalars(np.linalg.norm(eef_action[:3])))
        rr.log("action/eef_rotation", rr.Scalars(np.linalg.norm(eef_action[3:6])))
        rr.log("action/gripper", rr.Scalars(gripper_actions[frame_idx]))
        
        # 记录时间序列数据点
        rr.log("timeseries/state/eef_x", rr.Scalars(eef_positions[frame_idx, 0]))
        rr.log("timeseries/state/eef_y", rr.Scalars(eef_positions[frame_idx, 1]))
        rr.log("timeseries/state/eef_z", rr.Scalars(eef_positions[frame_idx, 2]))
        rr.log("timeseries/state/gripper", rr.Scalars(gripper_values[frame_idx]))
        rr.log("timeseries/action/eef_translation_norm", rr.Scalars(np.linalg.norm(eef_actions[frame_idx, :3])))
        rr.log("timeseries/action/eef_rotation_norm", rr.Scalars(np.linalg.norm(eef_actions[frame_idx, 3:6])))
        rr.log("timeseries/action/gripper", rr.Scalars(gripper_actions[frame_idx]))
    
    # 记录完整轨迹（作为线）
    rr.log("trajectory/eef_path", rr.LineStrips3D([eef_positions], colors=[[0, 0, 255]]))
    
    print(f"\n✅ 可视化完成！共 {num_frames} 帧")
    print(f"任务: {data.get('task', 'N/A')}")


def main():
    if len(sys.argv) < 2:
        print("用法: python view_lerobot_parquet_with_rerun.py <parquet_file_path>")
        print("\n示例:")
        print("  python view_lerobot_parquet_with_rerun.py /home/wyz/.cache/huggingface/lerobot/your_hf_username/pick_blue_bottle_libero_downsample4x/data/chunk-000/episode_000000.parquet")
        sys.exit(1)
    
    parquet_path = Path(sys.argv[1])
    
    if not parquet_path.exists():
        print(f"❌ 文件不存在: {parquet_path}")
        sys.exit(1)
    
    # 读取数据
    data = read_lerobot_parquet(parquet_path)
    
    # 可视化
    visualize_with_rerun(data, parquet_path)


if __name__ == "__main__":
    main()

