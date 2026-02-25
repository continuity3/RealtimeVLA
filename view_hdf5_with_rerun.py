"""
使用 rerun 读取和可视化 HDF5 文件。

这个脚本读取 HDF5 文件，并使用 rerun 进行可视化：
- 图像（RGB、深度图等）
- 夹爪反馈数据
- 时间序列数据

Usage:
    python view_hdf5_with_rerun.py /path/to/test_output.h5
"""

import sys
from pathlib import Path

import numpy as np
import h5py
import rerun as rr


def quaternion_to_direction(quat, base_direction=np.array([0, 0, 1])):
    """
    将四元数转换为方向向量。
    
    Args:
        quat: 四元数 (x, y, z, w) 或 (w, x, y, z)
        base_direction: 基础方向向量（默认 z 轴）
    
    Returns:
        旋转后的方向向量
    """
    # 假设四元数格式为 (x, y, z, w)
    if len(quat) == 4:
        x, y, z, w = quat
    else:
        return base_direction
    
    # 四元数旋转公式
    # v' = q * v * q^-1
    # 简化版本：直接计算旋转后的向量
    # 对于单位向量 [0, 0, 1]，旋转后的向量为：
    # v' = 2 * (w^2 + z^2 - 0.5) * [0, 0, 1] + 2 * (x*z - w*y) * [1, 0, 0] + 2 * (w*x + y*z) * [0, 1, 0]
    # 简化：直接使用四元数旋转公式
    vx, vy, vz = base_direction
    qx, qy, qz, qw = quat
    
    # 四元数旋转向量公式
    t2 = qw * qx
    t3 = qw * qy
    t4 = qw * qz
    t5 = -qx * qx
    t6 = qx * qy
    t7 = qx * qz
    t8 = -qy * qy
    t9 = qy * qz
    t10 = -qz * qz
    
    vx_new = 2 * ((t8 + t10) * vx + (t6 - t4) * vy + (t3 + t7) * vz) + vx
    vy_new = 2 * ((t4 + t6) * vx + (t5 + t10) * vy + (t9 - t2) * vz) + vy
    vz_new = 2 * ((t7 - t3) * vx + (t2 + t9) * vy + (t5 + t8) * vz) + vz
    
    return np.array([vx_new, vy_new, vz_new])


def read_hdf5(hdf5_path: Path):
    """
    读取 HDF5 文件。
    
    Returns:
        dict: 包含图像、夹爪数据、时间戳等的字典
    """
    print(f"正在读取 HDF5 文件: {hdf5_path}")
    
    data = {}
    
    with h5py.File(hdf5_path, 'r') as f:
        # 读取时间戳
        if 'time' in f:
            data['time'] = f['time'][:]
            print(f"时间戳形状: {data['time'].shape}")
            print(f"时间范围: [{data['time'][0]:.3f}, {data['time'][-1]:.3f}]")
        else:
            data['time'] = None
        
        # 读取 topics
        if 'topics' in f:
            topics_group = f['topics']
            data['topics'] = {}
            
            # 遍历所有 topic
            for topic_name in topics_group.keys():
                topic_group = topics_group[topic_name]
                topic_info = {}
                
                # 检查是否有 data 字段
                if 'data' in topic_group:
                    topic_data = topic_group['data'][:]
                    topic_info['data'] = topic_data
                    topic_info['shape'] = topic_data.shape
                    topic_info['dtype'] = topic_data.dtype
                    
                    # 如果有 names 字段，也读取
                    if 'names' in topic_group:
                        names = topic_group['names'][:]
                        # 尝试解码字符串
                        if names.dtype.kind == 'S':  # 字节字符串
                            names = [n.decode('utf-8') if isinstance(n, bytes) else n for n in names]
                        topic_info['names'] = names
                    
                    print(f"Topic {topic_name}: shape={topic_data.shape}, dtype={topic_data.dtype}")
                
                # 检查是否有 position 和 orientation 字段（用于 3D 轨迹）
                if 'position' in topic_group:
                    topic_info['position'] = topic_group['position'][:]
                    print(f"  -> position: shape={topic_info['position'].shape}")
                
                if 'orientation' in topic_group:
                    topic_info['orientation'] = topic_group['orientation'][:]
                    print(f"  -> orientation: shape={topic_info['orientation'].shape}")
                
                if topic_info:
                    data['topics'][topic_name] = topic_info
        
        # 读取 valid 标记（如果有）
        if 'valid' in f:
            valid_group = f['valid']
            data['valid'] = {}
            for key in valid_group.keys():
                data['valid'][key] = valid_group[key][:]
                print(f"Valid {key}: shape={data['valid'][key].shape}")
        
        # 读取 meta（如果有）
        if 'meta' in f:
            meta_group = f['meta']
            data['meta'] = {}
            for key in meta_group.keys():
                item = meta_group[key]
                if isinstance(item, h5py.Dataset):
                    data['meta'][key] = item[:]
                else:
                    data['meta'][key] = dict(item.attrs)
            print(f"Meta keys: {list(data['meta'].keys())}")
    
    return data


def visualize_with_rerun(data: dict, hdf5_path: Path):
    """
    使用 rerun 可视化数据。
    
    Args:
        data: 从 HDF5 文件读取的数据
        hdf5_path: HDF5 文件路径（用于设置标题）
    """
    # 初始化 rerun
    rr.init("HDF5 数据可视化", spawn=True)
    
    # 确定帧数
    num_frames = 0
    if 'topics' in data and data['topics']:
        # 从第一个 topic 获取帧数
        first_topic = list(data['topics'].values())[0]
        if 'data' in first_topic:
            num_frames = first_topic['data'].shape[0]
    
    if num_frames == 0:
        print("❌ 没有数据可可视化")
        return
    
    print(f"\n可视化 {num_frames} 帧数据...")
    
    # 获取时间戳
    timestamps = data.get('time')
    if timestamps is None:
        timestamps = np.arange(num_frames)
    
    # 记录每一帧
    for frame_idx in range(num_frames):
        # 使用时间戳作为时间轴
        if timestamps is not None:
            # 将纳秒时间戳转换为秒（float）
            time_seconds = float(timestamps[frame_idx] / 1e9)
            rr.set_time("time", timestamp=time_seconds)
        else:
            rr.set_time("frame", sequence=frame_idx)
        
        # 遍历所有 topics
        for topic_name, topic_info in data.get('topics', {}).items():
            # 跳过无效帧
            if 'valid' in data:
                valid_key = topic_name
                if valid_key in data['valid']:
                    if not data['valid'][valid_key][frame_idx]:
                        continue
            
            # 处理位置和方向数据（3D 轨迹）
            if 'position' in topic_info and 'orientation' in topic_info:
                position = topic_info['position'][frame_idx]  # (3,)
                orientation = topic_info['orientation'][frame_idx]  # (4,) quaternion
                clean_name = topic_name.replace('_', '/').replace('topics/', '')
                
                # 记录 3D 位置点
                rr.log(f"trajectories/{clean_name}/position", rr.Points3D([position], colors=[[255, 0, 0]]))
                
                # 记录方向（使用四元数创建箭头）
                try:
                    direction_vec = quaternion_to_direction(orientation, base_direction=np.array([0, 0, 0.05]))
                    rr.log(
                        f"trajectories/{clean_name}/orientation",
                        rr.Arrows3D(
                            origins=[position],
                            vectors=[direction_vec],
                            colors=[[0, 255, 0]],
                        ),
                    )
                except Exception as e:
                    pass  # 如果转换失败，跳过方向箭头
                
                # 记录标量值
                for i, coord in enumerate(['x', 'y', 'z']):
                    rr.log(f"scalars/{clean_name}/position_{coord}", rr.Scalars(position[i]))
            
            # 如果没有 data 字段，跳过
            if 'data' not in topic_info:
                continue
                
            topic_data = topic_info['data']
            
            # 处理图像数据
            if len(topic_data.shape) == 4 and topic_data.shape[-1] == 3:
                # 图像 (T, H, W, 3) - 可能是 BGR 格式（来自 OpenCV/ROS）
                img = topic_data[frame_idx]
                if img.size > 0:
                    # 确保数据类型正确
                    if img.dtype != np.uint8:
                        # 归一化到 0-255
                        if img.max() <= 1.0:
                            img = (img * 255).astype(np.uint8)
                        else:
                            img = img.astype(np.uint8)
                    
                    # 将 BGR 转换为 RGB（OpenCV/ROS 使用 BGR，rerun 期望 RGB）
                    img = img[..., ::-1]  # 反转最后一个维度 (BGR -> RGB)
                    
                    # 清理 topic 名称用于显示
                    clean_name = topic_name.replace('_', '/').replace('topics/', '')
                    rr.log(f"images/{clean_name}", rr.Image(img))
            
            # 处理深度图像
            elif len(topic_data.shape) == 4 and topic_data.shape[-1] == 1:
                # 深度图像 (T, H, W, 1)
                depth_img = topic_data[frame_idx, :, :, 0]
                if depth_img.size > 0:
                    clean_name = topic_name.replace('_', '/').replace('topics/', '')
                    rr.log(f"depth/{clean_name}", rr.DepthImage(depth_img, meter=1000.0))
            
            # 处理标量数据（如夹爪反馈）
            elif len(topic_data.shape) == 2:
                # 2D 数组 (T, N)
                values = topic_data[frame_idx]
                clean_name = topic_name.replace('_', '/').replace('topics/', '')
                
                # 如果有名称，分别记录每个值
                if 'names' in topic_info and topic_info['names']:
                    names = topic_info['names']
                    for i, name in enumerate(names):
                        if i < len(values):
                            rr.log(f"scalars/{clean_name}/{name}", rr.Scalars(values[i]))
                else:
                    # 否则记录所有值
                    for i, val in enumerate(values):
                        rr.log(f"scalars/{clean_name}/value_{i}", rr.Scalars(val))
            
            # 处理 1D 数据
            elif len(topic_data.shape) == 1:
                value = topic_data[frame_idx]
                clean_name = topic_name.replace('_', '/').replace('topics/', '')
                rr.log(f"scalars/{clean_name}", rr.Scalars(value))
    
    # 记录完整轨迹（作为线）
    for topic_name, topic_info in data.get('topics', {}).items():
        if 'position' in topic_info:
            positions = topic_info['position']  # (T, 3)
            clean_name = topic_name.replace('_', '/').replace('topics/', '')
            # 为不同的轨迹使用不同的颜色
            colors = {
                'left': [255, 0, 0],
                'right': [0, 0, 255],
                'A': [255, 255, 0],
                'B': [0, 255, 255],
            }
            color = [128, 128, 128]  # 默认灰色
            for key, col in colors.items():
                if key.lower() in topic_name.lower():
                    color = col
                    break
            rr.log(f"trajectories/{clean_name}/path", rr.LineStrips3D([positions], colors=[color]))
    
    print(f"\n✅ 可视化完成！共 {num_frames} 帧")
    print(f"Topics: {list(data.get('topics', {}).keys())}")


def main():
    if len(sys.argv) < 2:
        print("用法: python view_hdf5_with_rerun.py <hdf5_file_path>")
        print("\n示例:")
        print("  python view_hdf5_with_rerun.py /home/wyz/realsense_ws/data/rosbag2_2026_01_26-11_46_14/test_output.h5")
        sys.exit(1)
    
    hdf5_path = Path(sys.argv[1])
    
    if not hdf5_path.exists():
        print(f"❌ 文件不存在: {hdf5_path}")
        sys.exit(1)
    
    # 读取数据
    data = read_hdf5(hdf5_path)
    
    # 可视化
    visualize_with_rerun(data, hdf5_path)


if __name__ == "__main__":
    main()

