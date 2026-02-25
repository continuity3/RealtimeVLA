"""
随机读取10个训练数据样本，保存action为npy，state为txt
"""
import os
import sys
import pathlib
import numpy as np

# 添加项目路径
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from openpi.training import config as _config
from datasets import load_dataset

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["JAX_PLATFORMS"] = "cpu"

print("=" * 80)
print("随机读取10个训练数据样本")
print("=" * 80)

# 加载数据集
config_name = "pi05_pick_blue_bottle_libero_downsample4x"
local_data_path = "/home/wyz/.cache/huggingface/lerobot/your_hf_username/pick_blue_bottle_libero_downsample4x"

config = _config.get_config(config_name)
data_config = config.data.create(config.assets_dirs, config.model)

# 加载数据集
data_dir = pathlib.Path(local_data_path) / "data"
if not data_dir.exists():
    data_dir = pathlib.Path(local_data_path)

parquet_files = sorted(list(data_dir.rglob("*.parquet")))
if parquet_files:
    print(f"Found {len(parquet_files)} parquet files")
    hf_dataset = load_dataset("parquet", data_files=[str(f) for f in parquet_files], split="train")
else:
    hf_dataset = load_dataset("parquet", data_dir=str(data_dir), split="train")

dataset_length = len(hf_dataset)
print(f"Dataset length: {dataset_length}")

# 随机选择10个索引
np.random.seed(42)  # 设置随机种子以便复现
random_indices = np.random.choice(dataset_length, size=10, replace=False)
random_indices = sorted(random_indices)  # 排序以便查看
print(f"Randomly selected indices: {random_indices}")

# 存储所有样本的 action 和 state
all_actions = []
all_states = []

# 读取每个样本的完整 episode 数据
for i, idx in enumerate(random_indices):
    idx = int(idx)  # 转换为 Python int 类型
    sample = hf_dataset[idx]
    
    # 获取当前样本的 episode_index
    episode_idx = sample.get("episode_index", idx)
    
    # 找到该 episode 的所有帧
    episode_actions = []
    episode_states = []
    episode_frame_indices = []
    
    for j in range(len(hf_dataset)):
        frame_sample = hf_dataset[j]
        frame_episode_idx = frame_sample.get("episode_index")
        if frame_episode_idx == episode_idx:
            episode_frame_indices.append(j)
            
            # 获取 action
            action = frame_sample.get("actions")
            if action is not None:
                if hasattr(action, 'numpy'):
                    action = action.numpy()
                elif not isinstance(action, np.ndarray):
                    action = np.array(action)
                if len(action.shape) > 1:
                    action = action[0]
                episode_actions.append(action)
            else:
                episode_actions.append(np.zeros(7))
            
            # 获取 state
            state = frame_sample.get("state")
            if state is not None:
                if hasattr(state, 'numpy'):
                    state = state.numpy()
                elif not isinstance(state, np.ndarray):
                    state = np.array(state)
                if len(state.shape) > 1:
                    state = state[0]
                episode_states.append(state)
            else:
                episode_states.append(np.zeros(8))
    
    # 转换为 numpy 数组
    episode_actions = np.array(episode_actions)  # (T, 7)
    episode_states = np.array(episode_states)    # (T, 8)
    
    all_actions.append(episode_actions)
    all_states.append(episode_states)
    
    # 详细打印每个样本
    print(f"\n{'='*80}")
    print(f"Sample {i+1} (index {idx}, episode {episode_idx}):")
    print(f"{'='*80}")
    print(f"Episode length: {len(episode_actions)} frames")
    print(f"Actions shape: {episode_actions.shape}")
    print(f"States shape: {episode_states.shape}")
    print(f"First action: {episode_actions[0]}")
    print(f"First state: {episode_states[0]}")

print(f"\n✅ Loaded {len(all_actions)} samples (episodes)")
print(f"  Each sample contains a full episode with multiple frames")

# 为每个样本单独保存文件
# 每个样本：1个state.txt + 1个action.txt + 1个action.npy
# 总共：10个state.txt + 10个action.txt + 10个action.npy = 30个文件
print(f"\n{'='*80}")
print("保存每个样本的单独文件...")
print(f"{'='*80}")

for i, (idx, actions, states) in enumerate(zip(random_indices, all_actions, all_states)):
    idx = int(idx)
    sample_num = i + 1
    num_frames = len(actions)
    
    # 保存 state 为 txt 文件（整个 episode 的所有 states）
    state_file = f"sample_{sample_num}_state.txt"
    with open(state_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Sample {sample_num} (index {idx}) - Complete Episode States\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Episode length: {num_frames} frames\n")
        f.write(f"State dimension: {states.shape[1]}\n")
        f.write(f"State format: [EEF_pos(3), EEF_ori(3), gripper_value(1), -gripper_value(1)]\n\n")
        
        for frame_idx, state in enumerate(states):
            f.write(f"Frame {frame_idx + 1}/{num_frames}:\n")
            f.write(f"  EEF Position (x, y, z):          [{state[0]:.6f}, {state[1]:.6f}, {state[2]:.6f}]\n")
            f.write(f"  EEF Orientation (axis-angle):    [{state[3]:.6f}, {state[4]:.6f}, {state[5]:.6f}]\n")
            f.write(f"  Gripper value:                   {state[6]:.6f}\n")
            f.write(f"  -Gripper value:                  {state[7]:.6f}\n")
            f.write(f"  Full state: {state}\n")
            f.write("\n")
    
    # 保存 action 为 txt 文件（整个 episode 的所有 actions）
    action_txt_file = f"sample_{sample_num}_action.txt"
    with open(action_txt_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Sample {sample_num} (index {idx}) - Complete Episode Actions\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Episode length: {num_frames} frames\n")
        f.write(f"Action dimension: {actions.shape[1]}\n")
        f.write(f"Action format: [EEF_delta_pos(3), EEF_delta_ori(3), gripper_action(1)]\n\n")
        
        for frame_idx, action in enumerate(actions):
            f.write(f"Frame {frame_idx + 1}/{num_frames}:\n")
            f.write(f"  EEF delta position (x, y, z):     [{action[0]:.6f}, {action[1]:.6f}, {action[2]:.6f}]\n")
            f.write(f"  EEF delta orientation (axis-angle): [{action[3]:.6f}, {action[4]:.6f}, {action[5]:.6f}]\n")
            f.write(f"  Gripper action:                  {action[6]:.6f}\n")
            f.write(f"  Full action: {action}\n")
            f.write("\n")
    
    # 保存 action 为 npy 文件（整个 episode 的所有 actions）
    action_npy_file = f"sample_{sample_num}_action.npy"
    np.save(action_npy_file, actions)
    
    print(f"  Sample {sample_num} (index {idx}, {num_frames} frames):")
    print(f"    ✅ {state_file} ({num_frames} states)")
    print(f"    ✅ {action_txt_file} ({num_frames} actions)")
    print(f"    ✅ {action_npy_file} (shape: {actions.shape})")

total_frames = sum(len(actions) for actions in all_actions)
print(f"\n✅ 总共保存了 {len(all_actions) * 3} 个文件")
print(f"   - {len(all_states)} 个 state.txt 文件（包含 {total_frames} 个 states）")
print(f"   - {len(all_actions)} 个 action.txt 文件（包含 {total_frames} 个 actions）")
print(f"   - {len(all_actions)} 个 action.npy 文件（包含 {total_frames} 个 actions）")

print("\n" + "=" * 80)
print("完成！")
print("=" * 80)

