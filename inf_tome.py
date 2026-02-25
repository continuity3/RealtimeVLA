import os
import sys
import time
import math
import tqdm
import imageio
import pathlib
import dataclasses
import collections
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import importlib

from datasets import load_dataset
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi_client import image_tools

sys.path.append("/home/wyz/openpi/third_party/libero")
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["TORCH_DISABLE_DYNAMO"] = "1"
os.environ["PYTORCH_JIT"] = "0"

# ✅ 强制重新加载 transformers 模块以确保使用最新的代码（包含 ToMe 修改）
print("🔄 Reloading transformers modules to ensure ToMe modifications are loaded...")
modules_to_reload = [
    'transformers.models.paligemma.modeling_paligemma',
    'transformers.models.siglip.modeling_siglip',
]

for module_name in modules_to_reload:
    if module_name in sys.modules:
        del sys.modules[module_name]
        print(f"  ✅ Cleared cache for {module_name}")

# 重新导入以确保使用最新代码
try:
    from transformers import PaliGemmaForConditionalGeneration
    print("  ✅ Successfully reloaded transformers modules")
except Exception as e:
    print(f"  ⚠️ Warning during reload: {e}")

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


# ------------------------------------------------------------
# ✅ 加载策略
# ------------------------------------------------------------
def load_policy():
    print("Loading policy checkpoint ...")
    config = _config.get_config("pi05_libero")
    ckpt = "/home/wyz/openpi/checkpoints/pi05_libero_pytorch.pt"

    def _no_compile(x, *args, **kwargs):
        return x
    torch.compile = _no_compile

    try:
        torch.jit._state.disable()
    except Exception:
        pass

    policy = policy_config.create_trained_policy(config, ckpt, pytorch_device="cuda:0")
    print("✅ Policy loaded:", type(policy))
    if hasattr(policy, "model"):
        print(" → Using model:", policy.model.__class__.__module__)
    return policy


# ------------------------------------------------------------
# ✅ 工具函数
# ------------------------------------------------------------
def _quat2axisangle(quat):
    quat = np.array(quat)
    quat = np.clip(quat, -1.0, 1.0)
    den = np.sqrt(max(1e-6, 1.0 - quat[3] ** 2))
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _get_libero_env(task, resolution, seed):
    task_desc = task.language
    bddl_path = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": bddl_path, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_desc


# ------------------------------------------------------------
# ✅ 参数类
# ------------------------------------------------------------
@dataclasses.dataclass
class Args:
    task_suite_name: str = "libero_spatial"
    num_trials_per_task: int = 1       # ✅ 每个任务只跑一次
    num_steps_wait: int = 10
    replan_steps: int = 5
    resize_size: int = 224
    video_out_path: str = "./libero_videos_prune"
    seed: int = 7


# ------------------------------------------------------------
# ✅ 主评估函数
# ------------------------------------------------------------
def eval_libero_prune(args: Args):
    policy = load_policy()
    np.random.seed(args.seed)

    # ----------------------
    print("🔥 Warming up policy (compile + autotune)...")
    dummy_element = {
        "observation/image": np.zeros((args.resize_size, args.resize_size, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((args.resize_size, args.resize_size, 3), dtype=np.uint8),
        "observation/state": np.zeros(8, dtype=np.float32),
        "prompt": "dummy task",
    }
    with torch.inference_mode():
        _ = policy.infer(dummy_element)
    torch.cuda.synchronize()
    print("✅ Warmup done, start evaluation...\n")

    # ----------------------
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks = task_suite.n_tasks

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    # 最大步数设定
    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    else:
        max_steps = 400

    total_episodes, total_successes = 0, 0
    total_infer_times = []

    print(f"\n🔹 Running evaluation on suite: {args.task_suite_name} ({num_tasks} tasks)")

    for task_id in tqdm.tqdm(range(num_tasks), desc="Tasks"):
        task = task_suite.get_task(task_id)
        init_states = task_suite.get_task_init_states(task_id)
        init_states = init_states[:args.num_trials_per_task]

        env, desc = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        task_episodes, task_successes = 0, 0
        task_infer_times = []

        for ep in range(args.num_trials_per_task):
            obs = env.set_init_state(init_states[ep])
            env.reset()
            action_plan = collections.deque()
            replay_imgs = []

            # 环境稳定等待
            for _ in range(args.num_steps_wait):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

            done = False
            t = 0
            while t < max_steps + args.num_steps_wait:
                try:
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
                    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, args.resize_size, args.resize_size))
                    replay_imgs.append(img)

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist,
                            "observation/state": np.concatenate(
                                (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                            ),
                            "prompt": str(desc),
                        }

                        # ✅ 纯推理时间测量
                        with torch.inference_mode():
                            torch.cuda.synchronize()
                            t0 = time.perf_counter()
                            actions = policy.infer(element)["actions"]
                            torch.cuda.synchronize()
                            infer_time = (time.perf_counter() - t0) * 1000
                            task_infer_times.append(infer_time)

                        action_plan.extend(actions[: args.replan_steps])

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())

                    if done:
                        task_successes += 1
                        total_successes += 1
                        break

                    t += 1

                except Exception as e:
                    print(f"[⚠️] Exception in task {task_id}, ep {ep}: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            tag = "success" if done else "failure"
            seg = desc.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{seg}_{tag}_ep{ep}.mp4",
                [np.asarray(x) for x in replay_imgs],
                fps=10,
            )

        # ===== 每个任务结果 =====
        sr = 100 * task_successes / max(task_episodes, 1)
        total_sr = 100 * total_successes / max(total_episodes, 1)
        avg_infer = np.mean(task_infer_times)
        total_infer_times.extend(task_infer_times)

        print(f"✅ Task {task_id}: {desc}")
        print(f" Success rate: {task_successes}/{task_episodes} = {sr:.1f}%")
        print(f" Avg inference time: {avg_infer:.2f} ms")
        print(f" Overall success: {total_successes}/{total_episodes} = {total_sr:.1f}%\n")

    # ===== 汇总结果 =====
    print("=" * 60)
    print(f"🎯 Final success rate on {args.task_suite_name}: {total_successes}/{total_episodes} = {(100 * total_successes / total_episodes):.2f}%")
    print(f"⚙️ Avg inference time over all tasks: {np.mean(total_infer_times):.2f} ± {np.std(total_infer_times):.2f} ms")
    print("=" * 60)


# ------------------------------------------------------------
# ✅ 运行入口
# ------------------------------------------------------------
if __name__ == "__main__":
    args = Args(task_suite_name="libero_spatial", num_trials_per_task=1)  # ✅ 每个任务只跑一次
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        eval_libero_prune(args)

