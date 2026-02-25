#!/usr/bin/env python3
"""
策略服务器 - Pick Blue Bottle 微调模型
基于 serve_policy.py 重新编写，硬编码 checkpoint 路径
"""

import dataclasses
import logging
import pathlib
import socket
import sys

import numpy as np
import tyro

# 确保 openpi 模块可以被导入
# openpi 模块在 src/ 目录下
# 注意：建议使用 `uv run` 运行此脚本，以确保所有依赖正确加载
project_root = pathlib.Path(__file__).parent.parent.absolute()
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from openpi.policies import policy as _policy
    from openpi.policies import policy_config as _policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as _config
    from openpi_client import base_policy as _base_policy
except ImportError as e:
    print(f"❌ Failed to import openpi modules: {e}")
    print("   Please run this script with: uv run scripts/serve_policy_pick_blue_bottle.py")
    print("   Or ensure you're in the correct Python environment with all dependencies installed.")
    sys.exit(1)


# =========================
# Hard-coded Configuration
# =========================

# Checkpoint 路径（硬编码）
CHECKPOINT_DIR = pathlib.Path(
    "/home/wyz/openpi/checkpoints/"
    "pi05_pick_blue_bottle_libero_downsample4x/"
    "plate_new_finetune/"
    "30000"  # 使用最新的 30000 步 checkpoint
)

# Config 名称
CONFIG_NAME = "pi05_pick_blue_bottle_libero_downsample4x"

# 默认 prompt
DEFAULT_PROMPT = "Pick up the blue square and move it to the blue plate"


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # If provided, will be used in case the "prompt" key is not present in the data
    default_prompt: str | None = DEFAULT_PROMPT

    # Port to serve the policy on
    port: int = 8000

    # Record the policy's behavior for debugging
    record: bool = False


def create_policy(args: Args) -> _base_policy.BasePolicy:
    """Create a policy from the hard-coded checkpoint."""
    checkpoint_path = CHECKPOINT_DIR
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_path}\n"
            f"Please check the path in the script."
        )
    
    logging.info(f"Loading checkpoint from: {checkpoint_path}")
    logging.info(f"Using config: {CONFIG_NAME}")
    
    # Load config
    train_config = _config.get_config(CONFIG_NAME)
    
    # Create base policy
    base_policy = _policy_config.create_trained_policy(
        train_config,
        str(checkpoint_path),
        default_prompt=args.default_prompt,
    )
    
    logging.info("✅ Policy created (actions already unnormalized by OpenPI)")
    
    return base_policy


def warmup_policy(policy: _base_policy.BasePolicy, default_prompt: str) -> None:
    """
    Warmup the policy by running a dummy inference.
    This compiles the model and prepares it for faster subsequent inference.
    """
    logging.info("🔥 Warming up policy (compiling model for faster inference)...")
    
    # Create a dummy observation matching LIBERO format
    dummy_observation = {
        "observation/state": np.zeros(8, dtype=np.float32),  # 8-dim state
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),  # HWC format
        "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),  # HWC format
        "prompt": default_prompt,
    }
    
    try:
        # Run a dummy inference to trigger compilation
        result = policy.infer(dummy_observation)
        logging.info(f"✅ Policy warmup completed. Action shape: {result.get('actions', 'N/A')}")
        
        # Log action sample for sanity check
        if "actions" in result:
            actions = result["actions"]
            if isinstance(actions, np.ndarray) and actions.size > 0:
                sample_action = actions[0] if actions.ndim > 1 else actions
                logging.info(
                    "Warmup action sample (final action): %s",
                    sample_action
                )
    except Exception as e:
        logging.warning(f"⚠️  Policy warmup failed (non-fatal): {e}")
        logging.warning("   First inference may be slower due to compilation delay.")


def main(args: Args) -> None:
    """Main function to start the policy server."""
    logging.basicConfig(level=logging.INFO, force=True)
    
    # Create policy (actions are already unnormalized by OpenPI)
    policy = create_policy(args)
    policy_metadata = getattr(policy, 'metadata', {})
    
    # Warmup policy to compile model for faster inference
    warmup_policy(policy, args.default_prompt or DEFAULT_PROMPT)

    # Record the policy's behavior if requested
    if args.record:
        from openpi.policies import policy as _policy
        policy = _policy.PolicyRecorder(policy, "policy_records")

    # Get hostname and IP
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s, port: %d)", hostname, local_ip, args.port)

    # Create and start server
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    
    logging.info("=" * 80)
    logging.info("✅ Policy server started successfully!")
    logging.info(f"   Checkpoint: {CHECKPOINT_DIR}")
    logging.info(f"   Config: {CONFIG_NAME}")
    logging.info(f"   Default prompt: {args.default_prompt}")
    logging.info(f"   Port: {args.port}")
    logging.info(f"   Host: {hostname} ({local_ip})")
    logging.info("=" * 80)
    
    server.serve_forever()


if __name__ == "__main__":
    main(tyro.cli(Args))
