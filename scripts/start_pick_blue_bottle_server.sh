#!/bin/bash
# 启动 Pick Blue Bottle 策略服务器

echo "🚀 Starting Pick Blue Bottle Policy Server..."
echo "   Checkpoint: /home/wyz/openpi/checkpoints/pi05_pick_blue_bottle_libero_downsample4x/pick_blue_bottle_finetune/20000"
echo "   Config: pi05_pick_blue_bottle_libero_downsample4x"
echo "   Port: 8000"
echo ""

cd /home/wyz/openpi

# 使用 uv 运行（推荐，与 serve_policy.py 保持一致）
if command -v uv &> /dev/null; then
    uv run python scripts/serve_policy_pick_blue_bottle.py --port 8000
else
    # 或者直接使用 python（脚本内部已设置 PYTHONPATH）
    python3 scripts/serve_policy_pick_blue_bottle.py --port 8000
fi

