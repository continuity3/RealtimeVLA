#!/bin/bash
# 启动 Pick Blue Bottle ROS2 订阅节点

echo "🚀 Starting Pick Blue Bottle ROS2 Subscriber..."
echo "   Policy Server: localhost:8000"
echo ""

cd /home/wyz/openpi

# 检查参数
MODE="${1:-ros2}"  # 默认 ROS2 模式，可以用 "test" 来运行测试模式

if [ "$MODE" == "test" ]; then
    echo "🧪 Running in TEST MODE (no ROS2 required)"
    uv run packages/pose_subscriber_pick_blue_bottle.py \
        --test-mode \
        --host localhost \
        --port 8000 \
        --use-realsense \
        --show-camera
else
    echo "🤖 Running in ROS2 MODE"
    
    # Source ROS2 环境（如果存在）
    if [ -f "/opt/ros/humble/setup.bash" ]; then
        source /opt/ros/humble/setup.bash
        echo "   ✅ ROS2 Humble environment sourced"
    elif [ -f "/opt/ros/foxy/setup.bash" ]; then
        source /opt/ros/foxy/setup.bash
        echo "   ✅ ROS2 Foxy environment sourced"
    fi
    
    # 尝试使用 packages/.venv (Python 3.10, ROS2 兼容)
    # 如果依赖缺失，会回退到 uv run (Python 3.11, 会自动切换到测试模式)
    if [ -f "packages/.venv/bin/python3" ]; then
        echo "   Trying packages/.venv/bin/python3 (Python 3.10, ROS2 compatible)..."
        export PYTHONPATH="${PYTHONPATH}:$(pwd)/src:$(pwd)"
        
        # 检查依赖是否可用
        if packages/.venv/bin/python3 -c "import etils" 2>/dev/null; then
            echo "   ✅ Dependencies available, using packages/.venv"
            packages/.venv/bin/python3 packages/pose_subscriber_pick_blue_bottle.py \
                --host localhost \
                --port 8000 \
                --use-realsense \
                --publish-actions \
                --record data/recordings
        else
            echo "   ⚠️  Dependencies missing in packages/.venv"
            echo "   💡 To install: cd packages && uv pip install etils etils[epath]"
            echo "   🔄 Falling back to uv run (will auto-switch to test mode if ROS2 unavailable)"
            uv run packages/pose_subscriber_pick_blue_bottle.py \
                --host localhost \
                --port 8000 \
                --use-realsense \
                --publish-actions \
                --record data/recordings
        fi
    else
        echo "   Using uv run (if ROS2 unavailable, will auto-switch to test mode)"
        export PYTHONPATH="${PYTHONPATH}:$(pwd)/src:$(pwd)"
        uv run packages/pose_subscriber_pick_blue_bottle.py \
            --host localhost \
            --port 8000 \
            --use-realsense \
            --publish-actions \
            --record data/recordings
    fi
fi

