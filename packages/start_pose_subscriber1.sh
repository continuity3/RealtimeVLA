#!/bin/bash
# 启动 pose_subscriber1.py (带反归一化功能)

echo "🚀 Starting pose_subscriber1.py with unnormalization..."
echo "   Policy Server: localhost:8000"
echo ""

cd /home/wyz/openpi

# 检查参数
MODE="${1:-ros2}"  # 默认 ROS2 模式，可以用 "test" 来运行测试模式

if [ "$MODE" == "test" ]; then
    echo "🧪 Running in TEST MODE (no ROS2 required)"
    uv run packages/pose_subscriber1.py \
        --test-mode \
        --host localhost \
        --port 8000 \
        --use-realsense \
        --show-camera
else
    echo "🤖 Running in ROS2 MODE"
    echo "   Sourcing ROS2 environment..."
    
    # Source ROS2 环境（如果存在）
    if [ -f "/opt/ros/humble/setup.bash" ]; then
        source /opt/ros/humble/setup.bash
        echo "   ✅ ROS2 Humble environment sourced"
    elif [ -f "/opt/ros/foxy/setup.bash" ]; then
        source /opt/ros/foxy/setup.bash
        echo "   ✅ ROS2 Foxy environment sourced"
    else
        echo "   ⚠️  ROS2 setup.bash not found, trying to continue anyway..."
    fi
    
    # 设置 PYTHONPATH 以确保 openpi 模块可以被导入
    export PYTHONPATH="${PYTHONPATH}:$(pwd)/src:$(pwd)"
    
    # 尝试使用 uv run（如果依赖可用），否则使用系统 Python
    # 注意：uv run 使用 Python 3.11，可能与 ROS2 不兼容
    # 如果 ROS2 导入失败，脚本会自动切换到测试模式
    if command -v uv &> /dev/null; then
        echo "   Using uv run (may fall back to test mode if ROS2 incompatible)..."
        uv run packages/pose_subscriber1.py \
            --host localhost \
            --port 8000 \
            --use-realsense \
            --publish-actions \
            --record data/recordings
    else
        echo "   Using system Python..."
        python3 packages/pose_subscriber1.py \
            --host localhost \
            --port 8000 \
            --use-realsense \
            --publish-actions \
            --record data/recordings
    fi
fi

