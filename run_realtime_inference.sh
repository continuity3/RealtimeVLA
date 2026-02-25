#!/bin/bash

# 获取脚本所在目录
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_DIR"

# 默认参数
HOST="localhost"
PORT=8000
PROMPT="Pick up the blue square and place it in the blue tray."
RECORD_DIR="data/realtime_inference"
INFERENCE_RATE=2.0

if [ "$1" == "server" ]; then
    echo "启动策略服务器..."
    echo "使用官方权重: checkpoints/pi05_libero_pytorch.pt"
    
    PYTHONPATH="" uv run scripts/serve_policy.py \
        --port $PORT \
        policy:local-checkpoint \
        --policy.path="checkpoints/pi05_libero_pytorch.pt" \
        --policy.config=pi05_libero

elif [ "$1" == "client" ]; then
    echo "启动实时推理客户端..."
    
    # 解析参数：检查第二个参数是否是 --test-mode 或 --args.test-mode
    TEST_MODE_FLAG=""
    CUSTOM_PROMPT="$PROMPT"
    
    if [ "$2" == "--test-mode" ] || [ "$2" == "--args.test-mode" ]; then
        TEST_MODE_FLAG="--args.test-mode"
        # 第三个参数可能是 prompt
        if [ -n "$3" ]; then
            CUSTOM_PROMPT="$3"
        fi
    elif [ -n "$2" ] && [[ ! "$2" =~ ^-- ]]; then
        # 第二个参数不是以 -- 开头，可能是 prompt
        CUSTOM_PROMPT="$2"
    fi
    
    # 确保 ROS2 环境已设置（如果不是测试模式）
    if [ -z "$TEST_MODE_FLAG" ]; then
        if [ -f "/opt/ros/humble/setup.bash" ]; then
            source /opt/ros/humble/setup.bash
        elif [ -f "/opt/ros/iron/setup.bash" ]; then
            source /opt/ros/iron/setup.bash
        else
            echo "警告: 未找到 ROS2 setup.bash。请手动 source 或使用 --test-mode。"
        fi
    else
        echo "测试模式: 不使用 ROS2"
    fi
    
    echo "使用 prompt: $CUSTOM_PROMPT"
    echo "记录目录: $RECORD_DIR"
    echo "推理频率: $INFERENCE_RATE Hz"
    
    # 构建命令参数（tyro 需要 --args. 前缀）
    CMD_ARGS=(
        "--args.host" "$HOST"
        "--args.port" "$PORT"
        "--args.prompt" "$CUSTOM_PROMPT"
        "--args.record-dir" "$RECORD_DIR"
        "--args.inference-rate" "$INFERENCE_RATE"
        "--args.show-camera"
    )
    
    # 添加测试模式标志（如果有）
    if [ -n "$TEST_MODE_FLAG" ]; then
        CMD_ARGS+=("$TEST_MODE_FLAG")
    fi
    
    # 运行推理脚本
    uv run python3 examples/libero/realtime_inference.py "${CMD_ARGS[@]}"

else
    echo "用法: $0 [server|client] [--test-mode] [prompt]"
    echo ""
    echo "示例:"
    echo "  终端1: $0 server"
    echo "  终端2: $0 client"
    echo "  终端2 (测试模式): $0 client --test-mode"
    echo "  终端2 (自定义prompt): $0 client \"Pick up the blue square and place it in the blue tray.\""
    exit 1
fi

