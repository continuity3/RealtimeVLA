# 在 Gazebo 中运行 ALOHA 仿真

这个示例展示了如何在 Gazebo 仿真环境中运行 OpenPI ALOHA 策略。

## 前置要求

1. **ROS2 环境**：需要安装 ROS2 (推荐 Humble 或 Foxy)
2. **Gazebo**：需要安装 Gazebo (推荐 Gazebo 11 或更新版本)
3. **ALOHA Gazebo 模型**：需要有一个 Gazebo 世界文件包含 ALOHA 机器人模型

## 安装依赖

```bash
# 安装 ROS2 相关包
sudo apt-get install -y \
    ros-humble-desktop \
    python3-rosdep \
    python3-colcon-common-extensions

# 安装 OpenCV bridge
sudo apt-get install -y ros-humble-cv-bridge

# 安装 openpi-client
cd /home/wyz/openpi
pip install -e packages/openpi-client
```

## 使用方法

### 1. 启动 Gazebo 仿真

首先，在终端 1 中启动 Gazebo 仿真（包含 ALOHA 机器人）：

```bash
# 启动 Gazebo（根据你的实际 Gazebo 启动命令调整）
gazebo --verbose your_aloha_world.world

# 或者使用 ros2 launch（如果有 launch 文件）
ros2 launch aloha_gazebo aloha_sim.launch.py
```

确保 Gazebo 中发布了以下话题：
- `/camera_high/image_raw` - 高角度相机图像
- `/camera_low/image_raw` - 低角度相机图像  
- `/camera_left_wrist/image_raw` - 左手腕相机图像
- `/camera_right_wrist/image_raw` - 右手腕相机图像
- `/aloha/joint_states` - 关节状态（Float64MultiArray，14 维）
- `/aloha/joint_commands` - 关节命令（Float64MultiArray，14 维）

### 2. 启动策略服务器

在终端 2 中启动策略服务器：

```bash
cd /home/wyz/openpi
uv run scripts/serve_policy.py --env ALOHA_SIM
```

### 3. 运行策略客户端

在终端 3 中运行策略客户端：

```bash
cd /home/wyz/openpi
source /opt/ros/humble/setup.bash  # 根据你的 ROS2 版本调整
python3 examples/aloha_gazebo/main.py \
    --host localhost \
    --port 8000 \
    --action-horizon 10
```

## 自定义话题名称

如果你的 Gazebo 设置使用了不同的话题名称，可以通过参数指定：

```bash
python3 examples/aloha_gazebo/main.py \
    --image-topic-high /your/camera/high/topic \
    --image-topic-low /your/camera/low/topic \
    --image-topic-left-wrist /your/left/wrist/topic \
    --image-topic-right-wrist /your/right/wrist/topic \
    --state-topic /your/joint/states/topic \
    --action-topic /your/joint/commands/topic
```

## 故障排除

### 问题：找不到图像话题

**解决方案**：检查 Gazebo 中相机是否正确配置并发布话题。可以使用以下命令查看可用话题：

```bash
ros2 topic list
ros2 topic echo /camera_high/image_raw --once
```

### 问题：状态话题格式不正确

**解决方案**：确保状态话题发布的是 `Float64MultiArray` 类型，包含 14 个浮点数（左右机械臂各 7 个：6 个关节 + 1 个夹爪）。

### 问题：动作没有执行

**解决方案**：检查动作话题是否正确订阅，以及 Gazebo 中的机器人控制器是否正确配置。

## 注意事项

1. **话题格式**：确保所有图像话题都是 `sensor_msgs/Image` 类型，状态话题是 `std_msgs/Float64MultiArray` 类型
2. **图像格式**：图像会自动转换为 RGB8 格式并调整大小为 224x224
3. **状态维度**：状态必须是 14 维（左右机械臂各 7 维）
4. **动作格式**：动作会被发布为 `Float64MultiArray`，包含 14 个浮点数

## 与真实机器人集成

如果你有真实的 ALOHA 机器人，可以参考 `examples/aloha_real` 目录中的代码，它展示了如何与真实硬件集成。

