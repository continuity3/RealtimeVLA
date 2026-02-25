#!/usr/bin/env python3
"""
使用 ffmpeg 从 ROS2 bag 提取视频（更兼容的格式）

用法:
    python3 extract_video_with_ffmpeg.py <rosbag_directory> [--topic <topic_name>] [--output <output_video>]
    
需要先安装 ffmpeg: sudo apt install ffmpeg
"""

import argparse
import pathlib
import subprocess
import sys
import tempfile
import shutil

try:
    import cv2
    import numpy as np
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    import sqlite3
except ImportError as e:
    print(f"❌ 缺少必要的依赖: {e}")
    sys.exit(1)


def extract_images_from_bag(bag_dir: pathlib.Path, topic: str = "/camera/camera/color/image_raw"):
    """从 ROS2 bag 中提取图像到临时目录"""
    images = []
    
    db3_files = list(bag_dir.glob("*.db3"))
    if not db3_files:
        raise ValueError(f"在 {bag_dir} 中找不到 .db3 文件")
    
    db3_path = db3_files[0]
    print(f"📦 读取 ROS2 bag: {db3_path}")
    
    conn = sqlite3.connect(str(db3_path))
    cursor = conn.cursor()
    
    query = """
        SELECT m.timestamp, m.data, m.id
        FROM messages m
        INNER JOIN topics t ON m.topic_id = t.id
        WHERE t.name = ?
        ORDER BY m.timestamp
    """
    
    cursor.execute(query, (topic,))
    rows = cursor.fetchall()
    
    if not rows:
        print(f"⚠️  话题 {topic} 中没有找到消息")
        conn.close()
        return []
    
    print(f"📸 找到 {len(rows)} 条图像消息")
    
    try:
        Image = get_message("sensor_msgs/msg/Image")
    except Exception as e:
        print(f"❌ 无法加载 sensor_msgs/msg/Image: {e}")
        conn.close()
        return []
    
    for i, (timestamp, data, msg_id) in enumerate(rows):
        try:
            msg = deserialize_message(data, Image)
            encoding = msg.encoding
            
            if encoding == "rgb8":
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            elif encoding == "bgr8":
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif encoding == "mono8":
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif encoding == "16UC1":
                img = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
                img = (img / img.max() * 255).astype(np.uint8) if img.max() > 0 else img.astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                try:
                    nparr = np.frombuffer(msg.data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    else:
                        continue
                except:
                    continue
            
            images.append(img)
            
            if (i + 1) % 10 == 0:
                print(f"  已解码 {i + 1}/{len(rows)} 张图像...")
                
        except Exception as e:
            print(f"⚠️  解码图像 {i} 失败: {e}")
            continue
    
    conn.close()
    print(f"✅ 成功提取 {len(images)} 张图像")
    return images


def create_video_with_ffmpeg(images, output_path: pathlib.Path, fps: int = 30):
    """使用 ffmpeg 创建视频（更兼容的格式）"""
    if not images:
        print("❌ 没有图像可以转换为视频")
        return False
    
    # 检查 ffmpeg 是否可用
    if not shutil.which("ffmpeg"):
        print("❌ ffmpeg 未安装。请安装: sudo apt install ffmpeg")
        return False
    
    # 创建临时目录保存图像帧
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = pathlib.Path(tmpdir)
        print(f"💾 保存图像帧到临时目录...")
        
        for i, img in enumerate(images):
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            frame_path = tmpdir_path / f"frame_{i:05d}.jpg"
            cv2.imwrite(str(frame_path), img_bgr)
        
        print(f"🎬 使用 ffmpeg 创建视频...")
        
        # 使用 ffmpeg 创建视频
        # -y 覆盖输出文件
        # -framerate 输入帧率
        # -i 输入图像模式
        # -c:v libx264 使用 H.264 编码
        # -pix_fmt yuv420p 确保兼容性
        # -crf 23 质量设置（18-28，越小质量越高）
        cmd = [
            "ffmpeg",
            "-y",  # 覆盖输出文件
            "-framerate", str(fps),
            "-i", str(tmpdir_path / "frame_%05d.jpg"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "23",
            "-preset", "medium",
            str(output_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✅ 视频已保存: {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ ffmpeg 执行失败: {e}")
            print(f"错误输出: {e.stderr}")
            return False


def main():
    parser = argparse.ArgumentParser(description="使用 ffmpeg 从 ROS2 bag 提取视频")
    parser.add_argument("bag_dir", type=pathlib.Path, help="ROS2 bag 目录路径")
    parser.add_argument(
        "--topic",
        type=str,
        default="/camera/camera/color/image_raw",
        help="要提取的图像话题"
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="输出视频文件路径"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="视频帧率"
    )
    
    args = parser.parse_args()
    
    if not args.bag_dir.exists():
        print(f"❌ 目录不存在: {args.bag_dir}")
        sys.exit(1)
    
    if args.output is None:
        args.output = args.bag_dir / "video_color.mp4"
    
    # 提取图像
    images = extract_images_from_bag(args.bag_dir, topic=args.topic)
    
    if not images:
        print("❌ 没有提取到图像")
        sys.exit(1)
    
    # 使用 ffmpeg 创建视频
    if create_video_with_ffmpeg(images, args.output, fps=args.fps):
        print(f"\n✅ 完成! 视频文件: {args.output}")
    else:
        print("\n❌ 视频创建失败")
        sys.exit(1)


if __name__ == "__main__":
    main()



















