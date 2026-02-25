#!/usr/bin/env python3
"""
从 ROS2 bag 文件中提取视频

用法:
    python3 extract_video_from_rosbag.py <rosbag_directory> [--topic <topic_name>] [--output <output_video>]

示例:
    python3 extract_video_from_rosbag.py pick_blue_bottle/rosbag2_2026_01_09-21_24_48 --topic /camera/camera/color/image_raw
"""

import argparse
import pathlib
import sys
from typing import Optional

try:
    import cv2
    import numpy as np
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    import sqlite3
    CV2_AVAILABLE = True
except ImportError as e:
    CV2_AVAILABLE = False
    print(f"❌ 缺少必要的依赖: {e}")
    print("请安装: pip install opencv-python rclpy")
    sys.exit(1)


def extract_images_from_bag(
    bag_dir: pathlib.Path,
    topic: str = "/camera/camera/color/image_raw",
    output_dir: Optional[pathlib.Path] = None
) -> list[np.ndarray]:
    """从 ROS2 bag 中提取图像"""
    images = []
    
    # 查找 .db3 文件
    db3_files = list(bag_dir.glob("*.db3"))
    if not db3_files:
        raise ValueError(f"在 {bag_dir} 中找不到 .db3 文件")
    
    db3_path = db3_files[0]
    print(f"📦 读取 ROS2 bag: {db3_path}")
    
    # 连接 SQLite 数据库
    conn = sqlite3.connect(str(db3_path))
    cursor = conn.cursor()
    
    # 查询消息
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
        # 列出所有可用的话题
        cursor.execute("SELECT name FROM topics")
        topics = [row[0] for row in cursor.fetchall()]
        print(f"可用的话题: {', '.join(topics)}")
        conn.close()
        return []
    
    print(f"📸 找到 {len(rows)} 条图像消息")
    
    # 获取消息类型
    try:
        Image = get_message("sensor_msgs/msg/Image")
    except Exception as e:
        print(f"❌ 无法加载 sensor_msgs/msg/Image: {e}")
        conn.close()
        return []
    
    # 解码图像
    for i, (timestamp, data, msg_id) in enumerate(rows):
        try:
            # 反序列化消息
            msg = deserialize_message(data, Image)
            
            # 从 ROS Image 消息中提取图像数据
            # 根据编码格式解码
            encoding = msg.encoding
            
            if encoding == "rgb8":
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            elif encoding == "bgr8":
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif encoding == "mono8":
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif encoding == "16UC1":  # 深度图像
                img = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
                # 归一化到 0-255
                img = (img / img.max() * 255).astype(np.uint8) if img.max() > 0 else img.astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                # 尝试作为 JPEG 解码
                try:
                    nparr = np.frombuffer(msg.data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    else:
                        print(f"⚠️  无法解码图像 {i}，编码: {encoding}")
                        continue
                except:
                    print(f"⚠️  无法解码图像 {i}，编码: {encoding}")
                    continue
            
            images.append(img)
            
            if (i + 1) % 10 == 0:
                print(f"  已解码 {i + 1}/{len(rows)} 张图像...")
                
        except Exception as e:
            print(f"⚠️  解码图像 {i} 失败: {e}")
            continue
    
    conn.close()
    print(f"✅ 成功提取 {len(images)} 张图像")
    
    # 保存图像到目录（可选）
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(images):
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_dir / f"frame_{i:05d}.jpg"), img_bgr)
        print(f"📁 图像已保存到: {output_dir}")
    
    return images


def images_to_video(
    images: list[np.ndarray],
    output_path: pathlib.Path,
    fps: int = 30
) -> None:
    """将图像序列转换为视频"""
    if not images:
        print("❌ 没有图像可以转换为视频")
        return
    
    h, w = images[0].shape[:2]
    print(f"🎬 创建视频: {w}x{h}, {len(images)} 帧, {fps} FPS")
    
    # 尝试使用 imageio（更兼容）
    try:
        import imageio
        print("   使用 imageio 生成视频（更兼容的格式）...")
        
        # 转换为 uint8 并确保是 RGB 格式
        video_images = []
        for img in images:
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            if img.shape[2] == 3:
                video_images.append(img)
        
        imageio.mimwrite(str(output_path), video_images, fps=fps, codec='libx264', quality=8)
        print(f"✅ 视频已保存: {output_path}")
        return
    except ImportError:
        print("   imageio 不可用，使用 OpenCV...")
    except Exception as e:
        print(f"   imageio 失败: {e}，尝试 OpenCV...")
    
    # 回退到 OpenCV，尝试多种编码器
    codecs = [
        ('avc1', 'H.264/AVC1'),
        ('XVID', 'XVID'),
        ('mp4v', 'MPEG-4'),
        ('X264', 'x264'),
    ]
    
    for codec_name, codec_desc in codecs:
        try:
            print(f"   尝试使用 {codec_desc} ({codec_name}) 编码...")
            fourcc = cv2.VideoWriter_fourcc(*codec_name)
            temp_path = output_path.with_suffix(f'.temp{output_path.suffix}')
            out = cv2.VideoWriter(str(temp_path), fourcc, fps, (w, h))
            
            if not out.isOpened():
                print(f"   ❌ {codec_desc} 编码器无法打开，尝试下一个...")
                continue
            
            for i, img in enumerate(images):
                # 转换为 BGR（OpenCV 需要）
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                out.write(img_bgr)
                
                if (i + 1) % 10 == 0:
                    print(f"  写入帧 {i + 1}/{len(images)}...")
            
            out.release()
            
            # 检查文件是否创建成功
            if temp_path.exists() and temp_path.stat().st_size > 0:
                temp_path.replace(output_path)
                print(f"✅ 视频已保存: {output_path} (使用 {codec_desc})")
                return
            else:
                print(f"   ❌ {codec_desc} 编码失败，尝试下一个...")
        except Exception as e:
            print(f"   ❌ {codec_desc} 编码出错: {e}，尝试下一个...")
            continue
    
    raise RuntimeError("所有编码器都失败了，无法创建视频")


def main():
    parser = argparse.ArgumentParser(description="从 ROS2 bag 提取视频")
    parser.add_argument("bag_dir", type=pathlib.Path, help="ROS2 bag 目录路径")
    parser.add_argument(
        "--topic",
        type=str,
        default="/camera/camera/color/image_raw",
        help="要提取的图像话题 (默认: /camera/camera/color/image_raw)"
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="输出视频文件路径 (默认: <bag_dir>/video.mp4)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="视频帧率 (默认: 30)"
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="同时保存单独的图像帧"
    )
    
    args = parser.parse_args()
    
    if not args.bag_dir.exists():
        print(f"❌ 目录不存在: {args.bag_dir}")
        sys.exit(1)
    
    # 设置输出路径
    if args.output is None:
        args.output = args.bag_dir / "video.mp4"
    
    # 提取图像
    images = extract_images_from_bag(
        args.bag_dir,
        topic=args.topic,
        output_dir=args.bag_dir / "frames" if args.save_frames else None
    )
    
    if not images:
        print("❌ 没有提取到图像")
        sys.exit(1)
    
    # 转换为视频
    images_to_video(images, args.output, fps=args.fps)
    
    print(f"\n✅ 完成! 视频文件: {args.output}")


if __name__ == "__main__":
    main()

