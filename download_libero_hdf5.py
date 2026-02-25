#!/usr/bin/env python3
"""
下载Libero HDF5数据集的脚本

如果自动下载失败，可以：
1. 使用VPN后重新运行此脚本
2. 手动从以下链接下载zip文件，然后解压到指定目录
3. 使用其他下载工具（如aria2c, curl等）
"""

import sys
import os
sys.path.append('/home/wyz/openpi/third_party/libero')

from libero.libero import get_libero_path
import libero.libero.utils.download_utils as download_utils

# 数据集下载链接
DATASET_LINKS = {
    "libero_object": "https://utexas.box.com/shared/static/avkklgeq0e1dgzxz52x488whpu8mgspk.zip",
    "libero_goal": "https://utexas.box.com/shared/static/iv5e4dos8yy2b212pkzkpxu9wbdgjfeg.zip",
    "libero_spatial": "https://utexas.box.com/shared/static/04k94hyizn4huhbv5sz4ev9p2h1p6s7f.zip",
    "libero_100": "https://utexas.box.com/shared/static/cv73j8zschq8auh9npzt876fdc1akvmk.zip",
}

def check_existing_files(download_dir):
    """检查已存在的HDF5文件"""
    import glob
    from pathlib import Path
    
    print("\n检查已存在的HDF5文件...")
    hdf5_files = []
    for dataset_name in ["libero_spatial", "libero_object", "libero_goal", "libero_100"]:
        dataset_dir = os.path.join(download_dir, dataset_name)
        if os.path.exists(dataset_dir):
            files = glob.glob(os.path.join(dataset_dir, "*.hdf5"))
            if files:
                hdf5_files.extend(files)
                print(f"\n✅ {dataset_name}: 找到 {len(files)} 个HDF5文件")
                for f in sorted(files)[:3]:
                    print(f"   - {os.path.basename(f)}")
                if len(files) > 3:
                    print(f"   ... 还有 {len(files) - 3} 个文件")
    
    return hdf5_files

def main():
    # 获取数据集路径
    download_dir = get_libero_path("datasets")
    download_dir = os.path.abspath(download_dir)
    
    print("=" * 80)
    print("Libero HDF5数据集下载工具")
    print("=" * 80)
    print(f"\n下载目录: {download_dir}")
    
    # 确保目录存在
    os.makedirs(download_dir, exist_ok=True)
    
    # 检查已存在的文件
    existing_files = check_existing_files(download_dir)
    
    if existing_files:
        print(f"\n✅ 已找到 {len(existing_files)} 个HDF5文件")
        print(f"\n示例文件路径: {existing_files[0]}")
        return
    
    # 如果没有文件，尝试下载
    print("\n未找到HDF5文件，尝试下载 libero_spatial 数据集...")
    print("这将下载10个HDF5文件，大约2.88GB")
    print("\n如果下载失败，请：")
    print("1. 检查网络连接（可能需要VPN）")
    print("2. 手动下载zip文件:")
    print(f"   {DATASET_LINKS['libero_spatial']}")
    print(f"3. 解压到: {download_dir}/libero_spatial/")
    print("\n开始下载...")
    
    try:
        download_utils.libero_dataset_download(
            datasets="libero_spatial",
            download_dir=download_dir,
            check_overwrite=False
        )
        
        # 检查下载结果
        print("\n检查下载结果...")
        download_utils.check_libero_dataset(download_dir=download_dir)
        
        # 列出下载的文件
        existing_files = check_existing_files(download_dir)
        if existing_files:
            print(f"\n✅ 成功下载 {len(existing_files)} 个HDF5文件!")
            
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("\n请尝试手动下载:")
        print(f"1. 访问: {DATASET_LINKS['libero_spatial']}")
        print(f"2. 下载zip文件")
        print(f"3. 解压到: {download_dir}/libero_spatial/")
        print("\n或者使用其他下载工具:")
        print(f"wget '{DATASET_LINKS['libero_spatial']}' -O {download_dir}/libero_spatial.zip")
        print(f"unzip {download_dir}/libero_spatial.zip -d {download_dir}/")

if __name__ == "__main__":
    main()














