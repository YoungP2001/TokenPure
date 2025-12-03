from datasets import load_dataset
from PIL import Image
import io
import re
import json
import os

# 设置代理
os.environ["http_proxy"] = "http://127.0.0.1:17890"
os.environ["https_proxy"] = "http://127.0.0.1:17890"
import os
import wget
import tarfile
import shutil


def download_and_process_tar_files(num_files=10, save_dir="dataset_files"):
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 基础URL
    base_url = "https://huggingface.co/datasets/jackyhate/text-to-image-2M/resolve/main/data_512_2M/data_{i:06d}.tar"

    # 下载前10个tar文件
    for i in range(num_files):
        file_url = base_url.format(i=i)
        file_name = os.path.basename(file_url)
        file_path = os.path.join(save_dir, file_name)

        try:
            # 下载文件
            print(f"Downloading file {i + 1}/{num_files}: {file_name}")
            wget.download(file_url, out=file_path)

            # 解压tar文件
            print(f"\nExtracting {file_name}...")
            with tarfile.open(file_path, "r") as tar:
                tar.extractall(path=save_dir)

            # 删除压缩包
            os.remove(file_path)
            print(f"Deleted compressed file: {file_name}")

        except Exception as e:
            print(f"Error processing file {i}: {e}")
            continue

    print(f"\nProcess completed. Files extracted to: {os.path.abspath(save_dir)}")


if __name__ == "__main__":
    # 设置保存目录
    dataset_files="/opt/liblibai-models/user-workspace2/datasets/text2img_dataset_files"
    download_and_process_tar_files(num_files=10,save_dir=dataset_files)
