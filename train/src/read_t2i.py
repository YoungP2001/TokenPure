import os
import json


def process_all_subfolders(root_dir, output_file):
    """
    处理根目录下所有子文件夹中的图片和JSON文件，生成指定格式的JSON数据

    参数:
        root_dir: 根文件夹路径
        output_file: 输出JSON文件路径
    """
    # 获取根目录下的所有子文件夹
    subfolders = [f for f in os.listdir(root_dir)
                  if os.path.isdir(os.path.join(root_dir, f))]

    if not subfolders:
        print("错误: 根目录下没有找到子文件夹")
        return

    # 存储所有处理结果
    results = []

    # 遍历每个子文件夹
    for subfolder in subfolders:
        subfolder_path = os.path.join(root_dir, subfolder)
        print(f"正在处理子文件夹: {subfolder_path}")

        # 查找当前子文件夹中的所有JSON文件
        json_files = []
        for file in os.listdir(subfolder_path):
            if file.endswith('.json'):
                json_files.append(file)

        # 处理每个JSON文件
        for json_filename in json_files:
            try:
                # 构建文件路径
                json_path = os.path.join(subfolder_path, json_filename)
                # 获取对应的图片路径（替换扩展名）
                img_filename = os.path.splitext(json_filename)[0] + '.jpg'
                img_path = os.path.join(subfolder_path, img_filename)

                # 检查图片文件是否存在
                if not os.path.exists(img_path):
                    print(f"警告: 图片文件不存在 - {img_path}，跳过此JSON文件")
                    continue

                # 读取JSON文件并提取prompt
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)

                # 确保prompt字段存在
                if 'prompt' not in json_data:
                    print(f"警告: JSON文件中没有找到prompt字段 - {json_path}，跳过此文件")
                    continue

                # 获取相对路径（相对于根目录）
                relative_img_path = os.path.relpath(img_path, root_dir)
                # 构建../../形式的相对路径（根据需要调整层级）
                # 这里假设输出文件将保存在root_dir的上两级目录
                # 如果实际情况不同，可以修改这里的相对路径计算方式
                source_target_path = os.path.join("/opt/liblibai-models/user-workspace2/datasets/ipadapter/", relative_img_path)

                # 创建结果条目
                result = {
                    "source": source_target_path,
                    "caption": json_data['prompt'],
                    "target": source_target_path
                }

                results.append(result)
                # 实时写入文件，边处理边保存
                with open(output_file, 'a', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False)
                    f.write('\n')

            except Exception as e:
                print(f"处理 {json_path} 时出错: {str(e)}")
                continue

        print(f"子文件夹 {subfolder} 处理完成\n")

    print(f"所有子文件夹处理完成，共生成 {len(results)} 条记录")
    print(f"结果已保存到: {output_file}")


if __name__ == "__main__":
    # 根文件夹路径
    root_directory = "/opt/liblibai-models/user-workspace2/datasets/ipadapter/"
    # 输出文件路径
    output_filename = "/opt/liblibai-models/user-workspace2/users/yp/easycontrol/train/examples/flux_ipadapter2.json"

    # 检查目录是否存在
    if not os.path.isdir(root_directory):
        print(f"错误: 目录 '{root_directory}' 不存在，请检查路径是否正确。")
    else:
        # 先清空输出文件（如果存在）
        with open(output_filename, 'w', encoding='utf-8') as f:
            pass

        process_all_subfolders(root_directory, output_filename)
