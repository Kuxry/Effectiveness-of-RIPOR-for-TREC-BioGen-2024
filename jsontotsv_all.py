import json
import os
from tqdm import tqdm  # 导入 tqdm 模块

# 定义输入和输出目录
input_dir = "../data01"  # 替换为你的json文件存放路径
train_output_file = "big_train_all.tsv"  # 训练集输出的tsv文件路径

# 获取输入目录中的所有文件
all_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]

# 处理所有文件
with open(train_output_file, 'w', encoding='utf-8') as train_out_file:

    # 使用 tqdm 创建一个进度条，遍历所有文件
    for filename in tqdm(all_files, desc="Processing all files"):
        filepath = os.path.join(input_dir, filename)

        # 打开并读取每个JSON文件
        with open(filepath, 'r', encoding='utf-8') as json_file:
            data_list = json.load(json_file)

            # 确保JSON文件是一个列表
            if isinstance(data_list, list):
                for data in data_list:
                    # 提取PubMedId和Abstract字段
                    pubmed_id = data.get("PubMedId", "")
                    abstract = data.get("Abstract", "")
                    # abstract = data.get("Title", "")

                    # 如果Abstract是None，则跳过当前记录
                    if abstract is None:
                        continue

                    # 清除abstract中的换行符
                    abstract = abstract.replace('\n', ' ').replace('\r', '')

                    # 写入TSV文件
                    train_out_file.write(f"{pubmed_id}\t{abstract}\n")

print("Training data saved to", train_output_file)
