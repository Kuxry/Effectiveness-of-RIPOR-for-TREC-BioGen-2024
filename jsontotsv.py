import json
import os
import random
from tqdm import tqdm  # 导入 tqdm 模块

# 定义输入和输出目录
input_dir = "../data01"  # 替换为你的json文件存放路径
train_output_file = "big_original_train.tsv"  # 训练集输出的tsv文件路径

total_files = 1219  # 总文件数量
sampled_files = int(total_files * 0.3)  # 抽取60%的文件数量

# 随机抽取60%的文件
random.seed(42)  # 设置随机种子，保证每次抽样结果一致
sampled_indices = random.sample(range(1, total_files + 1), sampled_files)

# 处理训练集文件
with open(train_output_file, 'w', encoding='utf-8') as train_out_file:


    # 使用 tqdm 创建一个进度条，遍历抽取的文件
    for i in tqdm(sampled_indices, desc="Processing training files"):
        filename = f"pubmed24n{i:04d}.json"
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
                    #abstract = data.get("Title", "")

                    # 如果Abstract是None，则跳过当前记录
                    if abstract is None:
                        continue

                    # 清除abstract中的换行符
                    abstract = abstract.replace('\n', ' ').replace('\r', '')

                    # 写入TSV文件
                    train_out_file.write(f"{pubmed_id}\t{abstract}\n")

print("Training data saved to", train_output_file)
