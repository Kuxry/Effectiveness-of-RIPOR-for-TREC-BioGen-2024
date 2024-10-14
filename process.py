import json
import os
from tqdm import tqdm  # 导入 tqdm 模块

# 定义输入文件和输出文件路径
input_file = "pqal.json"  # 替换为你的json文件路径
output_file = "raw.tsv"  # 输出的tsv文件路径

# 打开并读取JSON文件
with open(input_file, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# 打开输出文件以写入模式
with open(output_file, 'w', encoding='utf-8') as out_file:


    # 遍历JSON文件中的数据
    for id_, content in tqdm(data.items(), desc="Processing document"):
        # 提取ID和QUESTION字段
        question = content.get("QUESTION", "")

        # 写入TSV文件
        out_file.write(f"{id_}\t{question}\n")

print(f"Data has been saved to {output_file}")
