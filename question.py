import json
import pandas as pd

# 定义输入和输出文件路径
input_json = "ori_pqau.json"  # 替换为你的 JSON 文件路径
output_tsv = "questions.tsv"

# 读取 JSON 数据
with open(input_json, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# 创建一个列表，用于存储数据
records = []

# 遍历 JSON 数据，提取 id 和 question 并存储到列表中
for id, content in data.items():
    question = content.get("QUESTION", "")
    records.append((id, question))

# 使用 pandas 创建 DataFrame
df = pd.DataFrame(records, columns=["id", "question"])

# 检查并移除重复的 id 保持唯一性，以第一个出现的为准
df_unique = df.drop_duplicates(subset=["id"], keep="first")

# 将处理后的数据写入 TSV 文件
df_unique.to_csv(output_tsv, sep='\t', index=False, header=False)

print("Questions TSV file generated successfully with unique ids as 'questions.tsv'.")
