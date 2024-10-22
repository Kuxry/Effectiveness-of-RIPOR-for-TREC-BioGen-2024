import pandas as pd
import json

# 读取原始的 queries 文件
input_file = "generated_queries.tsv"
df = pd.read_csv(input_file, sep='\t', header=None, names=["docid", "query"])

# 将数据按行拆分并按 JSON 格式保存到新的文件
output_file = "queries.json"
with open(output_file, 'w') as f:
    for _, row in df.iterrows():
        # 将 query 按换行符分割
        queries = row["query"].split('\n')
        for query in queries:
            if query.strip():  # 确保 query 非空
                json_obj = {"docid": str(row["docid"]), "query": query.strip()}
                f.write(json.dumps(json_obj) + '\n')

print(f"JSON data with string docid saved to {output_file}")
