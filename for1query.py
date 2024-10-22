import pandas as pd

# 读取原始的 queries 文件
input_file = "generated_queries.tsv"
df = pd.read_csv(input_file, sep='\t', header=None, names=["docid", "query"])

# 定义一个函数，只提取 query 列中的第一行
def extract_first_line(query):
    # 使用换行符分割内容，并提取第一行
    return query.split('\n')[0]

# 应用函数到 query 列
df['query'] = df['query'].apply(extract_first_line)

# 移除重复的 docid，保留第一个出现的记录
df = df.drop_duplicates(subset="docid", keep="first")

# 保存结果到新的 TSV 文件
output_file = "first_queries.tsv"
df.to_csv(output_file, sep='\t', index=False, header=False)

print(f"First line of each query for each docid saved to {output_file}")
