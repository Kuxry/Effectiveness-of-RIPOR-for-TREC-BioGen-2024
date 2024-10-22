import pandas as pd

# 读取 raw.tsv 文件
input_file = "raw.tsv"
df = pd.read_csv(input_file, sep='\t', header=None, names=["qid", "question"])

# 截取前 200 条
top_200 = df.head(500)
top_200_file = "top500.tsv"
top_200.to_csv(top_200_file, sep='\t', index=False, header=False)
print(f"前 500 条内容已保存到 {top_200_file}")

# 截取前 100 条
top_100 = df.head(1000)
top_100_file = "top1000.tsv"
top_100.to_csv(top_100_file, sep='\t', index=False, header=False)
print(f"前 1000 条内容已保存到 {top_100_file}")
