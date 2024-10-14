import pandas as pd
from rank_bm25 import BM25Okapi
import json
from tqdm import tqdm  # 导入tqdm库

# 读取查询文件
queries_df = pd.read_csv('data2/train_queries/raw.tsv', sep='\t', header=None, names=['qid', 'query'])

# 读取文档文件，并忽略格式不正确的行，指定内容列为字符串类型
docs_df = pd.read_csv(
    'data2/full_collection/raw.tsv',
    sep='\t',
    header=None,
    names=['docid', 'content'],
    on_bad_lines='skip',
    dtype={'docid': str, 'content': str},  # 指定数据类型
    low_memory=False  # 防止警告
)

# 填充缺失值为空字符串，确保内容列为字符串类型
docs_df['content'] = docs_df['content'].fillna('')

# 对文档内容进行分词
tokenized_docs = [doc.split(" ") for doc in docs_df['content']]

# 初始化BM25Okapi实例
bm25 = BM25Okapi(tokenized_docs)

# 准备存储输出结果的列表
results = []

# 只处理前5个查询
for _, query_row in tqdm(queries_df.iterrows(), total=len(queries_df), desc="Processing queries"):
    qid = query_row['qid']
    query = query_row['query'].split(" ")

    # 获取BM25排序后的分数
    doc_scores = bm25.get_scores(query)

    # 将文档ID和得分排序
    sorted_docs_scores = sorted(zip(docs_df['docid'], doc_scores), key=lambda x: x[1], reverse=True)[:20]

    # 拆分排序后的文档ID和得分
    if sorted_docs_scores:  # 确保有排序结果
        sorted_docids, sorted_scores = zip(*sorted_docs_scores)

        # 生成结果字典
        result = {
            "qid": str(qid),
            "docids": list(sorted_docids),
            "scores": list(sorted_scores)
        }

        results.append(result)

output_file = 'data2/big_train_score_sample.json'  # 在此处指定文件名和路径

# 写入文件时添加进度条
with open(output_file, 'w') as f:
    for result in tqdm(results, desc="Writing results to file"):
        json.dump(result, f)
        f.write("\n")  # 每个结果一行
