import json
import numpy as np
import pandas as pd

# 读取 run.json 文件
with open("../run.json", "r") as file:
    results = json.load(file)

# 存储每个查询的统计信息
summary_stats = []
ranked_results = {}

# 遍历所有查询并计算每个查询的统计信息
for qid, docs in results.items():
    scores = list(docs.values())

    # 计算基本统计数据
    max_score = max(scores)
    min_score = min(scores)
    avg_score = np.mean(scores)
    median_score = np.median(scores)
    quantiles = np.percentile(scores, [25, 50, 75])  # 25%, 50%, 75% 四分位数

    # 保存统计结果到列表中
    summary_stats.append({
        "Query ID": qid,
        "Max Score": max_score,
        "Min Score": min_score,
        "Average Score": avg_score,
        "Median Score": median_score,
        "25th Percentile": quantiles[0],
        "50th Percentile": quantiles[1],
        "75th Percentile": quantiles[2],
    })

    # 排序文档并计算相对比率
    sorted_docs = sorted(docs.items(), key=lambda x: x[1], reverse=True)
    relative_ratios = [score / max_score for _, score in sorted_docs]

    # 存储排名结果和相对比率
    ranked_results[qid] = {
        "ranked_docs": sorted_docs,
        "relative_ratios": relative_ratios
    }

# 将统计信息和排名结果保存到 JSON 文件
with open("summary_stats.json", "w") as summary_file:
    json.dump(summary_stats, summary_file, indent=4)

with open("ranked_results.json", "w") as ranked_file:
    json.dump(ranked_results, ranked_file, indent=4)

# 输出确认信息
print("统计结果已保存到 summary_stats.json")
print("排名结果已保存到 ranked_results.json")
