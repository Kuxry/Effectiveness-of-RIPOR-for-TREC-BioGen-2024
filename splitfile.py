import os

# 定义输入的TSV文件和输出文件路径
input_tsv = "paper_train.tsv"
abstracts_output_file = "abstracts.tsv"
titles_output_file = "titles.tsv"

# 打开输入和输出文件
with open(input_tsv, 'r', encoding='utf-8') as tsv_file, \
        open(abstracts_output_file, 'w', encoding='utf-8') as abstracts_out_file, \
        open(titles_output_file, 'w', encoding='utf-8') as titles_out_file:
    # 逐行读取 TSV 文件
    for line in tsv_file:
        # 分割TSV文件中的每一行
        parts = line.strip().split('\t')

        # 检查是否有足够的字段
        if len(parts) < 3:
            print(f"Skipping incomplete line: {line.strip()}")
            continue

        pubmed_id, abstract, title = parts

        # 将 PubMedId 和 abstract 写入到 abstracts_output_file 中
        abstracts_out_file.write(f"{pubmed_id}\t{abstract}\n")

        # 将 PubMedId 和 title 写入到 titles_output_file 中
        titles_out_file.write(f"{pubmed_id}\t{title}\n")

print("Abstracts and Titles TSV files generated successfully as 'abstracts.tsv' and 'titles.tsv'.")
