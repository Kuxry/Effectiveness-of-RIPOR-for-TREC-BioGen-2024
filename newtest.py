def process_file(file_path):
    """逐行读取文件并处理格式正确的行"""
    with open(file_path, 'r') as f:
        line_count = 0
        skipped_count = 0
        for line in f:
            line_count += 1
            if line.strip():  # 忽略空行
                parts = line.split('\t')
                if len(parts) == 2:  # 只有分割为两部分的行才是有效的
                    doc_id, content = parts
                    #print(f"Processed Line {line_count}: doc_id = {doc_id}, content = {content[:50]}...")  # 只打印内容的前50个字符
                else:
                    skipped_count += 1
                    print(f"Skipping malformed line {line_count}: {repr(line)}")  # 打印跳过的行
        print(f"Total lines processed: {line_count}, Skipped lines: {skipped_count}")

# 调用函数
docs_file = 'data2/full_collection/big_original_train.tsv'
process_file(docs_file)
