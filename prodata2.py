import mmap
import pandas as pd
from rank_bm25 import BM25Okapi
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
import gc
import psutil


def process_queries(queries_chunk, bm25, doc_ids, output_file, progress_queue, save_batch_size=10):
    results = []
    batch_count = 0

    # 查询循环
    for _, query_row in queries_chunk.iterrows():
        qid = query_row['qid']
        query = query_row['query'].split(" ")

        doc_scores = bm25.get_scores(query)
        sorted_docs_scores = sorted(zip(doc_ids, doc_scores), key=lambda x: x[1], reverse=True)[:20]

        if sorted_docs_scores:
            sorted_docids, sorted_scores = zip(*sorted_docs_scores)
            result = {
                "qid": str(qid),
                "docids": list(sorted_docids),
                "scores": list(sorted_scores)
            }
            results.append(result)

        progress_queue.put(1)  # 更新进度条

        batch_count += 1
        if batch_count % save_batch_size == 0:  # 每处理10个查询保存一次
            save_results(results, output_file)
            results.clear()  # 清空已保存的结果

    # 保存剩余结果
    if results:
        save_results(results, output_file)
        results.clear()

    # 清理查询相关的内存
    del queries_chunk
    gc.collect()


def save_results(results, output_file):
    with open(output_file, 'a') as f:
        for result in results:
            json.dump(result, f)
            f.write("\n")


def update_progress_bar(progress_queue, total_queries):
    with tqdm(total=total_queries, desc="Processing queries") as pbar:
        while True:
            item = progress_queue.get()
            if item is None:
                break
            pbar.update(item)


def check_memory(threshold=70):
    """检查系统内存使用情况，如果超过阈值，则返回True，表示需要清理内存。"""
    mem = psutil.virtual_memory()
    return mem.percent > threshold


def read_document_chunk(file_path, chunk_size=512):
    """逐块读取大文件的文档内容"""
    with open(file_path, 'r+b') as f:
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            offset = 0
            while offset < mm.size():
                # 读取指定大小的区块
                chunk = mm[offset:offset + chunk_size]
                offset += chunk_size
                yield chunk.decode('utf-8', errors='ignore')

def read_document_chunk_with_buffer(file_path, chunk_size=512):
    """逐块读取大文件的文档内容，处理跨块的行"""
    with open(file_path, 'r+b') as f:
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            offset = 0
            buffer = ''  # 用于暂存不完整的行

            while offset < mm.size():
                chunk = mm[offset:offset + chunk_size]
                offset += chunk_size

                # 将当前块转换为字符串并拼接到缓冲区中
                buffer += chunk.decode('utf-8', errors='ignore')

                # 使用 splitlines() 来确保只处理完整的行
                lines = buffer.splitlines(keepends=False)

                # 暂存最后一行，因为它可能是不完整的
                buffer = lines.pop() if lines else ''

                # 返回当前块中所有完整的行
                for line in lines:
                    yield line

            # 如果缓冲区中还有残留的不完整行，也返回它
            if buffer:
                yield buffer

def main():
    queries_df = pd.read_csv('data2/train_queries/raw.tsv', sep='\t', header=None, names=['qid', 'query'])
    output_file = 'data2/big_train_score_sample_new_test.json'
    open(output_file, 'w').close()  # 清空输出文件内容

    docs_file = 'data2/full_collection/big_original_train.tsv'

    doc_ids = []
    tokenized_docs = []
    chunk_count = 0
    valid_line_count = 0  # 有效行计数
    invalid_line_count = 0  # 无效行计数
    buffer = ""  # 用于暂存单独的标点符号行

    previous_line = ""  # 用于存储上一行，方便合并

    # 逐块读取文档并进行处理
    for line in read_document_chunk_with_buffer(docs_file):
        if line.strip():  # 忽略空行
            line = line.replace('\u200b', '')  # 清理可能的不可见字符，如零宽空格

            # 如果上一行没有制表符，且当前行没有制表符，合并它们
            if previous_line and '\t' not in previous_line and len(previous_line.strip()) > 1:
                line = previous_line.strip() + " " + line.strip()
                previous_line = ""  # 清空缓冲区

            # 如果当前行依旧没有制表符，暂存起来，等待下一行
            if '\t' not in line:
                previous_line = line
                continue  # 跳过，等待下一行进行合并

            # 如果是单个标点符号
            if len(line.strip()) == 1 and line.strip() in '.,?!:;':
                buffer += line.strip()  # 存储在缓冲区中，等待下一个行
                continue
            else:
                if buffer:  # 如果缓冲区有内容，将其与当前行合并
                    line = buffer + " " + line
                    buffer = ""  # 清空缓冲区

                # 分割并处理有效行
                parts = line.split('\t', 1)  # 只分割一次
                if len(parts) == 2:
                    doc_id, content = parts
                    doc_ids.append(doc_id)
                    tokenized_docs.append(content.split(" "))
                    valid_line_count += 1  # 增加有效行计数
                else:
                    # 如果当前行格式不正确，将它存储为 previous_line，以备下一次合并
                    print(f"Skipping malformed line: {repr(line)}")
                    previous_line = line  # 将该行暂存，等待下一行
                    invalid_line_count += 1  # 增加无效行计数

        # 每处理一定量的文档，进行垃圾回收和内存监控
        chunk_count += 1
        if chunk_count % 1000 == 0:  # 每1000个文档监控一次
            gc.collect()
            if check_memory():
                print("内存使用过高，清理中...")
                doc_ids.clear()  # 清理已处理的文档ID
                tokenized_docs.clear()  # 清理已处理的文档内容
                gc.collect()

    # 输出有效和无效行的统计
    print(f"Total valid lines: {valid_line_count}")
    print(f"Total invalid lines: {invalid_line_count}")

    # 初始化BM25模型
    bm25 = BM25Okapi(tokenized_docs)

    query_batch_size = 50  # 每批次处理50个查询
    total_queries = len(queries_df)

    manager = Manager()
    progress_queue = manager.Queue()

    # 使用一半的CPU核心进行并行处理
    pool = Pool(cpu_count())

    # 启动进度条更新进程
    progress_updater = pool.apply_async(update_progress_bar, args=(progress_queue, total_queries))

    # 分块处理查询，每个子进程处理一块
    for i in range(0, len(queries_df), query_batch_size):
        queries_chunk = queries_df.iloc[i:i + query_batch_size]
        pool.apply_async(process_queries, args=(queries_chunk, bm25, doc_ids, output_file, progress_queue))

        # 检查内存使用情况
        if check_memory():
            gc.collect()

    pool.close()
    pool.join()

    progress_queue.put(None)  # 完成标志
    progress_updater.get()


if __name__ == "__main__":
    main()
