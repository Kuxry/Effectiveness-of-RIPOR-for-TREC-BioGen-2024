import mmap
import pandas as pd
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
import gc
from whoosh import index
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import StandardAnalyzer
from whoosh.qparser import QueryParser
from whoosh.scoring import BM25F
import os

# 构建分片Whoosh索引
def create_sharded_whoosh_index(docs_file, index_dir, shard_size=10000):
    if not os.path.exists(index_dir):
        os.mkdir(index_dir)

    schema = Schema(doc_id=ID(stored=True), content=TEXT(analyzer=StandardAnalyzer()))

    shard_count = 0
    writer = None

    for i, line in enumerate(read_document_chunk_with_buffer(docs_file)):
        if i % shard_size == 0:
            if writer:
                writer.commit(optimize=True)
            shard_count += 1
            shard_dir = os.path.join(index_dir, f"shard_{shard_count}")
            if not os.path.exists(shard_dir):
                os.mkdir(shard_dir)
            ix = index.create_in(shard_dir, schema)
            writer = ix.writer()

        if line.strip():
            parts = line.split('\t', 1)
            if len(parts) == 2:
                doc_id, content = parts
                writer.add_document(doc_id=doc_id, content=content)

    if writer:
        writer.commit(optimize=True)

# 使用Whoosh分片索引执行查询并生成负样本

def process_queries_across_shards(queries_chunk, index_dir, output_file, progress_queue, debug_limit=5):
    results = []
    output_count = 0

    for _, query_row in queries_chunk.iterrows():
        qid = query_row['qid']
        query_text = query_row['query']
        shard_results = []

        # 遍历所有分片，收集每个分片的结果
        for shard_dir in os.listdir(index_dir):
            shard_path = os.path.join(index_dir, shard_dir)
            if os.path.isdir(shard_path):
                ix = index.open_dir(shard_path)

                with ix.searcher(weighting=BM25F(B=0.3, K1=1.5)) as searcher:
                    searcher.cached_doc_count = 10000
                    searcher.cached_fields = 10000

                    query = QueryParser("content", ix.schema).parse(query_text)
                    hits = searcher.search(query, limit=100)

                    if hits:
                        shard_results.extend([(hit["doc_id"], hit.score) for hit in hits])

        # 按照得分对所有分片的结果进行排序
        shard_results.sort(key=lambda x: x[1], reverse=True)
        top_docs = shard_results[:100]
        doc_ids, scores = zip(*top_docs) if top_docs else ([], [])

        result = {
            "qid": str(qid),
            "docids": list(doc_ids),
            "scores": list(scores)
        }
        results.append(result)

        # 输出前几个查询的结果
        if output_count < debug_limit:
            print(f"Debug - Query ID: {qid}, DocIDs: {doc_ids[:5]}, Scores: {scores[:5]}")
            output_count += 1

        progress_queue.put(1)

    save_results(results, output_file)
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

def read_document_chunk_with_buffer(file_path, chunk_size=1024):
    with open(file_path, 'r+b') as f:
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            offset = 0
            buffer = ''
            while offset < mm.size():
                chunk = mm[offset:offset + chunk_size]
                offset += chunk_size
                buffer += chunk.decode('utf-8', errors='ignore')
                lines = buffer.splitlines(keepends=False)
                buffer = lines.pop() if lines else ''
                for line in lines:
                    yield line
            if buffer:
                yield buffer

def main():
    queries_df = pd.read_csv('train_queries/titles.tsv', sep='\t', header=None, names=['qid', 'query'])
    output_file = 'bm25_tran_score_for_title.json'
    open(output_file, 'w').close()

    docs_file = 'full_collection/raw.tsv'
    index_dir = "whoosh_sharded_index"
    create_sharded_whoosh_index(docs_file, index_dir, shard_size=10000)

    manager = Manager()
    progress_queue = manager.Queue()
    total_queries = len(queries_df)

    pool = Pool(cpu_count())
    progress_updater = pool.apply_async(update_progress_bar, args=(progress_queue, total_queries))

    chunk_size = total_queries // cpu_count()
    query_chunks = [queries_df.iloc[i:i + chunk_size] for i in range(0, total_queries, chunk_size)]

    # 遍历每个分片索引，并行处理查询
    for shard_dir in os.listdir(index_dir):
        shard_path = os.path.join(index_dir, shard_dir)
        if os.path.isdir(shard_path):
            for chunk in query_chunks:
                pool.apply_async(process_queries_on_shard, args=(chunk, shard_path, output_file, progress_queue))

    pool.close()
    pool.join()

    progress_queue.put(None)
    progress_updater.get()

if __name__ == "__main__":
    main()
