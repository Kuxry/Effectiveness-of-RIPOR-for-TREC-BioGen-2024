import mmap
import pandas as pd
from rank_bm25 import BM25Okapi
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
import gc

def process_queries(queries_chunk, bm25, doc_ids, output_file, progress_queue):
    results = []

    for _, query_row in queries_chunk.iterrows():
        qid = query_row['qid']
        query = query_row['query'].split(" ")

        doc_scores = bm25.get_scores(query)
        sorted_docs_scores = sorted(zip(doc_ids, doc_scores), key=lambda x: x[1], reverse=True)[:100]

        if sorted_docs_scores:
            sorted_docids, sorted_scores = zip(*sorted_docs_scores)
            result = {
                "qid": str(qid),
                "docids": list(sorted_docids),
                "scores": list(sorted_scores)
            }
            results.append(result)
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
    queries_df = pd.read_csv('data3/test_query1000/raw.tsv', sep='\t', header=None, names=['qid', 'query'])
    output_file = 'data3/test_query1000/bm25_1000.json'
    open(output_file, 'w').close()

    docs_file = 'data3/full_collection_dir/raw.tsv'

    doc_ids = []
    tokenized_docs = []
    valid_line_count = 0
    buffer = ""
    previous_line = ""

    for line in read_document_chunk_with_buffer(docs_file):
        if line.strip():
            line = line.replace('\u200b', '')
            if previous_line and '\t' not in previous_line and len(previous_line.strip()) > 1:
                line = previous_line.strip() + " " + line.strip()
                previous_line = ""
            if '\t' not in line:
                previous_line = line
                continue
            if len(line.strip()) == 1 and line.strip() in '.,?!:;':
                buffer += line.strip()
                continue
            else:
                if buffer:
                    line = buffer + " " + line
                    buffer = ""
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    doc_id, content = parts
                    doc_ids.append(doc_id)
                    tokenized_docs.append(content.split(" "))
                    valid_line_count += 1
                else:
                    previous_line = line

    print(f"Total valid lines: {valid_line_count}")

    bm25 = BM25Okapi(tokenized_docs)

    manager = Manager()
    progress_queue = manager.Queue()
    total_queries = len(queries_df)

    pool = Pool(cpu_count())
    progress_updater = pool.apply_async(update_progress_bar, args=(progress_queue, total_queries))

    chunk_size = total_queries // cpu_count()
    query_chunks = [queries_df.iloc[i:i + chunk_size] for i in range(0, total_queries, chunk_size)]

    for chunk in query_chunks:
        pool.apply_async(process_queries, args=(chunk, bm25, doc_ids, output_file, progress_queue))

    pool.close()
    pool.join()

    progress_queue.put(None)
    progress_updater.get()

if __name__ == "__main__":
    main()
