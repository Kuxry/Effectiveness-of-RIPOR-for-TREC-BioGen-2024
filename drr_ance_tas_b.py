import torch
import torch.multiprocessing as mp
import pandas as pd
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from multiprocessing import Queue

# 假设您已经安装并能导入ColBERT和DeepRank
# 请根据实际安装路径和模块名称进行调整
#from colbert.modeling.colbert import ColBERT
from deeprank.model import DeepRank  # 根据实际DeepRank实现调整

# 设置 multiprocessing 的启动方式为 'spawn'
mp.set_start_method('spawn', force=True)


def initialize_colbert(device):
    """
    初始化ColBERT模型和tokenizer。
    """
    colbert_model = ColBERT.from_pretrained("bert-base-uncased").to(device)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return colbert_model, tokenizer


def initialize_deeprank(device):
    """
    初始化DeepRank模型和tokenizer。
    """
    deeprank_model = DeepRank.from_pretrained("bert-base-uncased").to(device)
    deeprank_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return deeprank_model, deeprank_tokenizer


def encode_documents_colbert(documents, device, batch_size=128):
    """
    使用ColBERT对文档进行编码。
    """
    colbert_model, tokenizer = initialize_colbert(device)
    embeddings = []
    colbert_model.to(device)
    colbert_model.eval()  # 设置为评估模式
    with torch.no_grad():
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            # 设置 max_length 和 truncation
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
            # ColBERT encoding
            embedding = colbert_model(**inputs)  # 根据实际ColBERT实现调整
            embeddings.append(embedding.cpu())  # 将编码结果移回CPU
            torch.cuda.empty_cache()
    return torch.cat(embeddings)


def process_queries_colbert_deeprank(rank, queries_df, doc_embeddings, doc_ids, doc_id_to_content, output_file,
                                     progress_queue):
    """
    处理查询，计算相似度并使用DeepRank进行重排序，保存结果。
    """
    device = torch.device(f"cuda:{rank}")
    colbert_model, tokenizer = initialize_colbert(device)
    deeprank_model, deeprank_tokenizer = initialize_deeprank(device)

    # 将 doc_embeddings 移动到当前设备
    doc_embeddings_gpu = doc_embeddings.to(device)

    colbert_model.eval()
    deeprank_model.eval()

    results = []
    with torch.no_grad():
        for _, query_row in tqdm(queries_df.iterrows(), total=len(queries_df),
                                 desc=f"ColBERT + DeepRank on GPU {rank}"):
            qid = query_row['qid']
            query = query_row['query']

            # 设置 max_length 和 truncation
            query_inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=512).to(device)
            query_embedding = colbert_model(**query_inputs)  # 根据实际ColBERT实现调整

            # 计算相似度，确保所有张量都在同一个设备
            scores = torch.nn.functional.cosine_similarity(query_embedding, doc_embeddings_gpu)
            sorted_docs_scores = sorted(zip(doc_ids, scores.tolist()), key=lambda x: x[1], reverse=True)[:100]

            if sorted_docs_scores:
                # DeepRank re-ranking
                reranked = []
                for doc_id, score in sorted_docs_scores:
                    doc_content = doc_id_to_content.get(doc_id, "")
                    # 准备DeepRank输入
                    deeprank_inputs = deeprank_tokenizer.encode_plus(query, doc_content, return_tensors='pt',
                                                                     truncation=True, max_length=512).to(device)
                    deeprank_score = deeprank_model(**deeprank_inputs).item()  # 根据实际DeepRank实现调整

                    reranked.append((doc_id, deeprank_score))

                # 按照DeepRank评分重新排序
                reranked = sorted(reranked, key=lambda x: x[1], reverse=True)[:100]

                sorted_docids, sorted_scores = zip(*reranked)
                result = {
                    "qid": str(qid),
                    "docids": list(sorted_docids),
                    "scores": list(sorted_scores)
                }
                results.append(result)

            # 更新进度条
            progress_queue.put(1)

    # 保存结果
    save_results(results, output_file)


def save_results(results, output_file):
    """
    将结果保存到JSON文件中。
    """
    with open(output_file, 'w') as f:
        for result in results:
            json.dump(result, f)
            f.write("\n")


def update_progress_bar(total_queries, progress_queue):
    """
    更新进度条。
    """
    with tqdm(total=total_queries, desc="Processing queries") as pbar:
        processed = 0
        while processed < total_queries:
            processed += progress_queue.get()
            pbar.update(1)


def load_data():
    """
    加载查询和文档数据。
    """
    queries_df = pd.read_csv('data3/test_query100/raw.tsv', sep='\t', header=None, names=['qid', 'query'])
    docs_file = 'data3/full_collection_dir/raw.tsv'
    doc_ids = []
    contents = []

    with open(docs_file, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    doc_id, content = parts
                    doc_ids.append(doc_id)
                    contents.append(content)

    print(f"Total documents loaded: {len(doc_ids)}")
    # 创建doc_id到内容的映射
    doc_id_to_content = dict(zip(doc_ids, contents))
    return queries_df, doc_ids, contents, doc_id_to_content


def main():
    queries_df, doc_ids, contents, doc_id_to_content = load_data()
    output_file = 'data3/test_query100/results_ColBERT_DeepRank.json'
    open(output_file, 'w').close()

    # 将内容和查询数据分为两部分，分别由两块 GPU 处理
    contents_split = [contents[:len(contents) // 2], contents[len(contents) // 2:]]
    queries_split = [queries_df.iloc[:len(queries_df) // 2], queries_df.iloc[len(queries_df) // 2:]]

    # 使用两个GPU并行对文档进行编码
    with torch.multiprocessing.Pool(2) as pool:
        doc_embeddings_split = pool.starmap(encode_documents_colbert, [
            (contents_split[0], "cuda:0"),
            (contents_split[1], "cuda:1")
        ])

    # 合并编码后的文档到CPU
    doc_embeddings = torch.cat([emb for emb in doc_embeddings_split]).share_memory_()

    total_queries = len(queries_df)

    processes = []
    progress_queue = Queue()

    # 启动进度条进程
    progress_process = mp.Process(target=update_progress_bar, args=(total_queries, progress_queue))
    progress_process.start()

    # 在两个 GPU 上并行处理查询，并使用独立的输出文件
    for gpu_id in range(2):
        # 为每个GPU指定独立的输出文件
        output_file_gpu = f'data3/test_query100/results_ColBERT_DeepRank_gpu{gpu_id}.json'
        p = mp.Process(target=process_queries_colbert_deeprank,
                       args=(gpu_id, queries_split[gpu_id], doc_embeddings, doc_ids, doc_id_to_content, output_file_gpu,
                             progress_queue))
        p.start()
        processes.append(p)

    # 等待所有进程完成
    for p in processes:
        p.join()

    # 结束进度条进程
    progress_queue.put(total_queries)
    progress_process.join()

    # 合并所有GPU的结果文件
    combined_output_file = 'data3/test_query100/results_ColBERT_DeepRank_combined.json'
    with open(combined_output_file, 'w') as outfile:
        for gpu_id in range(2):
            output_file_gpu = f'data3/test_query100/results_ColBERT_DeepRank_gpu{gpu_id}.json'
            with open(output_file_gpu, 'r') as infile:
                for line in infile:
                    outfile.write(line)
    print(f"Combined results saved to {combined_output_file}")


if __name__ == "__main__":
    main()
