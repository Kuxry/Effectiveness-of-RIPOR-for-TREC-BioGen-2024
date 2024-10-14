import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd
import json
from tqdm import tqdm

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def generate_queries(rank, world_size, data, num_queries, batch_size):
    setup(rank, world_size)

    # 设置当前进程的设备
    device = torch.device(f'cuda:{rank}')
    tokenizer = T5Tokenizer.from_pretrained('castorini/doc2query-t5-base-msmarco')
    model = T5ForConditionalGeneration.from_pretrained('castorini/doc2query-t5-base-msmarco').to(device)

    # 使用 DistributedDataParallel 包装模型
    model = DDP(model, device_ids=[rank])

    output_data = []
    for i in tqdm(range(0, len(data), batch_size), desc=f"Generating queries on rank {rank}"):
        batch = data.iloc[i:i + batch_size]

        # 清洗数据：确保 input_texts 是有效的字符串
        input_texts = batch[1].astype(str).tolist()  # 将所有内容转换为字符串
        input_texts = [text for text in input_texts if text.strip()]  # 移除空白或无效字符串

        if not input_texts:  # 如果没有有效数据，跳过当前批次
            continue

        input_ids = tokenizer(input_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).input_ids.to(device)

        outputs = model.module.generate(
            input_ids=input_ids,
            max_length=64,
            num_return_sequences=num_queries,
            num_beams=num_queries,
            early_stopping=True
        )

        for j in range(len(batch)):
            docid = batch.iloc[j, 0]
            queries = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs[j * num_queries:(j + 1) * num_queries]]
            for query in queries:
                output_data.append({"docid": str(docid), "query": query})

    # 保存结果
    output_file_path = f'doc2query_generated_queries_full_rank_{rank}.json'
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for entry in output_data:
            json_str = json.dumps(entry, separators=(',', ':'))
            f.write(json_str + '\n')

    cleanup()

def main():
    world_size = torch.cuda.device_count()
    data = pd.read_csv('doc2query.tsv', sep='\t', header=None)

    # 随机抽取 50% 的数据
    data = data.sample(frac=0.03, random_state=42).reset_index(drop=True)

    num_queries = 10
    batch_size = 32

    rank = int(os.environ['LOCAL_RANK'])  # 由 torchrun 传递的 rank
    generate_queries(rank, world_size, data, num_queries, batch_size)

if __name__ == "__main__":
    main()
