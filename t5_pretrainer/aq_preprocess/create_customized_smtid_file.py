import faiss
import numpy as np
import argparse
import os
import ujson
from joblib import Parallel, delayed
from tqdm import tqdm
import time

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="", type=str)
    parser.add_argument("--M", default=32, type=int)
    parser.add_argument("--bits", default=8, type=int)
    return parser.parse_args()

def process_batch_codes(batch, M, bits, code_length):
    result = []
    for u8_code in batch:
        bs = faiss.BitstringReader(faiss.swig_ptr(u8_code), code_length)
        result.append([bs.read(bits) for _ in range(M)])
    return result

if __name__ == "__main__":
    args = get_args()
    model_dir = args.model_dir
    M = args.M
    bits = args.bits

    print(f"model_dir: {model_dir}, codebook_num: {M}, codebook_size: {2 ** bits}")
    mmap_dir = os.path.join(model_dir, "mmap")

    # 加载 idx_to_docid
    idx_to_docid = {}
    text_ids_path = os.path.join(mmap_dir, "text_ids.tsv")
    with open(text_ids_path, 'r') as fin:
        for i, line in enumerate(tqdm(fin, desc="Loading idx_to_docid")):
            docid = line.strip()
            idx_to_docid[i] = docid
    print(f"size of idx_to_docid = {len(idx_to_docid)}")

    # 加载 doc_embeds 到内存中
    doc_embeds_path = os.path.join(mmap_dir, "doc_embeds.mmap")
    doc_embeds_size = os.path.getsize(doc_embeds_path) // 4  # float32占4字节
    doc_embeds = np.fromfile(doc_embeds_path, dtype=np.float32, count=doc_embeds_size)
    doc_embeds = doc_embeds.reshape(-1, 768)
    print(f"Loaded doc_embeds with shape: {doc_embeds.shape}")

    # 加载 FAISS 索引
    index_path = os.path.join(model_dir, "aq_index/model.index")
    index = faiss.read_index(index_path)
    rq = index.rq

    # 计算 unit8_codes，增大批处理大小
    print("Computing unit8_codes...")
    start_time = time.time()
    batch_size = 1024  # 尝试更大的批处理大小
    unit8_codes = []
    for i in tqdm(range(0, doc_embeds.shape[0], batch_size), desc="Generating unit8_codes"):
        batch = rq.compute_codes(doc_embeds[i:i+batch_size])
        unit8_codes.append(batch)
    unit8_codes = np.concatenate(unit8_codes, axis=0)
    end_time = time.time()
    print(f"Computed unit8_codes with shape: {unit8_codes.shape} in {end_time - start_time:.2f} seconds.")

    # 定义 code_length
    code_length = unit8_codes.shape[1]

    # 并行处理批量的编码
    print("Processing doc_encodings...")
    doc_encodings = []
    with tqdm(total=unit8_codes.shape[0], desc="Generating doc_encodings") as pbar:
        results = Parallel(n_jobs=-1)(
            delayed(process_batch_codes)(unit8_codes[i:i+batch_size], M, bits, code_length)
            for i in range(0, unit8_codes.shape[0], batch_size)
        )
        for result in results:
            doc_encodings.extend(result)
            pbar.update(len(result))

    # 生成 docid_to_smtid，带有进度条
    print("Creating docid_to_smtid...")
    docid_to_smtid = {}
    for idx, doc_enc in tqdm(enumerate(doc_encodings), desc="Building docid_to_smtid", total=len(doc_encodings)):
        docid = idx_to_docid[idx]
        docid_to_smtid[docid] = [-1] + doc_enc
    print(f"size of docid_to_smtid = {len(docid_to_smtid)}")

    out_dir = os.path.join(model_dir, "aq_smtid")
    os.makedirs(out_dir, exist_ok=True)

    # 增量保存 docid_to_smtid，带有进度条
    print("Saving docid_to_smtid...")
    chunk_size = 700000
    docid_to_smtid_items = list(docid_to_smtid.items())
    with open(os.path.join(out_dir, "docid_to_smtid.json"), "w") as fout:
        for i in tqdm(range(0, len(docid_to_smtid_items), chunk_size), desc="Writing docid_to_smtid"):
            chunk = dict(docid_to_smtid_items[i:i + chunk_size])
            fout.write(ujson.dumps(chunk) + '\n')

    # 生成 smtid_to_docids，带有进度条
    print("Creating smtid_to_docids...")
    smtid_to_docids = {}
    for docid, smtids in tqdm(docid_to_smtid.items(), desc="Building smtid_to_docids"):
        smtid = "_".join(map(str, smtids))
        smtid_to_docids.setdefault(smtid, []).append(docid)

    total_smtid = len(smtid_to_docids)
    lengths = np.array([len(docids) for docids in smtid_to_docids.values()])
    unique_smtid_num = np.sum(lengths == 1)
    print(f"unique_smtid_num = {unique_smtid_num}, total_smtid = {total_smtid}")
    print(f"percentage of smtid is unique = {unique_smtid_num / total_smtid:.3f}")
    quantiles = np.quantile(lengths, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    print(f"distribution of lengths: {quantiles}")

    # 保存 smtid_to_docids
    print("Saving smtid_to_docids...")
    smtid_to_docids_path = os.path.join(out_dir, "smtid_to_docids.json")
    chunk_size = 700000
    smtid_to_docids_items = list(smtid_to_docids.items())
    with open(smtid_to_docids_path, "w") as fout:
        for i in tqdm(range(0, len(smtid_to_docids_items), chunk_size), desc="Writing smtid_to_docids"):
            chunk = dict(smtid_to_docids_items[i:i + chunk_size])
            fout.write(ujson.dumps(chunk) + '\n')

    print("Processing completed successfully.")
