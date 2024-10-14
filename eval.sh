#!/bin/bash

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES='0,1'

# 设置PYTHONPATH，确保可以找到t5_pretrainer模块
export PYTHONPATH=/home/iiserver33/Workbench/likai/RIPOR/tmp/pycharm_project_697:$PYTHONPATH

# 阶段1: 数据集路径设置
# 1.1 原始文档集路径
collection_path="data/msmarco-full/full_collection"
# 1.2 查询集路径（包括多个查询文件）
q_collection_paths='["./data/msmarco-full/TREC_DL_2019/queries_2019/","./data/msmarco-full/TREC_DL_2020/queries_2020/","./data/msmarco-full/dev_queries/"]'

# 任务名称设置
task="t5seq_aq_retrieve_docids"

# 2. 文档ID到SMTID的映射路径
docid_to_smtid_path="experiments-full-t5seq-aq/t5_docid_gen_encoder_1/aq_smtid/docid_to_smtid.json"

# 3.1 最大token数量
max_new_token=32

# 3.3 预训练模型路径
pretrained_path="experiments-full-t5seq-aq/t5seq_aq_encoder_seq2seq_1_lng_knp_self_mnt_32_dcy_2/checkpoint"
# 3.4 输出目录路径
out_dir="experiments-full-t5seq-aq/t5seq_aq_encoder_seq2seq_1_lng_knp_self_mnt_32_dcy_2/out_test_top100"

# 额外的评估任务路径（包括多个评估文件）
eval_qrel_path='["./data/msmarco-full/dev_qrel.json","./data/msmarco-full/TREC_DL_2019/qrel.json","./data/msmarco-full/TREC_DL_2019/qrel_binary.json","./data/msmarco-full/TREC_DL_2020/qrel.json","./data/msmarco-full/TREC_DL_2020/qrel_binary.json"]'

# 检查输出目录是否存在，如果不存在则创建
if [ ! -d "$out_dir" ]; then
  mkdir -p "$out_dir"
fi

# 记录开始时间
echo "Script started at $(date)"

# 第一步: 生成 list_smtid_to_nextids
python -m t5_pretrainer.aq_preprocess.build_list_smtid_to_nextids \
    --docid_to_smtid_path=${docid_to_smtid_path}

if [ $? -ne 0 ]; then
  echo "Error in build_list_smtid_to_nextids. Exiting."
  exit 1
fi

# 第二步: 执行检索，使用torch.distributed.launch启动分布式任务
python -m torch.distributed.launch --nproc_per_node=2 --use_env -m t5_pretrainer.evaluate \
    --pretrained_path=${pretrained_path} \
    --out_dir=${out_dir} \
    --task=${task} \
    --docid_to_smtid_path=${docid_to_smtid_path} \
    --q_collection_paths=${q_collection_paths} \
    --batch_size=2 \
    --max_new_token_for_docid=${max_new_token} \
    --topk=100

if [ $? -ne 0 ]; then
  echo "Error in evaluate step. Exiting."
  exit 1
fi

# 第三步: 执行额外的评估任务，同样使用torch.distributed.launch
python -m torch.distributed.launch --nproc_per_node=2 --use_env -m t5_pretrainer.evaluate \
    --task=t5seq_aq_retrieve_docids_2 \
    --out_dir=${out_dir} \
    --q_collection_paths=${q_collection_paths} \
    --eval_qrel_path=${eval_qrel_path}

if [ $? -ne 0 ]; then
  echo "Error in additional evaluate step. Exiting."
  exit 1
fi

# 记录结束时间
echo "Script finished at $(date)"
