#!/bin/bash


data_root_dir=./data2
collection_path=$data_root_dir/full_collection/
queries_path=./data2/train_queries


# model dir
experiment_dir=experiments-full-t5seq-aq
pretrained_path=t5-base

# train_examples path

#teacher_score_path=./data/msmarco-full/bm25_run/qrel_added_qid_docids_teacher_scores.train.json
teacher_score_path=./data2/train_score_sample.json
#run_name=t5_docid_gen_encoder_0
run_name=t5_docid_gen_encoder_new1
output_dir="./$experiment_dir/"


#--nproc_per_node=8
#python -m torch.distributed.launch
torchrun --nproc_per_node=2 -m t5_pretrainer.main \
        --epochs=100 \
        --run_name=$run_name \
        --learning_rate=1e-4 \
        --loss_type=t5seq_pretrain_margin_mse \
        --model_name_or_path=t5-base \
        --model_type=t5_docid_gen_encoder \
        --teacher_score_path=$teacher_score_path \
        --output_dir=$output_dir \
        --task_names='["rank"]' \
        --wandb_project_name=full_t5seq_encoder \
        --use_fp16 \
        --collection_path=$collection_path \
        --max_length=128 \
        --per_device_train_batch_size=128 \
        --queries_path=$queries_path \
        --pretrained_path=$pretrained_path