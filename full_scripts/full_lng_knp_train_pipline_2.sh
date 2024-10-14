#!/bin/bash
task=t5seq_aq_encoder_seq2seq
experiment_dir=experiments-full-t5seq-aq

task=t5seq_aq_encoder_margin_mse_sub_smtid
data_root_dir=./data2
collection_path=$data_root_dir/full_collection/
queries_path=./data2/train_queries

data_dir="./$experiment_dir/t5_docid_gen_encoder_1"
docid_to_smtid_path=$data_dir/aq_smtid/docid_to_smtid.json
output_dir="./$experiment_dir/"

model_dir="./$experiment_dir/t5seq_aq_encoder_seq2seq_new1"
pretrained_path=$model_dir/checkpoint

decay=2
max_new_token=16
teacher_score_path="./$model_dir/sub_smtid_16_out/qid_smtid_docids_teacher_score.train.json"
run_name=t5seq_aq_encoder_seq2seq_1_lng_knp_mnt_"$max_new_token"_dcy_"$decay"

task_names='["rank","rank_4","rank_8"]'
echo $teacher_score_path

python -m torch.distributed.launch --nproc_per_node=2 -m t5_pretrainer.main \
        --epochs=120 \
        --run_name=$run_name \
        --learning_rate=1e-4 \
        --loss_type=t5seq_aq_encoder_lng_knp_margin_mse \
        --model_name_or_path=t5-base \
        --model_type=t5_docid_gen_encoder \
        --teacher_score_path=$teacher_score_path \
        --output_dir=$output_dir \
        --task_names=$task_names \
        --wandb_project_name=full_t5seq_encoder \
        --use_fp16 \
        --collection_path=$collection_path \
        --max_length=64 \
        --per_device_train_batch_size=96 \
        --queries_path=$queries_path \
        --pretrained_path=$pretrained_path \
        --smtid_as_docid
