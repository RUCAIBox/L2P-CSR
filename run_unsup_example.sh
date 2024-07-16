#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.

CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.launch --nproc_per_node 2 --master_port 23555 train.py \
    --model_name_or_path xxx \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir result/my-unsup-bert-base-uncased \
    --num_train_epochs 1 \
    --per_device_train_batch_size 128 \
    --learning_rate 1e-6 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model avg_sts \
    --load_best_model_at_end \
    --eval_steps 50 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --gradient_accumulation_steps 1 \
    --do_train \
    --do_eval \
    #--fp16 \
    "$@"

python simcse_to_huggingface.py --path result/my-unsup-bert-base-uncased

CUDA_VISIBLE_DEVICES=8 python evaluation.py \
        --model_name_or_path result/my-unsup-bert-base-uncased \
        --pooler cls_before_pooler \
        --task_set sts \
        --mode test
