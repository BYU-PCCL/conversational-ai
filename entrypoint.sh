#!/bin/sh

model="${1:-gpt2}"

python run_lm_finetuning.py \
    --output_dir=/tmp/output \
    --model_type=$model \
    --model_name_or_path=$model \
    --do_train \
    --train_data_file=train.txt \
    --line_by_line \
    --save_steps=1000 \
    --save_total_limit=25
