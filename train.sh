#!/bin/sh


project_dir=$(git rev-parse --show-toplevel 2> /dev/null) 

if [ -n "$project_dir" ]; then
    py=$(find $project_dir -executable -wholename '*bin/python3*' | head -n 1)
fi

py="${py:-python}"

model="${MODEL:-gpt2}"

apex_args=""
$py -c "import apex" 2> /dev/null && apex_args="--fp16 --fp16_opt_level=O2"

$py run_lm_finetuning.py \
    --model_type=$model \
    --model_name_or_path=$model \
    --do_train \
    --train_data_file=train.txt \
    --line_by_line \
    --num_train_epochs=100 \
    --output_dir=checkpoints/checkpoints_conversational-ai_$(date +%s) \
    --save_steps=1000 \
    --save_total_limit=3 \
    $apex_args \
    "$@"
    
