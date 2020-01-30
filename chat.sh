#!/bin/sh

project_dir=$(git rev-parse --show-toplevel 2> /dev/null) 

if [ -n "$project_dir" ]; then
    py=$(find $project_dir -executable -wholename '*bin/python3*' | head -n 1)
fi

chkpt_dir=${CHECKPOINT_DIR:-/mnt/pccfs/not_backed_up/will/checkpoints}

$py run_gpt2.py \
    --model_name_or_path="$chkpt_dir/$(ls -1r $chkpt_dir | grep -m 1 conversational-ai)" \
    "$@"

