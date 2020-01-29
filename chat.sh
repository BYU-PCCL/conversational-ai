#!/bin/sh

project_dir=$(git rev-parse --show-toplevel 2> /dev/null) 

if [ -n "$project_dir" ]; then
    py=$(find $project_dir -executable -wholename '*bin/python3*' | head -n 1)
fi

chkpt_dir=${1:-/mnt/pccfs/not_backed_up/will/checkpoints}
chkpt_dir=${chkpt_dir:-.}

$py run_gpt2.py \
    --model_name_or_path="$chkpt_dir/$(ls -1r $chkpt_dir | grep -m 1 conversational-ai)" \
    --length=20

