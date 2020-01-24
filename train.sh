#!/bin/sh

name=conversational-ai

chkpt_dir="${1:-/mnt/pccfs/not_backed_up/will/checkpoints/}"
mkdir -p "$chkpt_dir"
chkpt_dir="$(mktemp -dp $chkpt_dir --suffix=-$name)"

tb_runs_dir="${2:-/mnt/pccfs/not_backed_up/will/runs/}"
mkdir -p "$tb_runs_dir"

docker pull mwilliammyers/$name

# get the ID of the last GPU
gpu_id=$(nvidia-smi -L | awk 'END { gsub(":", ""); print $2 }')

if [ -x "$(command -v nvidia-docker)" ]; then
    NV_GPU=$gpu_id nvidia-docker run \
        -d --rm --name=$name \
        -v "$chkpt_dir/":/tmp/output/ \
        -v "$tb_runs_dir/":/workspace/runs \
        mwilliammyers/$name
else
    docker run \
        -d --rm --gpus=$gpu_id --name=$name \
        -v "$chkpt_dir/":/tmp/output/ \
        -v "$tb_runs_dir/":/workspace/runs \
        mwilliammyers/$name
fi

