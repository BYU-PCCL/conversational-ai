#!/bin/sh

name=conversational-ai

work_dir=/workspace

local_root_dir=/mnt/pccfs/not_backed_up/will

chkpt_dir=${1:-$local_root_dir/checkpoints/}
mkdir -p $chkpt_dir

tb_runs_dir=${2:-$local_root_dir/runs/}
mkdir -p $tb_runs_dir

docker pull pccl/$name

# get the ID of the last GPU
gpu_id=$(nvidia-smi -L | awk 'END { gsub(":", ""); print $2 }')

args="-d --rm --name=$name -v $chkpt_dir/:$work_dir/output/ -v $tb_runs_dir/:$work_dir/runs"

if [ -x "$(command -v nvidia-docker)" ]; then
    NV_GPU=$gpu_id nvidia-docker run $args pccl/$name
else
    docker run --gpus=$gpu_id $args pccl/$name
fi

