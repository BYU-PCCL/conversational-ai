#!/bin/sh

docker_image_name=conversational-ai
run_name="$docker_image_name-$(date +%s)"

work_dir=/workspace

local_root_dir=/mnt/pccfs/not_backed_up/will

chkpt_dir=${CHECKPOINT_DIR:-$local_root_dir/checkpoints/}
mkdir -p $chkpt_dir

tb_dir=${TENSORBOARD_DIR:-$local_root_dir/runs/}
mkdir -p $tb_dir

docker pull pccl/$docker_image_name

# get the ID of the last GPU
gpu_id=${NV_GPU:-$(nvidia-smi -L | awk 'END { gsub(":", ""); print $2 }')}

args="-d --rm --name=$run_name --ipc=host -e RUN_NAME=$run_name -v $chkpt_dir/:$work_dir/checkpoints/ -v $tb_dir/:$work_dir/runs"

if [ -x "$(command -v nvidia-docker)" ]; then
    NV_GPU=$gpu_id nvidia-docker run $args pccl/$docker_image_name ./train.sh $@
else
    docker run --gpus=$gpu_id $args pccl/$docker_image_name ./train.sh $@
fi

