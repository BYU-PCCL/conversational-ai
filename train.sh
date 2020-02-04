#!/bin/sh

docker_image_name=conversational-ai
run_name="$docker_image_name-$(date +%s)"

work_dir=/

local_root_dir=/mnt/pccfs/not_backed_up/will

chkpt_dir=${CHECKPOINT_DIR:-$local_root_dir/checkpoints/}
mkdir -p $chkpt_dir

docker pull pccl/$docker_image_name

gpu_id=${NV_GPU:-all}
args="-d --rm --name=$run_name --ipc=host -e RUN_NAME=$run_name -v $chkpt_dir/:$work_dir/checkpoint/"

if [ -x "$(command -v nvidia-docker)" ]; then
    NV_GPU=$gpu_id nvidia-docker run $args pccl/$docker_image_name $@
else
    docker run --gpus=$gpu_id $args pccl/$docker_image_name $@
fi

docker logs -f $run_name

