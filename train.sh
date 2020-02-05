#!/bin/sh

docker_image_name=conversational-ai

if [ -e "Dockerfile" ]; then
    docker build -t pccl/$docker_image_name .
else
    docker pull pccl/$docker_image_name
fi

echo ""

gpu_id=${NV_GPU:-all}
run_name="${docker_image_name}_$(date +%s)"
chkpt_dir=${CHECKPOINT_DIR:-/mnt/pccfs/not_backed_up/will/checkpoints}
args="-d --rm --name=$run_name -P --ipc=host -e RUN_NAME=$run_name -v $chkpt_dir/:/checkpoint/"

if [ -x "$(command -v nvidia-docker)" ]; then
    NV_GPU=$gpu_id nvidia-docker run $args pccl/$docker_image_name $@
else
    docker run --gpus=$gpu_id $args pccl/$docker_image_name $@
fi

echo ""

docker ps --filter=name=$docker_image_name

echo ""

docker logs -f $run_name

