#!/bin/sh

docker_image_name=conversational-ai

docker pull pccl/$docker_image_name

echo ""

gpu_id=${NV_GPU:-all}
run_name="${docker_image_name}_$(date +%s)"
chkpt_dir=${CONVERSATIONAL_AI_CHECKPOINT_DIR:-/mnt/pccfs/not_backed_up/will/checkpoints}

args=$(env | awk 'BEGIN { ORS=" " }; $0 ~ /CONVERSATIONAL_AI_/ { print "-e", $0 }')
args="-d --rm --name=$run_name -P --ipc=host -e CONVERSATIONAL_AI_RUN_NAME=$run_name -v $chkpt_dir/:/checkpoint/ $args"

if [ -x "$(command -v nvidia-docker)" ]; then
    NV_GPU=$gpu_id nvidia-docker run $args pccl/$docker_image_name $@
else
    docker run --gpus=$gpu_id $args pccl/$docker_image_name $@
fi

echo ""

docker ps --filter=name=$docker_image_name

echo ""

docker logs -f $run_name
