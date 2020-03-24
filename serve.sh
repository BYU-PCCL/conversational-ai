#!/bin/sh

docker_image_name=conversational-ai

docker pull pccl/$docker_image_name

echo ""

gpu_id=${NV_GPU:-all}

run_name=${docker_image_name}_server

chkpt_dir=${CONVERSATIONAL_AI_CHECKPOINT_DIR:-/mnt/pccfs/not_backed_up/will/checkpoints}
# shellcheck disable=SC2086,SC2012
chkpt=${CONVERSATIONAL_AI_CHECKPOINT:-$chkpt_dir/$(ls -1r $chkpt_dir | head -n 1)}

# chats_dir=${CONVERSATIONAL_AI_CHATS_DIR:-/mnt/pccfs/backed_up/will/chats}

args=$(env | awk 'BEGIN { ORS=" " }; $0 ~ /CONVERSATIONAL_AI_/ { print "-e", $0 }')
args="-d --rm --name=$run_name -p 8080:8080 --ipc=host -v $chkpt/:/checkpoint/ $args"

if [ -x "$(command -v nvidia-docker)" ]; then
    # shellcheck disable=SC2086
    NV_GPU=$gpu_id nvidia-docker run $args pccl/$docker_image_name python3 serve.py
else
    # shellcheck disable=SC2086
    docker run --gpus="$gpu_id" $args pccl/$docker_image_name python3 serve.py
fi

echo ""

docker ps --filter=name=$docker_image_name

echo ""

docker logs -f $run_name
