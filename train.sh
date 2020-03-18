#!/bin/sh

image_name=conversational-ai

docker pull pccl/$image_name

echo ""

gpu_id=${NV_GPU:-all}
run_name="${image_name}_$(date +%s)"
chkpt_dir="${CONVERSATIONAL_AI_CHECKPOINT_DIR:-/mnt/pccfs/not_backed_up/will/checkpoints/$run_name}"

args=$(env | awk 'BEGIN { ORS=" " }; $0 ~ /CONVERSATIONAL_AI_/ { print "-e", $0 }')
args="-d --rm --name=$run_name -P --ipc=host -e CONVERSATIONAL_AI_RUN_NAME=$run_name -v $chkpt_dir/:/models/ $args"

mkdir -p "$chkpt_dir"

if [ -x "$(command -v nvidia-docker)" ]; then
    NV_GPU=$gpu_id nvidia-docker run $args pccl/$image_name $@
else
    docker run --gpus=$gpu_id $args pccl/$image_name $@
fi

echo ""
docker ps --filter=name=$image_name

echo ""
docker port $run_name | sed "s#6006/tcp -> 0.0.0.0:#Tensorboard running at http://$(hostname):#"

echo "(Press CTRL+C to stop viewing the $image logs)"
docker logs -f $run_name
