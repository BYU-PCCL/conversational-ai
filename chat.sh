#!/bin/sh

docker run \
    -it \
    --rm \
    --gpus=${NV_GPU:-all} \
    --ipc=host \
    -v ${CHECKPOINT_DIR:-/mnt/pccfs/not_backed_up/will/checkpoints}/:/checkpoint/ \
    pccl/conversational-ai \
    python3 chat.py \
    $@

