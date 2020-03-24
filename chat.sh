#!/bin/sh

docker run \
    -it --rm --ipc=host --gpus=all \
    -v "${CONVERSATIONAL_AI_CHECKPOINT_DIR:-$(git rev-parse --show-toplevel || $PWD)/checkpoint}:/checkpoint/" \
    -v "$PWD/chats:/chats/" \
    pccl/conversational-ai \
    python3 chat.py "$@"
