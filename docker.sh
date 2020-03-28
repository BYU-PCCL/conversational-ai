#!/bin/sh

# Finetune the model in a docker container.
#
# usage: ./docker.sh
#
# Any environment variable starting with `CONVERSATIONAL_AI_` will be passed
# into the container. You can also limit the available GPUS by running e.g.:
# `NVIDIA_VISIBLE_DEVICES=0,1 ./docker.sh`


# setup variables & build up docker args

name="conversational-ai"
tag="${DOCKER_TAG:-latest}"
image="pccl/$name:$tag"

export CONVERSATIONAL_AI_RUN_NAME="${CONVERSATIONAL_AI_RUN_NAME:-$(basename $name)_$tag_$(hostname)_$(date +%s)}"

export CONVERSATIONAL_AI_MODEL_DIR="${CONVERSATIONAL_AI_MODEL_DIR:-/mnt/pccfs/not_backed_up/will/checkpoints/$CONVERSATIONAL_AI_RUN_NAME}"
export DAILY_DIALOG_PATH="${DAILY_DIALOG_PATH:-/mnt/pccfs/not_backed_up/data/daily_dialog/ijcnlp_dailydialog/dialogues_text.txt}"

export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-${NV_GPU:-all}}"

# use the syntax for the old version of nvidia-container-toolkit as the default
gpu_args="-e NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES --runtime=nvidia"
docker run --help | grep -q -- '--gpus' && gpu_args="--gpus=$NVIDIA_VISIBLE_DEVICES"

mkdir -p "$CONVERSATIONAL_AI_MODEL_DIR"
mount_args=""
for path in \
    "/mnt" \
    "$CONVERSATIONAL_AI_MODEL_DIR" \
    "$CONVERSATIONAL_AI_TRAIN_PATH" \
    "$CONVERSATIONAL_AI_VALIDATION_PATH" \
    "$DAILY_DIALOG_PATH"
do
    [ -n "$path" ] && [ -e "$path" ] && mount_args="-v $path:$path $mount_args"
done


# run the docker command(s)

if git remote get-url origin 2>/dev/null | grep -qi "$name"; then
    cd "$(git rev-parse --show-toplevel)" || exit

    printf "Building %s docker image...\n" "$image"
    docker build -t "$image" .

    # default to yes iff `DOCKER_PUSH` is not set
    if echo "${DOCKER_PUSH=1}" | grep -qiE "^(1|t(rue)?|y(es)?)$"; then
        printf "\nPushing %s docker image...\n" "$image"
        docker push "$image"
    fi
else
    printf "Pulling latest %s docker image...\n" "$image"
    docker pull "$image"
fi

printf "\nStarting container %s...\n" "$CONVERSATIONAL_AI_RUN_NAME"
# shellcheck disable=SC2086,SC2046
docker run --name="$CONVERSATIONAL_AI_RUN_NAME" \
    --detach --rm --publish-all \
    --ipc=host --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 \
    $gpu_args \
    $mount_args \
    $(env | awk 'BEGIN { ORS=" " }; $0 ~ /CONVERSATIONAL_AI_/ { print "-e", $0 }') \
    -e DAILY_DIALOG_PATH=$DAILY_DIALOG_PATH \
    "$image" \
    "$@"

printf "\n(Press CTRL+C to stop viewing the %s logs)\n" "$image"
docker logs -f "$CONVERSATIONAL_AI_RUN_NAME"
