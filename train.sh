#!/bin/sh

# Finetune the model in a docker container.
#
# usage: ./train.sh
#
# Any environment variable starting with `CONVERSATIONAL_AI_` will be passed
# into the container. You can also limit the available GPUS by running e.g.:
# `NV_GPU=0,1 ./train.sh`


image="pccl/conversational-ai"

[ -z "$NV_GPU" ] && export NV_GPU=all

[ -z "$CONVERSATIONAL_AI_RUN_NAME" ] && CONVERSATIONAL_AI_RUN_NAME="$(basename $image)_$(hostname)_$(date +%s)"
export CONVERSATIONAL_AI_RUN_NAME

[ -z "$CONVERSATIONAL_AI_MODEL_DIR" ] && export CONVERSATIONAL_AI_MODEL_DIR="/mnt/pccfs/not_backed_up/will/checkpoints/$CONVERSATIONAL_AI_RUN_NAME"

docker="docker"
gpu_args=""
# backwards compatability with old versions of nvidia-container-toolkit
if [ -x "$(command -v nvidia-docker)" ]; then
    docker="nvidia-docker"
elif [ -x "$(command -v docker)" ]; then
    gpu_args="--gpus=$NV_GPU"
else
    printf "Could not find docker executable; see: https://docker.com/get-started\n" 1>&2
    exit 1
fi

mkdir -p "$CONVERSATIONAL_AI_MODEL_DIR"
mount_args=""
for path in \
    "/mnt" \
    "$CONVERSATIONAL_AI_MODEL_DIR" \
    "$CONVERSATIONAL_AI_TRAIN_PATH" \
    "$CONVERSATIONAL_AI_VALIDATION_PATH"
do
    [ -n "$path" ] && [ -e "$path" ] && mount_args="-v $path:$path $mount_args"
done


if git remote get-url origin 2>/dev/null | grep -qi "$image"; then
    cd "$(git rev-parse --show-toplevel)" || exit

    printf "Building %s docker image...\n" "$image"
    docker build -t "$image" .

    # default to yes iff `DOCKER_PUSH` is not set
    if echo "${DOCKER_PUSH=1}" | grep -qiE "^(1|t(rue)?|y(es)?)$"; then
        printf "\nPushing %s docker image...\n" "$image"
        docker push "$image"
    fi
else
    printf "\nPulling latest %s docker image...\n" "$image"
    docker pull "$image"
fi

printf "\nStarting container %s...\n" "$CONVERSATIONAL_AI_RUN_NAME"
# shellcheck disable=SC2086,SC2046
$docker run --name="$CONVERSATIONAL_AI_RUN_NAME" \
    --detach --rm --publish-all --user=$(id -u):$(id -g) \
    --ipc=host --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 \
    $gpu_args \
    $mount_args \
    $(env | awk 'BEGIN { ORS=" " }; $0 ~ /CONVERSATIONAL_AI_/ { print "-e", $0 }') \
    "$image" \
    "$@"

docker port "$CONVERSATIONAL_AI_RUN_NAME" 6006 \
    | sed "s#0.0.0.0:#\nTensorboard running at http://$(hostname):#"

printf "\n(Press CTRL+C to stop viewing the %s logs)\n" "$image"
docker logs -f "$CONVERSATIONAL_AI_RUN_NAME"
