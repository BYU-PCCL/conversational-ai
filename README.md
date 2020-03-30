# conversational-ai

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## requirements

- `pip>=20.0`
- `setuptools>=41.0.0`
- `wheel>=0.34.2`

## installation

See the [Dockerfile](Dockerfile)...

## usage

### finetune

```
./docker.sh
```

or equivalently:

```
curl -fsSL git.io/pccl-conversational-ai-docker-sh | sh
```

### chat

to chat interactively with a trained model, do:

```
env DOCKER_PUSH=no DOCKER_ARGS=-it ./docker.sh chat.py
```
