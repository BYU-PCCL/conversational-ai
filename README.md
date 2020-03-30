# conversational-ai

![ci](https://github.com/BYU-PCCL/conversational-ai/workflows/ci/badge.svg)
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
./docker.py
```

or equivalently:

```
curl -fsSL git.io/pccl-conversational-ai-docker-py | python3
```

### chat

to chat interactively with a trained model, do:

```
./docker.py -tc 'python3 chat.py'
```
