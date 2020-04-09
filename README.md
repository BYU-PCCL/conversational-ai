# conversational-ai

[![ci](https://github.com/BYU-PCCL/conversational-ai/workflows/ci/badge.svg)](https://github.com/BYU-PCCL/conversational-ai/actions?query=workflow%3Aci)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## requirements

- `pip>=20.0`
- `setuptools>=41.0.0`
- `wheel>=0.34.2`

## installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip wheel setuptools && pip install -r requirements.txt
```

## usage

```
./docker.py --help
```

### finetune

```bash
python3 -m conversational_ai.t5_model --gin_file=finetune.gin
```

or

```bash
./docker.py --gin_file=finetune_3b.gin
```

### chat

to chat interactively with a trained model, do:

```bash
python3 -m conversational_ai.chat \
    --gin_location_prefix=./path/to/checkpoint/ \
    --gin_file=infer.gin
```

or

```
docker.py --tty -m conversational_ai.chat \
    --gin_location_prefix=./path/to/checkpoint/ \
    --gin_file=infer.gin
```
