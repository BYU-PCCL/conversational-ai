# conversational-ai

## train

with docker:

```
./docker-train.sh
```

without docker:

```
./train.sh
```

## chat

to run a trained GPT2 model, do:

```
source ./venv/bin/activate
python3 run_gpt2.py --model_name_or_path=/path/to/checkpoints/
```
