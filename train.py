#!/usr/bin/env python3

import os

import gpt_2_simple as gpt2

RUN_NAME = os.getenv("RUN_NAME", "conversational-ai")
# MODEL_NAME = os.getenv("MODEL_NAME", "1558M")
MODEL_NAME = os.getenv("MODEL_NAME", "355M")
STEPS = max(int(os.getenv("STEPS", 1000)), 10)
TRAIN_FILE = os.getenv("TRAIN_FILE", "train.txt")


if not os.path.isdir(os.path.join("models", MODEL_NAME)):
    print("Downloading ", MODEL_NAME, " model...")
    gpt2.download_gpt2(model_name=MODEL_NAME)

sess = gpt2.start_tf_sess()

if not os.path.isfile(TRAIN_FILE):
    import dataset

    dataset.write_to_file(TRAIN_FILE)

gpt2.finetune(
    sess,
    TRAIN_FILE,
    run_name=RUN_NAME,
    model_name=MODEL_NAME,
    multi_gpu=True,
    # TODO: figure out batch_size automatically
    batch_size=32,
    learning_rate=0.0001,
    sample_every=10000,
    max_checkpoints=5,
    save_every=STEPS // 10,
    steps=STEPS,
)

gpt2.generate(sess)
