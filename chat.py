#!/usr/bin/env python3

import os
import readline

import gpt_2_simple as gpt2

RUN_NAME = os.getenv("RUN_NAME", "conversational-ai")
# MODEL_NAME = os.getenv("MODEL_NAME", "1558M")
MODEL_NAME = os.getenv("MODEL_NAME", "355M")


if not os.path.isdir(os.path.join("models", MODEL_NAME)):
    print("Downloading ", MODEL_NAME, " model...")
    gpt2.download_gpt2(model_name=MODEL_NAME)

sess = gpt2.start_tf_sess()

gpt2.load_gpt2(sess, run_name=RUN_NAME, model_name=MODEL_NAME)

prefix = "<|startoftext|>"
while True:
    prefix += "> " + input("> ")

    output = gpt2.generate(
        sess, run_name=RUN_NAME, model_name=MODEL_NAME, prefix=prefix
    )

    print(output)
