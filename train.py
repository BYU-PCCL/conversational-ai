#!/usr/bin/env python3

import math
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"
import tensorflow as tf  # isort:skip

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import gpt_2_simple as gpt2  # isort:skip


RUN_NAME = os.getenv("CONVERSATIONAL_AI_RUN_NAME", "conversational-ai")
MODEL_NAME = os.getenv("CONVERSATIONAL_AI_MODEL_NAME", "355M")
STEPS = max(int(os.getenv("CONVERSATIONAL_AI_STEPS", 2000)), 1)
TRAIN_FILE = os.getenv("CONVERSATIONAL_AI_TRAIN_FILE", "train.txt")


def batch_size_for(devices):
    bs = os.getenv("CONVERSATIONAL_AI_BATCH_SIZE")
    if bs is not None:
        return int(bs)

    mem = min(x.memory_limit_bytes for x in devices if x.device_type == "GPU")
    mem /= 1024 ** 3
    return 2 ** int(math.log(mem, 2))


if not os.path.isdir(os.path.join("models", MODEL_NAME)):
    gpt2.download_gpt2(model_name=MODEL_NAME)


if not os.path.isfile(TRAIN_FILE):
    import dataset

    dataset.write_to_file(TRAIN_FILE)

sess = gpt2.start_tf_sess()

gpt2.finetune(
    sess,
    TRAIN_FILE,
    run_name=RUN_NAME,
    model_name=MODEL_NAME,
    multi_gpu=True,
    batch_size=batch_size_for(sess.list_devices()),
    steps=STEPS,
    learning_rate=0.0001,
    sample_every=1000,
    max_checkpoints=10,
    save_every=100 if STEPS > 100 else max(STEPS // 10, 1),
    print_every=25 if STEPS > 25 else 1,
)
