import datetime
import functools
import gzip
import json
import logging as py_logging
import os
import pprint
import random
import string
import sys
import time
import warnings
from contextlib import contextmanager

import t5
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

BASE_DIR = './'
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")


warnings.filterwarnings("ignore", category=DeprecationWarning)
tf.logging.set_verbosity(tf.logging.WARN)


@contextmanager
def tf_verbosity_level(level):
    og_level = tf.logging.get_verbosity()
    tf.logging.set_verbosity(level)
    yield
    tf.logging.set_verbosity(og_level)


MODEL_SIZE = "large"  # @param["small", "base", "large", "3B", "11B"]
PRETRAINED_DIR = os.path.join("gs://t5-data/pretrained_models", MODEL_SIZE)
MODEL_DIR = os.path.join(MODELS_DIR, MODEL_SIZE)

t5.data.TaskRegistry.add(
    "conversation",
    # Supply a function which returns a tf.data.Dataset.
    dataset_fn=nq_dataset_fn,
    splits=["train", "validation"],
    # Supply a function which preprocesses text from the tf.data.Dataset.
    # text_preprocessor=[trivia_preprocessor],
    # Use the same vocabulary that we used for pre-training.
    sentencepiece_model_path=t5.data.DEFAULT_SPM_PATH,
    # Lowercase targets before computing metrics.
    postprocess_fn=t5.data.postprocessors.lower_text,
    # We'll use accuracy as our evaluation metric.
    metric_fns=[t5.evaluation.metrics.accuracy],
    # Not required, but helps for mixing and auto-caching.
    num_input_examples=num_nq_examples,
)

# Set parallelism and batch size to fit on v2-8 TPU (if possible). Limit number
# of checkpoints to fit within 5GB (if possible).
model_parallelism, train_batch_size, keep_checkpoint_max = {
    "small": (1, 256, 16),
    "base": (2, 128, 8),
    "large": (8, 64, 4),
    "3B": (8, 16, 1),
    "11B": (8, 16, 1),
}[MODEL_SIZE]

tf.io.gfile.makedirs(MODEL_DIR)

model = t5.models.MtfModel(
    model_dir=MODEL_DIR,
    model_parallelism=model_parallelism,
    batch_size=train_batch_size,
    sequence_length={"inputs": 128, "targets": 32},
    learning_rate_schedule=0.003,
    save_checkpoints_steps=5000,
    keep_checkpoint_max=keep_checkpoint_max,
    iterations_per_loop=100,
)


model.finetune(
    mixture_or_task_name="trivia_all",
    pretrained_model_dir=PRETRAINED_DIR,
    finetune_steps=25000,
    mesh_shape="model:1,batch:1",
    mesh_devices=["gpu:0"],
)
