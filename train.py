"""Finetunes a T5 model on the Chit-Chat Datset as part of a chatbot.

The T5 model is described in the paper: https://arxiv.org/abs/1910.10683.
"""

import logging
import os
from pathlib import Path
from typing import Union

import t5
import tensorflow as tf

import dataset


def finetune(
    # steps: int = 25000,
    steps: int = 1010000,
    train_path: Union[str, Path] = "./train.tsv",
    validation_path: Union[str, Path] = "./validation.tsv",
    model_dir: Union[str, Path] = "./models",
    model_size: str = "large",
    model_parallelism: int = 8,
    # TODO: automatically figure out batch_size
    train_batch_size: int = 64,
    keep_checkpoint_max: int = 5,
    pretrained_dir: str = None,
) -> None:
    """Finetunes a T5 model."""
    model_size = model_size.lower()
    if not pretrained_dir:
        pretrained_dir = f"gs://t5-data/pretrained_models/{model_size}"

    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    tf.compat.v1.disable_v2_behavior()

    # TODO: use a tensorflow dataset to avoid writing to a file?
    if not Path(train_path).is_file():
        dataset.write_to_files(train_path, validation_path)

    t5.data.TaskRegistry.add(
        "conversation",
        t5.data.TextLineTask,
        split_to_filepattern={"train": train_path, "validation": validation_path},
        text_preprocessor=[
            partial(t5.data.preprocessors.parse_tsv, field_names=["inputs", "targets"]),
        ],
        postprocess_fn=t5.data.postprocessors.lower_text,
        metric_fns=[t5.evaluation.metrics.accuracy, t5.evaluation.metrics.rouge],
        sentencepiece_model_path=t5.data.DEFAULT_SPM_PATH,
        # num_input_examples=sum(1 for _ in TRAIN_FILE.open(mode="r")),
    )

    model = t5.models.MtfModel(
        model_dir=model_dir,
        # TODO: automatically determine GPU settings...
        mesh_shape="model:1,batch:1",
        mesh_devices=["gpu:0"],
        tpu=None,
        tpu_topology=None,
        model_parallelism=model_parallelism,
        batch_size=train_batch_size,
        # sequence_length={"inputs": 128, "targets": 32},
        learning_rate_schedule=0.003,
        save_checkpoints_steps=5000,
        keep_checkpoint_max=keep_checkpoint_max,
        iterations_per_loop=100,
    )

    model.finetune(
        mixture_or_task_name="conversation",
        pretrained_model_dir=pretrained_dir,
        finetune_steps=int(steps),
    )

    model.batch_size = train_batch_size * 4
    model.eval(mixture_or_task_name="conversation", checkpoint_steps="all")


if __name__ == "__main__":
    kwargs = dict(
        finetune_steps=os.getenv("CONVERSATIONAL_AI_FINETUNE_STEPS"),
        model_size=os.getenv("CONVERSATIONAL_AI_MODEL_DIR"),
        model_dir=os.getenv("CONVERSATIONAL_AI_MODEL_DIR"),
        train_file=os.getenv("CONVERSATIONAL_AI_TRAIN_FILE"),
    )

    finetune(**{k: v for k, v in kwargs.items() if v is not None})  # type: ignore
