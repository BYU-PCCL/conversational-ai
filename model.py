"""Finetunes a T5 model on the Chit-Chat Datset as part of a chatbot.

The T5 model is described in the paper: https://arxiv.org/abs/1910.10683.
"""

import json
import os
from contextlib import suppress
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import t5
import tensorflow as tf

import dataset


def finetune(
    steps: int = 25000,
    train_path: Union[str, Path] = "./train.tsv",
    validation_path: Union[str, Path] = "./validation.tsv",
    model_size: str = "large",
    model_dir: Union[str, Path] = "./models",
    model_parallelism: int = 1,
    data_parallelism: Optional[int] = None,
    global_batch_size: Union[int, Tuple[str, int]] = ("tokens_per_batch", 1024),
    sequence_length: Dict[str, int] = dict(inputs=256, targets=128),  # noqa: B008
    learning_rate_schedule: float = 0.003,
    keep_checkpoint_max: int = 5,
    save_checkpoints_steps: int = 1000,
    gpus: Optional[List[str]] = None,
    gpu_memory_growth: bool = True,
    run_name: Optional[str] = None,
    **kwargs: Dict[str, Any],
) -> None:
    """Finetunes a T5 model."""
    num_input_examples = None
    if not Path(train_path).is_file():
        train_len, val_len = dataset.write_to_files(train_path, validation_path)
        num_input_examples = dict(train=train_len, validation=val_len)

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
        num_input_examples=num_input_examples,
    )

    mesh_devices = list(_init_gpus(gpus or [], gpu_memory_growth))

    model_parallelism = max(model_parallelism, 1)
    if not data_parallelism:
        data_parallelism = max(len(mesh_devices) // model_parallelism, 1)

    Path(model_dir).mkdir(parents=True, exist_ok=True)

    model = t5.models.MtfModel(
        model_dir=model_dir,
        model_parallelism=model_parallelism,
        mesh_shape=f"model:{model_parallelism},batch:{data_parallelism}",
        mesh_devices=mesh_devices,
        tpu=None,
        tpu_topology=None,
        batch_size=global_batch_size,
        learning_rate_schedule=learning_rate_schedule,
        keep_checkpoint_max=keep_checkpoint_max,
        save_checkpoints_steps=save_checkpoints_steps,
        sequence_length=sequence_length,
        **kwargs,
    )

    model.finetune(
        mixture_or_task_name="conversation",
        pretrained_model_dir=f"gs://t5-data/pretrained_models/{model_size.lower()}",
        finetune_steps=int(steps),
    )

    model.batch_size = model.batch_size * 4
    model.eval(mixture_or_task_name="conversation", checkpoint_steps="all")


def _init_gpus(gpus: Iterable[str], memory_growth: bool) -> Iterable[str]:
    def _conf(gpu: Any) -> str:
        tf.config.experimental.set_memory_growth(gpu, memory_growth)
        return gpu.name.replace("/physical_device:", "").lower()

    all_gpus = set(map(_conf, tf.config.experimental.list_physical_devices("GPU")))

    return set(g.lower() for g in gpus).intersection(all_gpus) if gpus else all_gpus


if __name__ == "__main__":
    app = "CONVERSATIONAL_AI_"

    def _parse(key: str, value: Any) -> Tuple[str, Any]:
        with suppress(ValueError):
            value = json.loads(value)
        return key.replace(app, "").lower(), value

    finetune(**dict(_parse(k, v) for k, v in os.environ.items() if k.startswith(app)))
