"""Finetunes a T5 model on the Chit-Chat Datset as part of a chatbot.

The T5 model is described in the paper: https://arxiv.org/abs/1910.10683.
"""

import json
import logging
import os
from contextlib import contextmanager, suppress
from functools import partial
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple, Union

import tensorflow as tf

import dataset


def finetune(
    steps: int = 25000,
    train_path: Union[str, Path] = "./train.tsv",
    validation_path: Union[str, Path] = "./validation.tsv",
    model_dir: Union[str, Path] = "./models",
    model_size: str = "large",
    model_parallelism: int = 8,
    # TODO: automatically figure out batch_size
    train_batch_size: int = 64,
    keep_checkpoint_max: int = 5,
    pretrained_dir: Optional[str] = None,
    mesh_devices: Optional[List[str]] = None,
    mesh_shape: Optional[str] = None,
    gpu_memory_growth: bool = True,
    **kwargs: Dict[str, Any],  # so we can pass in all the args from the env
) -> None:
    """Finetunes a T5 model."""
    import t5  # import t5 here so we can set the log level first

    gpus = _init_gpus(gpu_memory_growth)
    if not gpus:
        raise RuntimeError("This model requires a GPU to finetune.")

    model_size = model_size.lower()
    if not pretrained_dir:
        pretrained_dir = f"gs://t5-data/pretrained_models/{model_size}"

    # TODO: use a tensorflow dataset to avoid writing to a file?
    if not Path(train_path).is_file():
        _, _ = dataset.write_to_files(train_path, validation_path)

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

    if not mesh_shape:
        # TODO: which should be larger `models_per_gpu` or `batch_size_per_gpu`?
        models_per_gpu, batch_size_per_gpu = next(_factors(len(gpus)))
        mesh_shape = f"model:{models_per_gpu},batch:{batch_size_per_gpu}"

    if not mesh_devices:
        # TODO: add support for tf.device upstream in T5?
        mesh_devices = [g.name.replace("/physical_device:", "").lower() for g in gpus]

    Path(model_dir).mkdir(parents=True, exist_ok=True)

    model = t5.models.MtfModel(
        model_dir=model_dir,
        mesh_shape=mesh_shape,
        mesh_devices=mesh_devices,
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


@contextmanager
def log_level(level: Union[str, int], name: str = "tensorflow") -> Generator:
    """Set and restore the log level for the desired package."""

    def _set_log_level(level: Union[str, int], name: str) -> None:
        level = level.upper() if isinstance(level, str) else level
        logging.getLogger(name).setLevel(level)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(logging.getLogger(name).level // 10)

    og_level = logging.getLogger(name).level
    _set_log_level(level, name)
    yield
    _set_log_level(og_level, name)


def _factors(n: int, include_trivial_factor: bool = False) -> Iterator[Tuple[int, int]]:
    start = 1 if include_trivial_factor or n == 1 else 2
    for i in range(start, int(pow(n, 1 / 2)) + 1):
        if n % i == 0:
            yield i, int(n / i)


def _init_gpus(memory_growth: bool = True) -> List[Any]:
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, memory_growth)
    return gpus


if __name__ == "__main__":
    app = "CONVERSATIONAL_AI_"

    def _parse(key: str, value: Any) -> Tuple[str, Any]:
        with suppress(ValueError):
            value = json.loads(value)
        return key.replace(app, "").lower(), value

    finetune(**dict(_parse(k, v) for k, v in os.environ.items() if k.startswith(app)))
