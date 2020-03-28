"""Finetunes a T5 model on the Chit-Chat Datset as part of a chatbot.

The T5 model is described in the paper: https://arxiv.org/abs/1910.10683.
"""

import json
import os
import tempfile
from contextlib import suppress
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import t5

import dataset

_DEFAULT_MODEL_DIR = "./models"  # HACK

# TODO: figure out CLI so we don't have to have **kwargs params for everything


def register_task(
    mixture_or_task_name: str,
    train_path: str,
    validation_path: str,
    num_input_examples: Optional[Dict[str, int]] = None,
) -> None:
    """Registers a task for use with training or evaluating a T5 model."""
    t5.data.TaskRegistry.add(
        mixture_or_task_name,
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


class T5:
    def __init__(self, model_size: str = "large", **kwargs,) -> None:
        """Configures and instantiates a T5 model."""
        kwargs = {
            "model_dir": _DEFAULT_MODEL_DIR,
            "model_parallelism": 1,
            "batch_size": ("tokens_per_batch", 1024),
            "sequence_length": dict(inputs=256, targets=128),
            "learning_rate_schedule": 0.003,
            "keep_checkpoint_max": 5,
            "save_checkpoints_steps": 1000,
            "tpu": None,
            "tpu_topology": None,
            "mesh_devices": ["gpu:0"],
            **kwargs,
        }

        kwargs["model_dir"] = str(kwargs["model_dir"])  # in case we get a `Path`
        kwargs["model_parallelism"] = max(kwargs["model_parallelism"], 1)
        if not kwargs.get("mesh_shape"):
            d_par = max(len(kwargs["mesh_devices"]) // kwargs["model_parallelism"], 1)
            kwargs["mesh_shape"] = f"model:{kwargs['model_parallelism']},batch:{d_par}"

        self._model_kwargs = kwargs
        self._model_size = model_size.lower()

        Path(self._model_kwargs["model_dir"]).mkdir(parents=True, exist_ok=True)
        self._model = t5.models.MtfModel(**kwargs)

    def finetune(
        self,
        mixture_or_task_name: str,
        steps: int = 10000,
        pretrained_model_dir: Optional[str] = None,
    ) -> None:
        """Finetunes a T5 model."""
        if pretrained_model_dir is None:
            pretrained_model_dir = f"gs://t5-data/pretrained_models/{self._model_size}"
        self._model.finetune(mixture_or_task_name, steps, pretrained_model_dir)

    def evaluate(self, mixture_or_task_name: str, steps: Optional[int] = -1) -> None:
        """Evaluates a T5 model."""
        self._model.batch_size = self._model.batch_size * 4
        self._model.eval(mixture_or_task_name, checkpoint_steps=steps)

    def predict(self, model_input: List[str]) -> List[str]:
        """Makes a prediction using a trained T5 model."""
        # HACK: get around t5's lame API that requires filesystem I/O
        with tempfile.TemporaryDirectory() as tmp:
            in_file = Path(tmp, "input.txt")
            in_file.write_text("\n".join(model_input))
            out_file = Path(tmp, "output.txt")

            self._model.predict(str(in_file), str(out_file), checkpoint_steps=-1)

            # will have the checkpoint appended to it so we glob to get all of them
            outputs = [p.read_text() for p in Path(tmp).glob(f"{out_file.name}*")]
            return "\n".join(outputs).split("\n")  # return the flattened list


if __name__ == "__main__":
    app = "CONVERSATIONAL_AI_"

    def _parse(key: str, value: Any) -> Tuple[str, Any]:
        with suppress(ValueError):
            value = json.loads(value)
        return key.replace(app, "").lower(), value

    kwargs = dict(_parse(k, v) for k, v in os.environ.items() if k.startswith(app))
    # TODO: figure out CLI so we don't have to do all this crap
    _ = kwargs.pop("run_name", None)
    other_kwargs = {k: kwargs.pop(k, None) for k in ["steps", "pretrained_model_dir"]}
    other_kwargs = {k: v for k, v in other_kwargs.items() if v is not None}
    mixture_or_task_name = kwargs.pop("mixture_or_task_name", "conversation")
    model_dir = kwargs.setdefault("model_dir", _DEFAULT_MODEL_DIR)
    train_path = str(kwargs.pop("train_path", Path(model_dir, "train.tsv")))
    val_path = str(kwargs.pop("validation_path", Path(model_dir, "validation.tsv")))

    num_input_examples = None
    if not Path(train_path).is_file():
        train_len, val_len = dataset.write_to_files(train_path, val_path)
        num_input_examples = dict(train=train_len, validation=val_len)
    register_task(mixture_or_task_name, train_path, val_path, num_input_examples)

    model = T5(**kwargs)
    model.finetune(mixture_or_task_name=mixture_or_task_name, **other_kwargs)
    model.evaluate(mixture_or_task_name=mixture_or_task_name)
