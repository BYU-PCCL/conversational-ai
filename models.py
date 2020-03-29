"""Finetunes a T5 model on the Chit-Chat Datset as part of a chatbot.

The T5 model is described in the paper: https://arxiv.org/abs/1910.10683.
"""

import os
import tempfile
from functools import partial
from pathlib import Path
from typing import Any, List, Optional, Union

import gin
import t5
from t5.models.mtf_model import utils

import dataset


class T5:
    """Wraps a `t5.models.MtfModel`."""

    def __init__(self, **kwargs) -> None:
        """Configures and instantiates a T5 model."""
        self._model = t5.models.MtfModel(**kwargs)
        Path(self._model._model_dir).mkdir(parents=True, exist_ok=True)

    @gin.configurable
    def finetune(
        self,
        mixture_or_task_name: str,
        steps: int = 10000,
        pretrained_model_dir: str = "gs://t5-data/pretrained_models/base",
    ) -> None:
        """Finetunes a T5 model."""
        self._model.finetune(mixture_or_task_name, steps, pretrained_model_dir)

    @gin.configurable
    def evaluate(self, mixture_or_task_name: str, steps: Optional[int] = -1) -> None:
        """Evaluates a T5 model."""
        self._model.batch_size = self._model.batch_size * 4
        self._model.eval(mixture_or_task_name, checkpoint_steps=steps)

    @gin.configurable
    def predict(self, model_input: List[str], **kwargs) -> List[str]:
        """Makes a prediction using a trained T5 model."""
        kwargs.setdefault("checkpoint_steps", -1)

        # HACK: get around t5's lame API that requires filesystem I/O
        with tempfile.TemporaryDirectory() as tmp:
            in_file = Path(tmp, "input.txt")
            in_file.write_text("\n".join(model_input))
            out_file = Path(tmp, "output.txt")

            self._model.predict(str(in_file), str(out_file), **kwargs)

            # will have the checkpoint appended to it so we glob to get all of them
            outputs = [p.read_text() for p in Path(tmp).glob(f"{out_file.name}*")]
            return "\n".join(outputs).split("\n")  # return the flattened list


@gin.configurable
def register_task(
    train_path: str, validation_path: str, mixture_or_task_name: str,
) -> None:
    """Registers a task for use with training or evaluating a T5 model."""
    num_input_examples = None
    if not Path(train_path).is_file():
        train_len, val_len = dataset.write_to_files(train_path, validation_path)
        num_input_examples = dict(train=train_len, validation=val_len)

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


def _gin_setdefault(key: str, value: str) -> Any:
    try:
        value = gin.query_parameter(key)
    except ValueError:
        gin.bind_parameter(key, value)
    return value


def _gin_glob_parse_configs_in(directory: Union[str, Path]) -> None:
    for p in Path(directory).glob("*.gin"):
        gin.parse_config_file(p)


if __name__ == "__main__":
    _gin_glob_parse_configs_in("./")

    utils.parse_gin_defaults_and_flags()

    with gin.unlock_config():
        # TODO: should we get `default_model_dir` from CONVERSATIONAL_AI_MODEL_DIR?
        default_model_dir = os.getenv("CONVERSATIONAL_AI_MODEL_DIR", "./checkpoints/t5")
        model_dir = _gin_setdefault("MtfModel.model_dir", default_model_dir)
        _gin_setdefault("register_task.train_path", f"{model_dir}/train.tsv")
        _gin_setdefault("register_task.validation_path", f"{model_dir}/validation.tsv")
        _gin_glob_parse_configs_in(model_dir)

    register_task()

    model = T5()

    model.finetune()
    model.evaluate()
