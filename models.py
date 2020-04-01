"""Finetunes a T5 model on the Chit-Chat Datset as part of a chatbot.

The T5 model is described in the paper: https://arxiv.org/abs/1910.10683.
"""
import ast
import os
import tempfile
from functools import partial
from pathlib import Path
from typing import List, Optional

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

            # will have the checkpoint num appended to it so we glob to get all of them
            all_outputs = [p.read_text() for p in Path(tmp).glob(f"{out_file.name}*")]
            # TODO: should we return just the last one?
            outputs = filter(lambda x: x.strip(), "\n".join(all_outputs).split("\n"))
            # HACK: not sure if the model is outputing `b'output'` or something else is
            return [ast.literal_eval(line.strip()).decode("utf-8") for line in outputs]


@gin.configurable
def register_task(
    mixture_or_task_name: str,
    train_path: str = "./data/train.tsv",
    validation_path: str = "./data/train.tsv",
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


if __name__ == "__main__":
    """Usage: python3 models.py --gin_param="MtfModel.model_dir='./checkpoints'"

    See: https://github.com/google/gin-config for more info on configuration
    """
    for p in Path("./").glob("*.gin"):
        gin.parse_config_file(p)

    utils.parse_gin_defaults_and_flags()

    register_task()  # noqa: E1120

    model = T5()

    # TODO: should we also check `logging.getLogger("tensorflow").level` ?
    if int(os.getenv("TF_CPP_MIN_LOG_LEVEL", 0)) < 2:
        print("# Gin config", "# " + "=" * 78, gin.operative_config_str(), sep="\n")

    model.finetune()  # noqa: E1120
    model.evaluate()  # noqa: E1120
