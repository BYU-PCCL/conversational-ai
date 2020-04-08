"""Tasks for training a T5 model.

https://github.com/google-research/text-to-text-transfer-transformer
"""
import functools
from typing import Dict, Iterable

import chitchat_dataset as ccc
import t5
import tensorflow.compat.v1 as tf


def _compounding_dataset(num: int, split: str, shuffle_files: bool) -> tf.data.Dataset:
    """Creates a tf.data.Dataset of input/target examples.

    Each example has the form: `{"inputs": tf.string, "targets": tf.string}`.
    """

    def _generate_dataset() -> Iterable[Dict[str, str]]:
        for inputs, targets in ccc.CompoundingConversationDataset(prefix="converse: "):
            yield {"inputs": inputs, "targets": targets}

    dataset = tf.data.Dataset.from_generator(
        _generate_dataset,
        output_types={"inputs": tf.string, "targets": tf.string},
        output_shapes={"inputs": tf.TensorShape([]), "targets": tf.TensorShape([])},
    )
    return dataset.take(num) if split == "train" else dataset.skip(num)


t5.data.TaskRegistry.add(
    "conversation_v001_compounding",
    t5.data.Task,
    dataset_fn=functools.partial(_compounding_dataset, 124_990),
    splits=["train", "validation"],
    text_preprocessor=None,
    postprocess_fn=t5.data.postprocessors.lower_text,
    metric_fns=[
        t5.evaluation.metrics.accuracy,
        t5.evaluation.metrics.rouge,
        t5.evaluation.metrics.bleu,
    ],
    sentencepiece_model_path=t5.data.DEFAULT_SPM_PATH,
)
