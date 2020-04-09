"""Tasks for training a T5 model.

https://github.com/google-research/text-to-text-transfer-transformer
"""
import functools
from typing import Callable, Dict, Iterable, Iterator

import chitchat_dataset as ccc
import t5
import tensorflow.compat.v1 as tf


def _generate_compounding_conversations() -> Iterable[Dict[str, str]]:
    for inputs, targets in ccc.CompoundingConversationDataset(prefix="converse: "):
        yield {"inputs": inputs, "targets": targets}


def _dataset(
    split: str,
    shuffle_files: bool,
    generator: Callable[[], Iterable[Dict[str, str]]],
    keys: Iterator[str],
    num_train: int,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types={k: tf.string for k in keys},
        output_shapes={k: tf.TensorShape([]) for k in keys},
    )
    return dataset.take(num_train) if split == "train" else dataset.skip(num_train)


t5.data.TaskRegistry.add(
    "conversation_v001_compounding",
    t5.data.Task,
    dataset_fn=functools.partial(
        _dataset,
        generator=_generate_compounding_conversations,
        keys=["inputs", "targets"],
        num_train=124_990,
    ),
    splits=["train", "validation"],
    text_preprocessor=None,
    postprocess_fn=t5.data.postprocessors.lower_text,
    metric_fns=[
        t5.evaluation.metrics.accuracy,
        t5.evaluation.metrics.bleu,
        t5.evaluation.metrics.rouge,
    ],
    sentencepiece_model_path=t5.data.DEFAULT_SPM_PATH,
)


t5.data.TaskRegistry.add(
    "conversation_v001_nsp",
    t5.data.Task,
    dataset_fn=functools.partial(
        _dataset,
        generator=lambda: ({"text": "\n".join(c)} for c in ccc.ConversationDataset()),
        keys=["text"],
        num_train=6952,
    ),
    splits=["train", "validation"],
    text_preprocessor=t5.data.preprocessors.next_sentence_prediction,
    postprocess_fn=t5.data.postprocessors.lower_text,
    metric_fns=[t5.evaluation.metrics.accuracy],
    sentencepiece_model_path=t5.data.DEFAULT_SPM_PATH,
)
