"""Dataset utilities for the ChitChat Challenge dataset."""
from typing import Callable, Dict, Iterable, Iterator

import chitchat_dataset as ccc
import tensorflow.compat.v1 as tf

from conversational_ai.dataset import utils


def generate_compounding_conversations(**kwargs) -> Iterable[Dict[str, str]]:
    """Yields examples from `ccc.CompoundingConversationDataset`."""
    kwargs.setdefault("prefix", "prefix: ")
    for inputs, targets in ccc.CompoundingConversationDataset(**kwargs):
        yield {"inputs": inputs, "targets": targets}


def generate_conversations_as_str(**kwargs) -> Iterable[Dict[str, str]]:
    """Yields examples from ``ccc.ConversationDataset()`` as strings."""
    for convo in ccc.ConversationDataset():
        yield {"text": utils.convo_as_str(convo=convo, **kwargs)}


def dataset(
    split: str,
    shuffle_files: bool,
    generator: Callable[[], Iterable[Dict[str, str]]],
    keys: Iterator[str],
    num_train: int,
) -> tf.data.Dataset:
    """Creates a ``tf.data.Dataset``."""
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types={k: tf.string for k in keys},
        output_shapes={k: tf.TensorShape([]) for k in keys},
    )
    return dataset.take(num_train) if split == "train" else dataset.skip(num_train)
