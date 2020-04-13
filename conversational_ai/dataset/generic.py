"""Dataset utilities for generic datasets."""
import functools
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, Union

import chitchat_dataset as ccc
import tensorflow.compat.v1 as tf

from conversational_ai.dataset import utils


def _load_dataset(path: Union[str, Path]) -> Iterable[Iterable]:
    for chat in json.load(open(path)):
        dialog = chat.get("dialog", chat.get("dialogue", []))
        yield (msg.get("text") for msg in dialog)


def generate_compounding_conversations(
    path: Union[str, Path], **kwargs
) -> Iterable[Dict[str, str]]:
    """Yields examples from the dataset at ``path`` as a ."""
    kwargs.setdefault("prefix", "prefix: ")
    for convo in _load_dataset(path):
        for inputs, targets in ccc.compound_conversation(convo=convo, **kwargs):
            yield {"inputs": inputs, "targets": targets}


def generate_conversations_as_str(
    path: Union[str, Path], **kwargs
) -> Iterable[Dict[str, str]]:
    """Yields examples from the dataset at ``path`` as strings."""
    for convo in _load_dataset(path):
        yield {"text": utils.convo_as_str(convo=convo, **kwargs)}


def dataset(
    split: str,
    shuffle_files: bool,
    generator: Callable[[], Iterable[Dict[str, str]]],
    keys: Iterator[str],
    data_dir: Union[str, Path],
) -> tf.data.Dataset:
    """Creates a ``tf.data.Dataset``."""
    return tf.data.Dataset.from_generator(
        functools.partial(generator, path=f"{data_dir}/{split}.json"),
        output_types={k: tf.string for k in keys},
        output_shapes={k: tf.TensorShape([]) for k in keys},
    )
