"""Tasks for training a T5 model.

https://github.com/google-research/text-to-text-transfer-transformer
"""
from typing import Callable, Dict, Iterable

import chitchat_dataset as ccc
import t5
import tensorflow.compat.v1 as tf

_Examples = Iterable[Dict[str, str]]


# TODO: do not hardcode these...
_SPLIT_LENGTHS = {
    "conversation_v001_compounding": {"train": 124978, "validation": 6577},
}


def register() -> None:
    """Register a task for use with a T5 model."""
    t5.data.TaskRegistry.add(
        "conversation_v001_compounding",
        t5.data.Task,
        dataset_fn=_compounding_dataset_fn,
        splits=["train", "validation"],
        text_preprocessor=None,
        postprocess_fn=t5.data.postprocessors.lower_text,
        metric_fns=[t5.evaluation.metrics.accuracy, t5.evaluation.metrics.rouge],
        sentencepiece_model_path=t5.data.DEFAULT_SPM_PATH,
        num_input_examples=_SPLIT_LENGTHS["conversation_v001_compounding"],
    )


def _dataset(prep_convo: Callable[[Iterable[str]], _Examples]) -> tf.data.Dataset:
    """Create a tf.data.Dataset of input/target examples.

    Each example has the form: `{"inputs": tf.string, "targets": tf.string}`.
    """

    def _gen() -> _Examples:
        for convo in ccc.ConversationDataset():
            for example in prep_convo(convo):
                yield example

    return tf.data.Dataset.from_generator(
        _gen,
        output_types={"inputs": tf.string, "targets": tf.string},
        output_shapes={"inputs": tf.TensorShape([]), "targets": tf.TensorShape([])},
    )


def _compound_convo(convo: Iterable[str]) -> _Examples:
    """Build a successively longer convo by combining all previous utterances."""
    convo = list(c.strip() for c in convo if c.strip())
    for i in range(1, len(convo)):
        yield {
            "inputs": f"converse: {'<TURN>'.join(convo[:i])}",
            "targets": convo[i],
        }


def _compounding_dataset_fn(split: str, shuffle_files: bool) -> tf.data.Dataset:
    dataset = _dataset(prep_convo=_compound_convo)
    n = _SPLIT_LENGTHS["conversation_v001_compounding"]["train"]
    return dataset.take(n) if split == "train" else dataset.skip(n)
