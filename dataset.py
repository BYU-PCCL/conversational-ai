"""Prepares a dataset for consumption by the T5 model.

https://github.com/google-research/text-to-text-transfer-transformer
"""

import os
import random
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import chitchat_dataset as ccc

_Path = Union[str, Path]


def _prep_convo(convo: Iterable[str]) -> List[str]:
    result = []

    convo = list(c for c in convo if c.strip())
    for i in range(1, len(convo)):
        txt = f"conversation: {'<TURN>'.join(convo[:i])}\t{convo[i]}"

        txt = txt.replace(" ,", ",").replace(" .", ".").replace(" ?", "?")
        txt = txt.replace(" 's", "'s").replace("s ' ", "s' ")
        txt = txt.replace(" ' ", "'").replace(" ’ ", "'")

        result.append(txt)

    return result


def write_to_files(
    train_path: _Path = "train.tsv",
    validation_path: _Path = "validation.tsv",
    daily_dialog_path: Optional[_Path] = os.getenv("DAILY_DIALOG_PATH"),  # noqa: B008
    train_ratio: float = 0.9,
) -> Tuple[_Path, _Path]:
    """Write the CCC (& optionally daily dialog) dataset(s) to train/validation files."""
    dataset = []

    if daily_dialog_path:
        for line in Path(daily_dialog_path).open().readlines():
            dataset.extend(_prep_convo(re.split(r"\s*__eou__\s*", line)))

    for convo in ccc.Dataset().values():
        convo = _prep_convo(" ".join(u["text"] for u in m) for m in convo["messages"])
        dataset.extend(convo)

    random.shuffle(dataset)

    train_idx = int(train_ratio * len(dataset))
    Path(train_path).write_text("\n".join(dataset[:train_idx]))  # type: ignore
    Path(validation_path).write_text("\n".join(dataset[train_idx:]))  # type: ignore

    return train_path, validation_path


if __name__ == "__main__":
    import os

    write_to_files(
        os.getenv("CONVERSATIONAL_AI_TRAIN_PATH", "train.tsv"),
        os.getenv("CONVERSATIONAL_AI_VALIDATION_PATH", "validation.tsv"),
        daily_dialog_path=os.getenv("DAILY_DIALOG_PATH"),
    )
