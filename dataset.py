"""Prepares a dataset for consumption by the T5 model.

https://github.com/google-research/text-to-text-transfer-transformer
"""

import re
from pathlib import Path
from typing import Iterable, List, Optional, Union

import chitchat_dataset as ccc


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


def write_to_file(
    path: Union[str, Path] = "train.tsv",
    daily_dialog_path: Optional[Union[str, Path]] = None,
) -> Union[str, Path]:
    """Writes the CCC (and optionally the daily dialog) dataset to a file."""
    dataset = []

    if daily_dialog_path:
        for line in Path(daily_dialog_path).open().readlines():
            dataset.extend(_prep_convo(re.split(r"\s*__eou__\s*", line)))

    for convo in ccc.Dataset().values():
        convo = _prep_convo(" ".join(u["text"] for u in m) for m in convo["messages"])
        dataset.extend(convo)

    Path(path).write_text("\n".join(dataset))  # type: ignore

    return path


if __name__ == "__main__":
    import os

    write_to_file(os.getenv("TRAIN_FILE", "train.tsv"), os.getenv("DAILY_DIALOG_PATH"))
