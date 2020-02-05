import os
import re
from pathlib import Path

import chitchat_dataset as ccc


def _prep_convo(
    convo,
    start_token="<|startoftext|>",
    end_token="<|endoftext|>",
    # max_utterance_length=int(sys.argv[1]) if len(sys.argv) > 1 else 0,
):
    convo = ["> " + txt if i % 2 == 0 else txt for i, txt in enumerate(convo)]
    return "{}\n{}\n{}".format(start_token, "\n".join(convo), end_token)


def write_to_file(path="train.txt", daily_dialog_path="dialogues_text.txt"):
    dataset = []

    try:
        for line in Path(daily_dialog_path).open().readlines():
            dataset.append(_prep_convo(re.split(r"\s*__eou__\s*", line)))
    except FileNotFoundError:
        pass

    for convo in ccc.Dataset().values():
        convo = _prep_convo(" ".join(u["text"] for u in m) for m in convo["messages"])
        dataset.append(convo)

    Path(path).write_text("\n".join(dataset))

    return path


if __name__ == "__main__":
    import os

    write_to_file(os.getenv("TRAIN_FILE", "train.txt"))
