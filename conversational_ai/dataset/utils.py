"""Dataset utilities."""
from typing import Iterable

import chitchat_dataset as ccc


def convo_as_str(
    convo: Iterable[str],
    prefix: str = "",
    suffix: str = "",
    turn_prefixes: Iterable[str] = ["", ""],  # noqa: B006
    turn_suffix: str = "\n",
) -> str:
    """Returns a conversation as a single string."""
    new_convo = turn_suffix.join(ccc.prepend_cycle(convo, turn_prefixes))
    return f"{prefix}{new_convo}{suffix}"
