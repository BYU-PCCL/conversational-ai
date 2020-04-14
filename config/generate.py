"""Generates a gin config from the current task/mixture list.

Usage: `python3 -m config.generate`
"""
from itertools import chain, product
from pathlib import Path

import t5

import conversational_ai.tasks  # noqa: F401

WHITELIST = ["chitchat", "dailydialog", "convai2"]

sizes = ["small", "base", "large", "3b", "11b"]
mixtures = filter(
    lambda task: any(name in task for name in WHITELIST),
    chain(t5.data.TaskRegistry.names(), t5.data.MixtureRegistry.names()),
)

for size, mixture in product(sizes, mixtures):
    path = Path(f"./config/mixtures/{mixture}/{size}.gin")

    print(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    body = """include "finetune_{size}.gin"

MIXTURE_NAME = "{mixture}"

utils.run.model_dir = "./checkpoints/conversational-ai/{mixture}/{size}"
""".format(
        size=size, mixture=mixture
    )

    path.write_text(body)
