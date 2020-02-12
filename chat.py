#!/usr/bin/env python3

import contextlib
import os
import readline
import sys
import time
from pathlib import Path

from halo import Halo as Spinner


# TODO: this should not handle saving/updating the conversation
def chatbot(checkpoint, output_dir=None, length=128, **kwargs):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"
    import tensorflow as tf
    import gpt_2_simple as gpt2

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    checkpoint = Path(checkpoint)
    checkpoint_dir = checkpoint.parent.resolve()
    run_name = checkpoint.name

    output_file = None
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = Path(output_dir, f"{run_name}_{int(time.time())}")

    with Spinner(f"Loading model from {checkpoint}...", stream=sys.stderr):
        sess = gpt2.start_tf_sess()

        # HACK: avoid gpt2's unecessary printing that messes with our spinner...
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            gpt2.load_gpt2(sess, checkpoint_dir=checkpoint_dir, run_name=run_name)

    def _chat(user_input, conversation="", prompt="> "):
        conversation += f"{prompt}{user_input}\n"

        output = gpt2.generate(
            sess,
            checkpoint_dir=checkpoint_dir,
            run_name=run_name,
            prefix=conversation,
            return_as_list=True,
            length=length,
            nsamples=kwargs.get("batch_size", 1),
            truncate=prompt,
            include_prefix=False,
            **kwargs,
        )

        # TODO: handle nsamples > 1
        output = output[0]

        conversation += output

        if output_file:
            output_file.write_text(conversation)

        return output, conversation

    return _chat


def _run_interactive_chat(**kwargs):
    chat = chatbot(**kwargs)

    prompt = "> "
    conversation = ""
    try:
        while True:
            # TODO: why do we need an extra space for the prompt?
            user_input = input(prompt).strip()

            with Spinner("Thinking..."):
                output, conversation = chat(user_input, conversation, prompt=prompt)

            print(output, end="")
    except (KeyboardInterrupt, EOFError):
        # do not print exception
        sys.exit(0)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="chat with the model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ckpt = Path(os.getenv("CONVERSATIONAL_AI_CHECKPOINT_DIR", "checkpoint"))
    parser.add_argument(
        "-c",
        "--checkpoint",
        help="The directory of the model checkpoint",
        type=Path,
        default=max(
            (p for p in ckpt.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime,
        ),
    )

    parser.add_argument(
        "-b", "--batch-size", help="Batch size", default=1, type=int,
    )

    parser.add_argument(
        "-l",
        "--length",
        help="Length (number of tokens) of the generated texts",
        default=128,
        type=int,
    )

    parser.add_argument(
        "--output-dir",
        help="Also save chat under OUTPUT_DIR, with the same name as CHECKPOINT",
        type=Path,
        default=Path("./chats/"),
    )

    _run_interactive_chat(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
