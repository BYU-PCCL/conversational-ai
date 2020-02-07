#!/usr/bin/env python3

import contextlib
import itertools
import os
import readline
import sys
import threading
import time
from pathlib import Path


def chat(checkpoint, output_dir=None, length=128, **kwargs):
    import gpt_2_simple as gpt2

    checkpoint = Path(checkpoint)
    checkpoint_dir = checkpoint.parent.resolve()
    run_name = checkpoint.name

    output_file = None
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = Path(output_dir, f"{run_name}_{int(time.time())}")

    try:
        with Spinner(f"Loading model from {checkpoint}...", file=sys.stderr):
            sess = gpt2.start_tf_sess()

            # HACK: avoid gpt2's unecessary printing that messes with our spinner...
            with contextlib.redirect_stdout(open(os.devnull, "w")):
                gpt2.load_gpt2(sess, checkpoint_dir=checkpoint_dir, run_name=run_name)

        prompt = "> "
        conversation = ""
        while True:
            # TODO: why do we need an extra space for the prompt?
            conversation += f"{prompt}{input(prompt + ' ').strip()}\n"

            with Spinner("Thinking..."):
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

            print(output, end="")

            conversation += output

            if output_file:
                output_file.write_text(conversation)
    except (KeyboardInterrupt, EOFError):
        # do not print exception
        sys.exit(0)


def main():
    import argparse

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"
    import tensorflow as tf  # isort:skip

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

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
        "-b",
        "--batch-size",
        help="Batch size",
        default=8 if tf.test.is_gpu_available() else 1,
        type=int,
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

    chat(**vars(parser.parse_args()))


class Spinner:
    # TODO: avoid term escape sequences to make this platform independent
    def __init__(
        self,
        message="",
        symbols=["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
        delay=0.09,
        file=sys.stdout,
    ):
        self._spinner = itertools.cycle(symbols)
        self._busy = False
        self._delay = float(delay)
        self._file = file

        message = message + " " if not message.endswith(" ") else message
        # hide cursor; https://stackoverflow.com/a/10455937
        print(message, end="\033[?25l", file=self._file, flush=True)

    def _spinner_task(self):
        while self._busy:
            print(next(self._spinner), end="", file=self._file, flush=True)
            time.sleep(self._delay)
            print("\b", end="", file=self._file, flush=True)

    def __enter__(self):
        self._busy = True
        threading.Thread(target=self._spinner_task).start()

    def __exit__(self, exc_type, exc_val, traceback):
        self._busy = False
        # clear this line and show cursor; https://stackoverflow.com/a/5291396
        print("\033[2K\033[1G", end="\033[?25h", file=self._file, flush=True)


if __name__ == "__main__":
    main()
