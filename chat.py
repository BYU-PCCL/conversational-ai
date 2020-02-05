#!/usr/bin/env python3

import os
import readline
from pathlib import Path


def chat(checkpoint, length=128, **kwargs):
    import gpt_2_simple as gpt2

    sess = gpt2.start_tf_sess()

    checkpoint = Path(checkpoint)
    checkpoint_dir = checkpoint.parent.resolve()
    run_name = checkpoint.name

    gpt2.load_gpt2(sess, checkpoint_dir=checkpoint_dir, run_name=run_name)

    try:
        conversation = "<|startoftext|>"
        while True:
            conversation += "> " + input("> ")
            print(conversation)
            output = gpt2.generate(
                sess,
                checkpoint_dir=checkpoint_dir,
                run_name=run_name,
                prefix=conversation,
                return_as_list=True,
                length=length,
                nsamples=kwargs.get("batch_size", 1),
                **kwargs
            )

            output = "\n".join(output)

            print(output)

            conversation += output
    except (KeyboardInterrupt, EOFError):
        # do not print exception
        pass


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

    chat(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
