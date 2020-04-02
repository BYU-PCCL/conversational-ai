"""Simple chatbot script to chat with a trained T5 model."""
import logging
import os
import readline  # noqa: F401,W0611
import time
from pathlib import Path
from typing import List, Optional, Union


def interactive(
    checkpoint: Union[str, Path],
    output_dir: Optional[Union[str, Path]],
    prefix: str,
    turn_separator: str,
    context_window: int = 100,
    prompt: str = "> ",
) -> List[str]:
    """Runs an interactive chat session with the trained T5 model."""
    checkpoint_dir = Path(checkpoint)
    run_name = checkpoint_dir.name

    output_file = None
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = Path(output_dir, f"{run_name}__{int(time.time())}")

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # set log level to fatal
    os.environ["AUTOGRAPH_VERBOSITY"] = "0"  # turn off AutoGraph logging
    logging.getLogger("tensorflow").disabled = True
    import models  # just in case this or anything else imports tf

    m = models.T5(
        model_dir=str(checkpoint_dir),
        mesh_devices=["gpu:0"],
        mesh_shape="model:1,batch:1",
        tpu=None,
    )

    history: List[str] = []
    try:
        while True:
            inp = input(prompt)  # noqa: S322
            history.append(inp)

            # TODO: do not hardcode the task & separator tokens
            model_input = [prefix + turn_separator.join(history[-context_window:])]
            predictions = m.predict(model_input, temperature=0.0)

            prediction = "\n".join(predictions)
            history.append(prediction)
            print(prediction)
            if output_file:
                output = ""
                for i, turn in enumerate(history):
                    output += f"human: {turn}\n" if i % 2 == 0 else f"model: {turn}\n"
                output_file.write_text(output)
    except (KeyboardInterrupt, EOFError):
        # do not print a traceback
        return history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="chat with the model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        help="The directory containing the model checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        help="Also save chat under OUTPUT_DIR, with the same name as CHECKPOINT",
        type=Path,
        default="./chats/",
    )
    parser.add_argument(
        "--prefix",
        help="The task/prefix to prepend to the start of the conversation",
        default="converse: ",
    )
    parser.add_argument(
        "--turn-separator",
        help="The token to insert between conversation turns",
        default="<TURN>",
    )
    parser.add_argument(
        "--context-window",
        help="Length of the conversation context window",
        type=int,
        default=25,
    )

    args, _extra_args = parser.parse_known_args()

    _history = interactive(**vars(args))
