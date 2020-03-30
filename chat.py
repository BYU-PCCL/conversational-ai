"""Simple chatbot script to chat with a trained T5 model."""
import logging
import os
import readline  # noqa: F401,W0611
import time
from pathlib import Path
from typing import List, Optional, Union


def interactive(
    checkpoint: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
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
            model_input = [f"conversation: " + "<TURN>".join(history)]
            predictions = m.predict(model_input, temperature=0.0)

            prediction = "\n".join(predictions)
            history.append(prediction)
            print(prediction)
            if output_file:
                output_file.write_text(f"{prompt}{inp}\n{prediction}")
    except (KeyboardInterrupt, EOFError):
        # do not print a traceback
        return history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="chat with the model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    chkpt: Union[str, Path, None] = os.getenv("CONVERSATIONAL_AI_MODEL_DIR")
    if not chkpt:
        chkpt = max(
            (p for p in Path("./checkpoints/").iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
        )

    parser.add_argument(
        "-c",
        "--checkpoint",
        help="The directory of the model checkpoint",
        type=Path,
        default=chkpt,
    )

    parser.add_argument(
        "--output-dir",
        help="Also save chat under OUTPUT_DIR, with the same name as CHECKPOINT",
        type=Path,
        default=Path(os.getenv("CONVERSATIONAL_AI_CHATS_DIR", "./chats/")),
    )

    _history = interactive(**vars(parser.parse_args()))
