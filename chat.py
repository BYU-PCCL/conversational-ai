"""Simple chatbot script to chat with a trained T5 model."""
import os
import readline  # noqa: F401,W0611
import time
from pathlib import Path
from typing import List, Optional, Union

import gin

import t5_model


@gin.configurable
def chat_interactively(
    model_dir: Union[str, Path],
    prefix: str,
    turn_separator: str,
    output_file: Optional[Union[str, Path]] = "./chats/{run}__{timestamp}",
    context_window: int = 100,
    step: Optional[Union[int, str]] = "latest",
    prompt: str = "> ",
) -> List[str]:
    """Runs an interactive chat session with the trained T5 model."""
    if model_dir is None:
        model_dir = gin.query_parameter("utils.run.model_dir")
    model_dir = Path(model_dir)

    if output_file is not None:
        fmt = dict(run=model_dir.name, timestamp=int(time.time()))
        output_file = Path(str(output_file).format(**fmt))
        output_file.parent.mkdir(parents=True, exist_ok=True)

    history: List[str] = []
    try:
        while True:
            inp = input(prompt)
            history.append(inp)

            # TODO: do not hardcode the task & separator tokens
            inputs = [prefix + turn_separator.join(history[-context_window:])]
            predictions = t5_model.predict(inputs, model_dir=str(model_dir), step=step)

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
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    t5_model.init_gin_config()
    _history = chat_interactively()
