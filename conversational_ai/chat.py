"""Simple chatbot script to chat with a trained T5 model."""
import datetime
import os
import readline  # noqa: F401,W0611
from pathlib import Path
from typing import List, Optional, Union

import gin


@gin.configurable
def chat_interactively(
    model_dir: Union[str, Path],
    prefix: str,
    turn_separator: str,
    output_file: Optional[Union[str, Path]] = "./chats/chat_{timestamp}.txt",
    config_log_file: Optional[Union[str, Path]] = "./chats/chat_{timestamp}.gin",
    context_window: int = 100,
    step: Optional[Union[int, str]] = "latest",
    conversation_length_save_threshold: int = 0,
    prompt: str = "> ",
) -> List[str]:
    """Runs an interactive chat session with the trained T5 model."""
    # (in)directly import tf here so we can set TF_CPP_MIN_LOG_LEVEL in __main__ first
    from conversational_ai import t5_model

    if model_dir is None:
        model_dir = gin.query_parameter("utils.run.model_dir")
    model_dir = Path(model_dir)

    # hardcode the tz for now because some servers are in random timezones
    tz = datetime.timezone(-datetime.timedelta(hours=6))
    fmt = {
        **locals(),
        "run": model_dir.name,
        "step": step,
        "timestamp": datetime.datetime.now(tz=tz).isoformat(),
    }

    if output_file is not None:
        output_file = Path(str(output_file).format(**fmt))
        output_file.parent.mkdir(parents=True, exist_ok=True)

    history: List[str] = []
    try:
        while True:
            inp = input(prompt)
            history.append(inp)

            inputs = [prefix + turn_separator.join(history[-context_window:])]
            predictions = t5_model.predict(inputs, model_dir=str(model_dir), step=step)

            prediction = "\n".join(predictions)
            history.append(prediction)
            print(prediction)
            if output_file and len(history) >= conversation_length_save_threshold:
                output = ""
                for i, turn in enumerate(history):
                    output += f"human: {turn}\n" if i % 2 == 0 else f"model: {turn}\n"
                output_file.write_text(output)
    except (KeyboardInterrupt, EOFError):
        return history  # return without printing traceback
    except Exception:
        raise
    finally:
        if config_log_file and len(history) >= conversation_length_save_threshold:
            config_log_file = Path(str(config_log_file).format(**fmt))
            config_log_file.parent.mkdir(parents=True, exist_ok=True)
            config_log_file.write_text(gin.config_str())  # gin will have been init


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # disable tf C++ logging before importing
    from conversational_ai import t5_model

    t5_model.parse_gin_defaults_and_flags()
    _history = chat_interactively()
