"""Run a T5 model.

https://arxiv.org/abs/1910.10683
"""
import ast
import datetime
import logging as py_logging
import platform
import tempfile
from pathlib import Path
from typing import List, Optional, Union

import gin
import pkg_resources
import tensorflow.compat.v1 as tf
from mesh_tensorflow.transformer import utils
from t5.models.mtf_model import _get_latest_checkpoint_from_dir

logger = py_logging.getLogger("conversational-ai")


@gin.configurable
def logging(level: str = "INFO") -> None:
    """Initializes up logging."""
    py_logging.basicConfig(level=level)
    tf.get_logger().propagate = False  # https://stackoverflow.com/a/33664610
    tf.get_logger().setLevel(level)
    # tf.get_logger().addFilter(lambda record: "deprecat" not in record.msg)


def init_gin_config(log_config: bool = True) -> None:
    """Initialize all gin configuration."""
    gin.add_config_file_search_path(Path(__file__).parent.parent.joinpath("config"))
    gin.add_config_file_search_path(pkg_resources.resource_filename("t5.models", "gin"))

    utils.parse_gin_defaults_and_flags()

    # hardcode the tz for now because some servers are in random timezones
    tz = datetime.timezone(-datetime.timedelta(hours=6))
    # make it so we don't have to specify a unique model_dir each time
    model_dir = gin.query_parameter("utils.run.model_dir").format(
        hostname=platform.node(),
        timestamp=datetime.datetime.now(tz=tz).isoformat(timespec="milliseconds"),
    )

    with gin.unlock_config():
        gin.bind_parameter("utils.run.model_dir", model_dir)

    logging()

    if log_config:
        logger.debug("\n# Gin config\n# %s\n%s", "#" * 78, gin.config_str())


def run(**kwargs) -> None:
    """Run a T5 model for training, finetuning, evaluation etc."""
    tf.disable_v2_behavior()
    utils.run(**kwargs)


@gin.configurable
def predict(
    model_input: List[str],
    model_dir: str,
    step: Optional[Union[int, str]] = None,
    **kwargs,
) -> List[str]:
    """Get a prediction from the model."""
    if step == -1 or step == "latest":
        step = _get_latest_checkpoint_from_dir(model_dir)

    # HACK: use `decode` instead of `decode_from_file` (which `run` uses)
    with tempfile.TemporaryDirectory() as tmp:
        in_file = Path(tmp, "input.txt")
        in_file.write_text("\n".join(model_input))
        out_file = Path(tmp, "output.txt")

        with gin.unlock_config():
            gin.bind_parameter("utils.run.mode", "infer")
            gin.bind_parameter("utils.run.model_dir", model_dir)
            gin.bind_parameter("utils.run.eval_checkpoint_step", step)
            gin.bind_parameter("infer_model.input_filename", str(in_file))
            gin.bind_parameter("infer_model.output_filename", str(out_file))

        run(**kwargs)

        # will have the checkpoint num appended to it so we glob to get all of them
        all_outputs = [p.read_text() for p in Path(tmp).glob(f"{out_file.name}*")]
        # TODO: should we return just the last one?
        outputs = filter(lambda x: x.strip(), "\n".join(all_outputs).split("\n"))
        # HACK: not sure if the model is outputing `b'output'` or something else is
        return [ast.literal_eval(line.strip()).decode("utf-8") for line in outputs]


if __name__ == "__main__":
    init_gin_config()
    run()
