"""Utilities for working with the T5 model.

https://arxiv.org/abs/1910.10683
"""
import argparse
import ast
import datetime
import logging
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import gin
import pkg_resources
import tensorflow.compat.v1 as tf
from mesh_tensorflow.transformer import utils
from t5.models.mtf_model import _get_latest_checkpoint_from_dir

# HACK: figure out a better alternative to `RUN_TIMESTAMP` global variable?
# hardcode the tz for now because some servers are in random timezones
_tz = datetime.timezone(-datetime.timedelta(hours=6))
RUN_TIMESTAMP = datetime.datetime.now(tz=_tz).isoformat(timespec="milliseconds")


def run(**kwargs) -> None:
    """Runs a T5 model for training, finetuning, evaluation etc."""
    tf.disable_v2_behavior()

    if gin.query_parameter("utils.run.mode") == "eval":
        # Increase the recursion limit, see: https://github.com/pltrdy/rouge/issues/19
        length = gin.query_parameter("utils.run.sequence_length").get("inputs", 512)
        batch_size = 1024  # TODO: do not hardcode batch_size for recursionlimit calc
        sys.setrecursionlimit(batch_size * length + 10)

    utils.run(**kwargs)


@gin.configurable
def predict(
    model_input: List[str],
    model_dir: str,
    step: Optional[Union[int, str]] = None,
    **kwargs,
) -> List[str]:
    """Gets a prediction from the model."""
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
        # HACK: not sure if the model is outputting `b'output'` or something else is
        return [ast.literal_eval(line.strip()).decode("utf-8") for line in outputs]


@gin.configurable
def logging_file_handler(filename: str, **kwargs) -> logging.FileHandler:
    """Returns a ``logging.FileHandler``."""
    return logging.FileHandler(filename.format(timestamp=RUN_TIMESTAMP), **kwargs)


@gin.register
def logging_filter_log_records_for_chat(record: Any) -> bool:
    """Filters log records suitable for interactive chat."""
    return (
        record.msg.startswith("decoded")
        or record.msg.startswith("            ->")
        or record.msg.startswith("Restoring parameters from")
    )


@gin.configurable
def tf_logging(
    level: str = "INFO",
    filters: Optional[List[Callable[[Any], bool]]] = None,
    additional_handlers: Optional[List[Any]] = None,
) -> None:
    """Initializes Tensorflow logging."""
    tf.get_logger().propagate = False  # https://stackoverflow.com/a/33664610
    tf.get_logger().setLevel(level)
    for h in additional_handlers or []:
        tf.get_logger().addHandler(h)
    for f in filters or []:
        tf.get_logger().addFilter(f)


def parse_gin_defaults_and_flags() -> None:
    """Parses all default gin files and those provided via flags."""
    args = _parse_args()  # TODO: should we require that args be passed in?

    for path in [
        *(args.gin_location_prefix or []),
        Path(__file__).parent.parent.joinpath("config"),
        pkg_resources.resource_filename("t5.models", "gin"),
        pkg_resources.resource_filename("mesh_tensorflow.transformer", "gin"),
    ]:
        gin.add_config_file_search_path(path)

    try:
        # attempt to parse these first so they can be overridden later
        gin.parse_config_file("defaults.gin")
        gin.parse_config_file("operative_config.gin")
    except IOError:
        pass

    gin.parse_config_files_and_bindings(args.gin_file, args.gin_param)

    # make it so we don't have to specify a unique model_dir each time
    model_dir = gin.query_parameter("utils.run.model_dir").format(
        hostname=platform.node(), timestamp=RUN_TIMESTAMP,
    )
    with gin.unlock_config():
        gin.bind_parameter("utils.run.model_dir", model_dir)

    tf_logging()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gin_file",
        action="append",
        help="add a gin config file to parse",
        metavar="PATH",
    )
    parser.add_argument(
        "--gin_param",
        action="append",
        help="add a (properly quoted) gin parameter binding",
        metavar="PARAM",
    )
    parser.add_argument(
        "--gin_location_prefix",
        action="append",
        help="add a directory to search for gin configs in",
        metavar="DIR",
    )

    return parser.parse_args()


if __name__ == "__main__":
    parse_gin_defaults_and_flags()
    run()
