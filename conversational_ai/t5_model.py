"""Run a T5 model."""
import ast
import datetime
import os
import platform
import tempfile
from pathlib import Path
from typing import List, Optional, Union

import gin
import pkg_resources


def init_gin_config() -> str:
    """Initialize all gin configuration."""
    from mesh_tensorflow.transformer import utils

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

    return gin.config_str()


def run(log_level: str = "INFO", **kwargs) -> None:
    """Run a T5 model for training, finetuning, evaluation etc."""
    import tensorflow.compat.v1 as tf
    from mesh_tensorflow.transformer import utils

    tf.logging.set_verbosity(log_level)
    tf.disable_v2_behavior()
    utils.run(**kwargs)


@gin.configurable
def predict(
    model_input: List[str],
    model_dir: str,
    step: Optional[Union[int, str]] = None,
    log_level: str = "FATAL",
    **kwargs,
) -> List[str]:
    """Get a prediction from the model."""
    from t5.models.mtf_model import _get_latest_checkpoint_from_dir

    # HACK: use `decode` instead of `decode_from_file` (which run uses)
    with tempfile.TemporaryDirectory() as tmp:
        in_file = Path(tmp, "input.txt")
        in_file.write_text("\n".join(model_input))
        out_file = Path(tmp, "output.txt")

        with gin.unlock_config():
            gin.bind_parameter("infer_model.input_filename", str(in_file))
            gin.bind_parameter("infer_model.output_filename", str(out_file))
            gin.bind_parameter("utils.run.mode", "infer")

            if step == -1 or step == "latest":
                step = _get_latest_checkpoint_from_dir(model_dir)
            gin.bind_parameter("utils.run.eval_checkpoint_step", step)

        run(log_level=log_level, **kwargs)

        # will have the checkpoint num appended to it so we glob to get all of them
        all_outputs = [p.read_text() for p in Path(tmp).glob(f"{out_file.name}*")]
        # TODO: should we return just the last one?
        outputs = filter(lambda x: x.strip(), "\n".join(all_outputs).split("\n"))
        # HACK: not sure if the model is outputing `b'output'` or something else is
        return [ast.literal_eval(line.strip()).decode("utf-8") for line in outputs]


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    config = init_gin_config()
    print("# Gin config", "# " + "=" * 78, config, sep="\n")  # TODO: use logging
    run()
