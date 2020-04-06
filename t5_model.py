"""Run a T5 model."""
import datetime
import os
import platform
from pathlib import Path

import gin
import pkg_resources


def init_gin_config() -> None:
    """Initialize all gin configuration."""
    from mesh_tensorflow.transformer import utils

    gin.add_config_file_search_path(Path(__file__).parent.joinpath("config"))
    gin.add_config_file_search_path(pkg_resources.resource_filename("t5.models", "gin"))

    utils.parse_gin_defaults_and_flags()

    # hardcode the tz for now because some servers are in random timezons
    tz = datetime.timezone(-datetime.timedelta(hours=6))
    # make it so we don't have to specify a unique model_dir each time
    model_dir = gin.query_parameter("utils.run.model_dir").format(
        hostname=platform.node(),
        timestamp=datetime.datetime.now(tz=tz).isoformat(timespec="milliseconds"),
    )

    with gin.unlock_config():
        gin.bind_parameter("utils.run.model_dir", model_dir)

        try:
            gin.parse_config_file(Path(model_dir, "operative_config.gin"))
        except OSError:
            pass


def run(log_level: str = "INFO", **kwargs) -> None:
    """Run a T5 model for training, finetuning, evaluation etc."""
    import tensorflow.compat.v1 as tf
    from mesh_tensorflow.transformer import utils

    tf.logging.set_verbosity(log_level)
    tf.disable_v2_behavior()
    utils.run(**kwargs)


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    init_gin_config()
    run()
