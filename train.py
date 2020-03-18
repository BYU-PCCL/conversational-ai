import os
from functools import partial

import gin
# import t5
import t5.models.mesh_transformer as t5_mtf
import tensorflow.compat.v1 as tf
from mesh_tensorflow.transformer import utils

FINETUNE_STEPS = 25000
MODEL_DIR = "./models"

tf.disable_v2_behavior()
# tf.compat.v1.enable_eager_execution()
# gin.parse_config_file(os.path.join(MODEL_DIR, "operative_config.gin"))
utils.parse_gin_defaults_and_flags()


ckpt = tf.train.latest_checkpoint(MODEL_DIR)
if ckpt is None:
    raise ValueError("No checkpoints found in model directory: %s", MODEL_DIR)

checkpoint_step = int(re.sub(".*ckpt-", "", ckpt))
model_ckpt = "model.ckpt-" + str(checkpoint_step)
init_checkpoint = os.path.join(model.PRETRAINED_DIR, model_ckpt)

# vocabulary = t5.data.get_mixture_or_task(mixture_or_task_name).get_vocabulary()


utils.run(
    tpu_job_name=None,
    tpu=None,
    gcp_project=None,
    tpu_zone=None,
    model_dir="./models",
    # init_checkpoint=init_checkpoint,
    # vocabulary=vocabulary,
    train_dataset_fn=partial(t5_mtf.tsv_dataset_fn, filename="./train.tsv"),
    train_steps=(checkpoint_step + FINETUNE_STEPS),
)
