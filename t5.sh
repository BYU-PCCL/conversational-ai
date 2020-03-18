#!/bin/sh

t5_mesh_transformer  \
  --model_dir="${CONVERSATIONAL_AI_MODEL_DIR:-models/}" \
  --t5_tfds_data_dir="${CONVERSATIONAL_AI_DATA_DIR:-data/}" \
  --gin_file="dataset.gin" \
  --gin_file="gs://t5-data/pretrained_models/small/operative_config.gin" \
  --gin_param="utils.run.mesh_shape = 'model:1,batch:1'" \
  --gin_param="utils.run.mesh_devices = ['gpu:0']" \
  --gin_param="run.train_steps = 2000000" \
  --gin_file="learning_rate_schedules/constant_0_001.gin" \
  --gin_param="utils.run.train_dataset_fn = @t5.models.mesh_transformer.tsv_dataset_fn" \
  --gin_param="tsv_dataset_fn.filename = 'train.tsv'" \
  "$@"
