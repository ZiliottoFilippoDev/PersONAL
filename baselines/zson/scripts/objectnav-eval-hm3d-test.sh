#!/bin/bash

# --------------------------
# Environment setup
# --------------------------

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

# Activate your conda environment
# source /mnt/anaconda3/bin/activate zson
# (Adjust path if needed — see note below)

# --------------------------
# Your variables
# --------------------------
CKPT_DIR="data/checkpoints/zson_conf_B.pth"
DATA_PATH="data/datasets/hm3d/v1/{split}/{split}.json.gz"

# --------------------------
# Run the Python command
# --------------------------

set -x

python -u run.py \
  --exp-config configs/experiments/objectnav_mp3d.yaml \
  --run-type eval \
  TASK_CONFIG.TASK.SENSORS '["OBJECTGOAL_PROMPT_SENSOR"]' \
  TASK_CONFIG.TASK.MEASUREMENTS '["DISTANCE_TO_GOAL", "SUCCESS", "SPL", "SOFT_SPL", "AGENT_ROTATION", "AGENT_POSITION"]' \
  EVAL_CKPT_PATH_DIR $CKPT_DIR \
  EVAL.SPLIT "val" \
  NUM_ENVIRONMENTS 20 \
  TASK_CONFIG.DATASET.DATA_PATH $DATA_PATH \
  RL.POLICY.pretrained_encoder 'data/models/omnidata_DINO_02.pth' \
  RL.REWARD_MEASURE "distance_to_goal" \
  RL.POLICY.CLIP_MODEL "RN50" \
  EVAL.episodes_eval_data True
