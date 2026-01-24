#!/bin/bash

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

CKPT_DIR="data/checkpoints/zson_conf_B.pth"

# DATA_PATH="data/datasets/PersONAL/val/baselines/standard/easy/easy.json.gz"
# LOG_DIR="logs/PersONAL_v2/easy"

# DATA_PATH="data/datasets/PersONAL/val/baselines/standard/medium/medium.json.gz"
# LOG_DIR="logs/PersONAL_v2/medium"

DATA_PATH="data/datasets/PersONAL/val/baselines/standard/hard/hard.json.gz"
LOG_DIR="logs/PersONAL_v2/hard"

set -x

python -u run.py \
    --exp-config configs/experiments/objectnav_personal.yaml \
    --run-type eval \
    TASK_CONFIG.TASK.SENSORS '["OBJECTGOAL_PROMPT_SENSOR"]' \
    TASK_CONFIG.TASK.MEASUREMENTS '["DISTANCE_TO_GOAL", "SUCCESS", "SPL", "SOFT_SPL", "AGENT_ROTATION", "AGENT_POSITION"]' \
    EVAL_CKPT_PATH_DIR $CKPT_DIR \
    EVAL.SPLIT "val" \
    TASK_CONFIG.DATASET.DATA_PATH $DATA_PATH \
    RL.POLICY.pretrained_encoder 'data/models/omnidata_DINO_02.pth' \
    RL.REWARD_MEASURE "distance_to_goal" \
    RL.POLICY.CLIP_MODEL "RN50" \
    EVAL.episodes_eval_data True \
    NUM_ENVIRONMENTS 1 \
    LOG_DIR $LOG_DIR \
    # TASK_CONFIG.DATASET.CONTENT_SCENES '["wcojb4TFT35"]'
    # VIDEO_OPTION '["disk"]'