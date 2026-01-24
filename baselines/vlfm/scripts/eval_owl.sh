#!/usr/bin/env bash
# Copyright [2023] Boston Dynamics AI Institute, Inc.

# Ensure you have 'export VLFM_PYTHON=<PATH_TO_PYTHON>' in your .bashrc, where
# <PATH_TO_PYTHON> is the path to the python executable for your conda env
# (e.g., PATH_TO_PYTHON=`conda activate <env_name> && which python`)


export PERSONAL_PYTHON='/mnt/anaconda3/envs/vlfm_query_blip/bin/python'

easy_data_path="data/datasets/PersONAL/active/val/baselines/owl/easy/easy.json.gz"
medium_data_path="data/datasets/PersONAL/active/val/baselines/owl/medium/medium.json.gz"
hard_data_path="data/datasets/PersONAL/active/val/baselines/owl/hard/hard.json.gz"

session_name=eval_vlfm_owl

# Create a detached tmux session
tmux new-session -d -s ${session_name}

# Split the window vertically
tmux split-window -v -t ${session_name}:0

# Run commands in each pane
tmux send-keys -t ${session_name}:0.0 \
    "${PERSONAL_PYTHON} -m vlfm.run \
        PersONAL_args.log_dir=log/owl/easy \
        habitat.dataset.data_path=${easy_data_path} \
        habitat_baselines.rl.policy.name=HabitatITMPolicy_owlv2" C-m

tmux send-keys -t ${session_name}:0.1 \
    "${PERSONAL_PYTHON} -m vlfm.run \
        PersONAL_args.log_dir=log/owl/medium \
        habitat.dataset.data_path=${medium_data_path} \
        habitat_baselines.rl.policy.name=HabitatITMPolicy_owlv2" C-m


# Attach to the tmux session to view the windows
echo "Created tmux session '${session_name}'"
echo "Run the following to monitor all the server commands:"
echo "tmux attach-session -t ${session_name}"


