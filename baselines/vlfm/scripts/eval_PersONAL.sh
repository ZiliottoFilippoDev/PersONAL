#!/usr/bin/env bash
# Copyright [2023] Boston Dynamics AI Institute, Inc.

# Ensure you have 'export VLFM_PYTHON=<PATH_TO_PYTHON>' in your .bashrc, where
# <PATH_TO_PYTHON> is the path to the python executable for your conda env
# (e.g., PATH_TO_PYTHON=`conda activate <env_name> && which python`)


export PERSONAL_PYTHON='/mnt/anaconda3/envs/personal_vlfm/bin/python'

easy_data_path="data/datasets/PersONAL/active_new/val/test_baselines/easy/easy.json.gz"
medium_data_path="data/datasets/PersONAL/active_new/val/test_baselines/medium/medium.json.gz"
hard_data_path="data/datasets/PersONAL/active_new/val/test_baselines/hard/hard.json.gz"

session_name=eval_personal

# Create a detached tmux session
tmux new-session -d -s ${session_name}

# Split the window vertically
tmux split-window -v -t ${session_name}:0

# Run commands in each pane
tmux send-keys -t ${session_name}:0.0 \
    "${PERSONAL_PYTHON} -m vlfm.run \
        PersONAL_args.log_dir=log/easy \
        habitat.dataset.data_path=${easy_data_path}" C-m

tmux send-keys -t ${session_name}:0.1 \
    "${PERSONAL_PYTHON} -m vlfm.run \
        PersONAL_args.log_dir=log/medium \
        habitat.dataset.data_path=${medium_data_path}" C-m


# Attach to the tmux session to view the windows
echo "Created tmux session '${session_name}'"
echo "Run the following to monitor all the server commands:"
echo "tmux attach-session -t ${session_name}"



























# export PERSONAL_PYTHON='/mnt/anaconda3/envs/personal_vlfm/bin/python'

# session_name=eval_personal

# # Create a detached tmux session
# tmux new-session -d -s ${session_name}

# # Split the window vertically
# tmux split-window -v -t ${session_name}:0

# # Run commands in each pane
# tmux send-keys -t ${session_name}:0.0 \
# " for scene in \
#     'bCPU9suPUw9' 'GLAQ4DNUx5U' 'q5QZSEeHe5g' 'MHPLjHsuG27' \
#     'Nfvxx8J5NCo' 'svBbv1Pavdk' 'LT9Jq6dN3Ea' 'qyAac8rV8Zk' \
#     'mv2HUxq3B53' '5cdEh9F2hJL' 'ziup5kvtCCR' 'y9hTuugGdiq' \
#     'DYehNKdT76V' 'Dd4bFSTQ8gi' 'BAbdmeyTvMZ' '6s7QHgap2fW'; \
# do \
#     ${PERSONAL_PYTHON} -m vlfm.run \
#         PersONAL_args.log_dir=log/easy \
#         habitat.dataset.content_scenes=['bCPU9suPUw9']; \
# done" C-m

# tmux send-keys -t ${session_name}:0.1 \
# " for scene in \
#     'zt1RVoi7PcG' 'VBzV5z6i1WS' '4ok3usBNeis' 'CrMo8WxCyVb' \
#     'cvZr5TUy5C5' 'a8BtkwhxdRV' 'bxsVRursffK' 'QaLdnwvtxbs' \
#     'p53SfW6mjZe' 'q3zU7Yy5E5s' 'mL8ThkuaVTM' 'TEEsavR23oF' \
#     'h1zeeAwLh9Z' 'wcojb4TFT35' 'XB4GS9ShBRE';               \
# do \
#     ${PERSONAL_PYTHON} -m vlfm.ru \
#         PersONAL_args.log_dir=log/easy \
#         habitat.dataset.content_scenes=[\$scene]; \
# done" C-m


# # Attach to the tmux session to view the windows
# echo "Created tmux session '${session_name}'"
# echo "Run the following to monitor all the server commands:"
# echo "tmux attach-session -t ${session_name}"
