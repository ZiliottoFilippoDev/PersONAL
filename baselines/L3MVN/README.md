# L3MVN: Leveraging Large Language Models for Visual Target Navigation

**Paper Reference:** [Link](https://arxiv.org/abs/2304.05501)
**Author:** Bangguo Yu, Hamidreza Kasaei and Ming Cao
**Source Repo:** [Link](https://github.com/ybgdgh/L3MVN)

## Create habitat-lab directory

Make sure that the version of Habitat is `challenge-2022`. To install and make it compatible with PersONAL, please follow the procedure in the README of PersONAL. Please make sure the installation is done inside the directory `PersONAL/habitat-labs/challenge-2022/`.

```bash
#Enter the ZSON directory
cd PersONAL/baselines/L3MVN

#Symlink habitat-lab (present in parent dir)
ln -s PersONAL/habitat-labs/challenge-2022/habitat-lab habitat-lab
```

## Installation

```bash
#Create conda env
conda create -n personal_l3mvn_og python=3.9 cmake=3.14.0 -y
conda activate personal_l3mvn_og

#Install CUDA, Torch
conda install pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.3 -c pytorch -c conda-forge

#Install Habitat-Sim
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim; git checkout tags/challenge-2022; 
pip install -r requirements.txt 
python setup.py install --headless
cd ..

#Install Habitat-Lab
cd habitat-lab
pip install -e .
cd ..

#From detectron's installation guide. Follows the suggested versions (see section Install Pre-Built Detectron2 (Linux only))
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

#After taking off all the version specifications in the requirements.txt file
pip install -r requirements.txt
```

## Set up data

```bash

#Symlink data
ln -s habitat-lab/data/scene_datasets data/scene_datasets
ln -s habitat-lab/data/datasets/PersONAL/active data/objectgoal_PersONAL

```

## Evaluation: 

For the config file, refer to files under the path `L3MVN/envs/habitat/configs/tasks`.

```bash

python main_llm_zeroshot_personal.py \
--split val \
--eval 1 \
--auto_gpu_config 0 \
-n 1 \
--num_eval_episodes 2000 \
--load pretrained_models/llm_model.pt \
--use_gtsem 0  \
--task_config tasks/objectnav_personal.yaml \
--log_dir logs/easy
```
