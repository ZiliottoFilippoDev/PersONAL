# ZSON: Zero-Shot Object-Goal Navigation using Multimodal Goal Embeddings

**Paper Reference:** [Link](https://arxiv.org/abs/2206.12403)

**Author:** Arjun Majumdar*, Gunjan Aggarwal*, Bhavika Devnani, Judy Hoffman and Dhruv Batra

**Source Repo:** [Link](https://github.com/gunagg/zson)

## Create habitat-lab directory

Make sure that the version of Habitat is `challenge-2022`. To install and make it compatible with PersONAL, please follow the procedure in the README of PersONAL. Please make sure the installation is done inside the directory `PersONAL/habitat-labs/challenge-2022/`.

```bash
#Enter the ZSON directory
cd PersONAL/baselines/zson

#Symlink habitat-lab (present in parent dir)
ln -s \<PATH-TO-PersONAL\>/PersONAL/habitat-labs/challenge-2022/habitat-lab habitat-lab
```

## Create dataset directory

Symlink to PersONAL dataset.

```bash
mkdir -p data/datasets
ln -s PersONAL/data data/datasets/PersONAL/val
```

## Installation

```bash

#Create conda env
conda create -n personal_zson python=3.9 cmake=3.14.0 -y
conda activate personal_zson

#CUDA, Torch
conda install pytorch==1.10.2 torchvision==0.11.3 cudatoolkit=11.3 -c pytorch -c conda-forge

#Habitat-Sim
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim; git checkout tags/challenge-2022; 
pip install -r requirements.txt 
python setup.py install --headless
cd ..

#Habitat-Lab
cd habitat-lab
pip install -e .
cd ..

#ZSON requirements
pip install -r requirements.txt
python setup.py develop
```

## Download weights

All the required data can be downloaded from [here](https://huggingface.co/gunjan050/ZSON/tree/main).

The following trained checkpoints are to be downloaded into the directory `data/checkpoints`:

  - [`zson_conf_B.pth`](https://huggingface.co/gunjan050/ZSON/resolve/main/zson_conf_B.pth)


Download the models weights into `data/models/`:

   - [omnidata_DINO_02.pth](https://huggingface.co/gunjan050/ZSON/resolve/main/omnidata_DINO_02.pth)


## Evaluation


  ```
  sbatch scripts/objnav-eval-personal.sh
  ```

## Source Citation

```
@inproceedings{majumdar2022zson,
  title={ZSON: Zero-Shot Object-Goal Navigation using Multimodal Goal Embeddings},
  author={Majumdar, Arjun and Aggarwal, Gunjan and Devnani, Bhavika and Hoffman, Judy and Batra, Dhruv},
  booktitle={Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```
