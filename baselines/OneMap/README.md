
# One Map to Find Them All: Real-time Open-Vocabulary Mapping for Zero-shot Multi-Object Navigation 

**Paper Reference:** [Link](https://arxiv.org/abs/2409.11764)

**Author:** Finn Lukas Busch, Timon Homberger, Jesús Ortega-Peimbert, Quantao Yang, Olov Andersson

**Source Repo:** [Link](https://github.com/KTH-RPL/OneMap)


## Setting up the Conda Env

```bash
#Create env
conda create -n personal_onemap python=3.9 cmake=3.14.0 -y
conda activate personal_onemap

#CUDA nvcc
conda install -c nvidia cuda-toolkit=12.6 cuda-nvcc=12.6 -y

#Dependencies
python3 -m pip install gdown torch torchvision torchaudio meson
python3 -m pip install -r requirements.txt

python3 -m pip install --upgrade timm>=1.0.7

#Build planning utils
python3 -m pip install ./planning_cpp/

#Habitat-Sim
conda remove cuda-toolkit -y
conda install habitat-sim=0.2.5 withbullet -c conda-forge -c aihabitat
```

## Setting up the Directory

```bash
#Enter the OneMap directory
cd PersONAL/baselines/OneMap

#Symlink Habitat-Lab
ln -s \<PATH-TO-PersONAL\>/PersONAL/habitat-labs/v0.2.5/habitat-lab habitat-lab
```

## Evaluation

```bash
#Running evaluation on PersONAL
python3 PersONAL_eval_habitat.py \
--config config/mon/PersONAL_eval_conf.yaml  \
--PlanningConf.using_ov
```

## Source Citation

```
@INPROCEEDINGS{11128393,
      author={Busch, Finn Lukas and Homberger, Timon and Ortega-Peimbert, Jesús and Yang, Quantao and Andersson, Olov},
      booktitle={2025 IEEE International Conference on Robotics and Automation (ICRA)}, 
      title={One Map to Find Them All: Real-time Open-Vocabulary Mapping for Zero-shot Multi-Object Navigation}, 
      year={2025},
      volume={},
      number={},
      pages={14835-14842},
      keywords={Training;Three-dimensional displays;Uncertainty;Navigation;Semantics;Benchmark testing;Search problems;Probabilistic logic;Real-time systems;Videos},
      doi={10.1109/ICRA55743.2025.11128393},
}
```
