<p align="center">
  <img src="docs/sys.png" width="900", style="border-radius:10%">
  <h1 align="center">One Map to Find Them All: Real-time Open-Vocabulary Mapping for Zero-shot Multi-Object Navigation</h1>
  <h3 align="center">
    <a href="https://www.kth.se/profile/flbusch?l=en">Finn Lukas Busch</a>,
    <a href="https://www.kth.se/profile/timonh">Timon Homberger</a>,
    <a href="https://www.kth.se/profile/jgop">Jesús Ortega-Peimbert</a>,
    <a href="https://www.kth.se/profile/quantao?l=en">Quantao Yang</a>,
    <a href="https://www.kth.se/profile/olovand" style="white-space: nowrap;"> Olov Andersson</a>
  </h3>
  <p align="center">
    <a href="https://www.finnbusch.com/OneMap/">Project Website</a> , <a href="https://arxiv.org/pdf/2409.11764">Paper (arXiv)</a>
  </p>
</p>
<p align="center">
  <a href="https://github.com/KTH-RPL/OneMap/actions/workflows/docker-build.yml">
    <img src="https://github.com/KTH-RPL/OneMap/actions/workflows/docker-build.yml/badge.svg" alt="Docker Build">
  </a>
</p>

This repository contains the code for the paper "One Map to Find Them All: Real-time Open-Vocabulary Mapping for
Zero-shot Multi-Object Navigation". We provide a [dockerized environment](#setup-docker) to run the code or
you can [run it locally](#setup-local-without-docker).

In summary we open-source:
- The OneMap mapping and navigation code
- The evaluation code for single- and multi-object navigation
- The multi-object navigation dataset and benchmark
- The multi-object navigation dataset generation code, such that you can generate your own datasets

Source Repo : [https://github.com/KTH-RPL/OneMap](https://github.com/KTH-RPL/OneMap)

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
cd baselines/OneMap

#Symlink Habitat-Lab
ln -s \<PATH-TO-PersONAL\>/PersONAL/habitat-lab habitat-lab
```

## Evaluation

```bash
#Running evaluation on PersONAL
python3 PersONAL_eval_habitat.py \
--config config/mon/PersONAL_eval_conf.yaml  \
--PlanningConf.using_ov

#Reading results
python3 PersONAL_read_results \
--log_dir results/easy \
--PersONAL_data_type easy
```

## Citation
If you use this code in your research, please cite our paper:
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
