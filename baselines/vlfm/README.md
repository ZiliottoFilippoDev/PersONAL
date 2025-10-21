<p align="center">
  <img src="docs/teaser_v1.jpg" width="700">
  <h1 align="center">VLFM: Vision-Language Frontier Maps for Zero-Shot Semantic Navigation</h1>
  <h3 align="center">
    <a href="http://naoki.io/">Naoki Yokoyama</a>, <a href="https://faculty.cc.gatech.edu/~sha9/">Sehoon Ha</a>, <a href="https://faculty.cc.gatech.edu/~dbatra/">Dhruv Batra</a>, <a href="https://www.robo.guru/about.html">Jiuguang Wang</a>, <a href="https://bucherb.github.io">Bernadette Bucher</a>
  </h3>
  <p align="center">
    <a href="http://naoki.io/portfolio/vlfm.html">Project Website</a> , <a href="https://arxiv.org/abs/2312.03275">Paper (arXiv)</a>
  </p>
  <p align="center">
    <a href="https://github.com/bdaiinstitute/vlfm">
      <img src="https://img.shields.io/badge/License-MIT-yellow.svg" />
    </a>
    <a href="https://www.python.org/">
      <img src="https://img.shields.io/badge/built%20with-Python3-red.svg" />
    </a>
    <a href="https://github.com/jiuguangw/Agenoria/actions">
      <img src="https://github.com/bdaiinstitute/vlfm/actions/workflows/test.yml/badge.svg">
    </a>
    <a href="https://github.com/psf/black">
      <img src="https://img.shields.io/badge/code%20style-black-000000.svg">
    </a>
    <a href="https://github.com/astral-sh/ruff">
      <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json">
    </a>
    <a href="https://github.com/python/mypy">
      <img src="http://www.mypy-lang.org/static/mypy_badge.svg">
    </a>
  </p>
</p>

## :sparkles: Overview

Understanding how humans leverage semantic knowledge to navigate unfamiliar environments and decide where to explore next is pivotal for developing robots capable of human-like search behaviors. We introduce a zero-shot navigation approach, Vision-Language Frontier Maps (VLFM), which is inspired by human reasoning and designed to navigate towards unseen semantic objects in novel environments. VLFM builds occupancy maps from depth observations to identify frontiers, and leverages RGB observations and a pre-trained vision-language model to generate a language-grounded value map. VLFM then uses this map to identify the most promising frontier to explore for finding an instance of a given target object category. We evaluate VLFM in photo-realistic environments from the Gibson, Habitat-Matterport 3D (HM3D), and Matterport 3D (MP3D) datasets within the Habitat simulator. Remarkably, VLFM achieves state-of-the-art results on all three datasets as measured by success weighted by path length (SPL) for the Object Goal Navigation task. Furthermore, we show that VLFM's zero-shot nature enables it to be readily deployed on real-world robots such as the Boston Dynamics Spot mobile manipulation platform. We deploy VLFM on Spot and demonstrate its capability to efficiently navigate to target objects within an office building in the real world, without any prior knowledge of the environment. The accomplishments of VLFM underscore the promising potential of vision-language models in advancing the field of semantic navigation.

Source Repo : [https://github.com/bdaiinstitute/vlfm](https://github.com/bdaiinstitute/vlfm)

## Setting up the directory

To test the VLFM baseline on the PersONAL dataset, follow the instructions below. Before running, we need to set up a few paths and directories (as required by the source repo). 

If you need an overview of the changes made to the source code, please 
refer to the PersONAL_changes.txt file present in the vlfm directory.

### Create habitat-lab directory

```bash
#Enter the VLFM directory
cd vlfm

#Symlink habitat-lab (present in parent dir)
ln -s \<PATH-TO-PersONAL\>/PersONAL/habitat-lab habitat-lab
```

### Clone required repositories (as instructed in the source)

```bash

#GroundingDINO
git clone https://github.com/IDEA-Research/GroundingDINO

#MobileSAM
git clone https://github.com/ChaoningZhang/MobileSAM

#Yolov7
git clone https://github.com/WongKinYiu/yolov7

#depth_camera_filtering
git clone https://github.com/naokiyokoyama/depth_camera_filtering
mv depth_camera_filtering depth_camera_filtering_parent
mv depth_camera_filtering_parent/depth_camera_filtering depth_camera_filtering
rm -rf depth_camera_filtering_parent

#frontier_exploration
git clone https://github.com/naokiyokoyama/frontier_exploration
mv frontier_exploration frontier_exploration_parent
mv frontier_exploration_parent/frontier_exploration frontier_exploration
rm -rf frontier_exploration_parent
```

### Download model weights

The weights for MobileSAM, GroundingDINO, and Yolov7 must be saved to the `data/` directory. The weights can be downloaded from the following links:

  - `groundingdino_swint_ogc.pth` : [https://github.com/IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
  - `mobile_sam.pt` : [https://github.com/ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
  - `yolov7-e6e.pt` : [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)

## Setting up the Conda Env

```bash
#Create Conda env
conda create -n vlfm_query_gui python=3.9 cmake=3.14.0
conda activate vlfm_query_gui

#Setting up NVCC and CUDA Toolkit
conda install nvidia/label/cuda-11.7.0::cuda-nvcc -y

#Torch Installation
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

#Habitat Installation
#For habitat-lab, ensure the symlink is active (see previous section)
conda install habitat-sim=0.2.5 withbullet -c conda-forge -c aihabitat
cd habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines

#Lavis Installation
pip install salesforce-lavis

#Other Installations
pip install Flask open3d


# --- Check if installations are correct ----
# Start python in terminal
import lavis
import cv2
```

<details>
<summary>Debugging common installation bugs</summary>

<details>
<summary>Error: LayerId = cv2.dnn.DictValue</summary>

Reference : https://github.com/facebookresearch/nougat/issues/40  
Solution : Comment out the line containing `LayerId = cv2.dnn.DictValue` from the source `__init__.py` file.
</details>  

<details>
<summary>Error Message: GL::Context: cannot retrieve OpenGL version: GL::Renderer::Error::InvalidValue</summary>

Reference : https://github.com/facebookresearch/habitat-sim/pull/2519  
Solution : `conda remove libva libgl libglx libegl libglvnd`
</details>  

</details>


## Evaluation on PersONAL

For evaluation, make sure the current working directory is in `PersONAL/baselines/vlfm/`.

```bash
#Launch the models
./scripts/launch_vlm_servers.sh

#Run (evaluate) on PersONAL
python -m vlfm.run PersONAL_args.log_dir=log/junk

#Read the results from the evaluation runs
python -m read_results --log_dir log/junk/ --PersONAL_data_type easy
```


## :newspaper: Source License

VLFM is released under the [MIT License](LICENSE). This code was produced as part of Naoki Yokoyama's internship at the Boston Dynamics AI Institute in Summer 2023 and is provided "as is" without active maintenance. For questions, please contact [Naoki Yokoyama](http://naoki.io) or [Jiuguang Wang](https://www.robo.guru).

## :black_nib: Source Citation

If you use VLFM in your research, please use the following BibTeX entry.

```
@inproceedings{yokoyama2024vlfm,
  title={VLFM: Vision-Language Frontier Maps for Zero-Shot Semantic Navigation},
  author={Naoki Yokoyama and Sehoon Ha and Dhruv Batra and Jiuguang Wang and Bernadette Bucher},
  booktitle={International Conference on Robotics and Automation (ICRA)},
  year={2024}
}
```
