# PersONAL: Personalized Object Navigation And Localization

This repository will host the **code** and **dataset** for the paper:

**PersONAL: Towards a Comprehensive Benchmark for Personalized Embodied Agents**

---

## 📄 Paper
- [arXiv preprint](https://arxiv.org/abs/2509.19843)  

<p align="center">
  <img src="assets/teaser.png" alt="PersONAL teaser" width="80%">
</p>

---

## 📢 Updates
- Code and dataset will be released soon. Stay tuned! 🚀  
- Please consider starring ⭐ this repository to receive the latest updates.

---

## Installation

#### Setting up PersONAL Dataset

- cd PersONAL
- Cloning habitat-lab : git clone --branch v0.2.5 https://github.com/facebookresearch/habitat-lab
- Download HM3D data to habitat-lab/data
- Symlink to data content
  - Create directory path in habitat-lab : mkdir -p data/datasets/PersONAL/active/
  - Create SymLink : ln -s /mnt/PersONAL/data/split  habitat-lab/data/datasets/PersONAL/active/val
- Copy Dataset Info : cp habitat-utils/personalized_object_nav_dataset.py habitat-lab/habitat-lab/habitat/datasets/object_nav/
- Copy Task Info : cp habitat-utils/personalized_object_nav_task.py habitat-lab/habitat-lab/habitat/tasks/nav/
- Register Dataset: 
  - cp habitat-utils/register_personalized_dataset.py habitat-lab/habitat-lab/habitat/datasets/object_nav/__init__.py
  - cp habitat-utils/registration.py habitat-lab/habitat-lab/habitat/datasets/registration.py
- If we need to get the current position in the habitat env:
  - cp habitat-utils/RL_Env.py habitat-lab/habitat-lab/habitat/core/env.py

(NEED A TEST TO CONFIRM IF ALL WENT )

#### Testing Baselines : VLFM

- Clone the source code and cd vlfm
- Symlink the habitat-lab dir : ln -s /mnt/PersONAL/habitat-lab habitat-lab
- Follow instructions as found in source:
  - Clone the respective repos (as instructed in the source):
    - GroundingDINO
    - MobileSAM
    - yolov7
    - depth_camera_filtering (https://github.com/naokiyokoyama/depth_camera_filtering/tree/main/depth_camera_filtering)
    - frontier_exploration (https://github.com/naokiyokoyama/frontier_exploration/tree/main/frontier_exploration)

  - Download following weights to the data dir (cd data):
    - groundingdino_swint_ogc.pth
    - mobile_sam.pt
    - yolov7-e6e.pt


  - Changes made:
    - config file (habitat task type, dataset and scene_dataset)
    - vlfm run.py (changed config file path)


#### Testing Baselines : OneMap

(WRITE INTRO ABOUT ONEMAP. WHY WE CHOSE THIS)

Setting up the Env:
- Clone PersONAL
- cd PersONAL
- Clone the OneMap source repo locally and follow source instruction to set up the environment
- Symlink to source habitat-lab containing PersONAL dataset
- Changes
  - Added : eval/dataset_utils -> hm3d_PersONAL_dataset.py
  - Added : config/mon -> PersONAL_eval_conf.yaml
  - Added : PersONAL_eval_habitat.py
  - Added : eval/dataset_utils/__init__.py -> import PersONAL dataset
  - Added : eval/dataset_utils/common.py -> PersONAL_Episode

---

## 📑 Citation
If you find this work useful, please cite:

```bibtex
@article{ziliotto2025personal,
  title   = {PersONAL: Towards a Comprehensive Benchmark for Personalized Embodied Agents},
  author  = {Filippo Ziliotto and Jelin Raphael Akkara and Alessandro Daniele and Lamberto Ballan and Luciano Serafini and Tommaso Campari},
  journal = {arXiv preprint arXiv:2509.19843},
  year    = {2025}
}
