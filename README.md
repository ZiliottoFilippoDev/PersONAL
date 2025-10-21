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

## Setting up PersONAL Dataset

To set up the dataset in Habitat, please follow the below instructions. 

1) Clone the repository

```bash
git clone https://github.com/ZiliottoFilippoDev/PersONAL

cd PersONAL
```

2) Clone habitat-lab

```bash
git clone --branch v0.2.5 https://github.com/facebookresearch/habitat-lab
```

3) Download HM3D data (as per VLFM source repo)

First, set the relevant variables for installation.

```bash
MATTERPORT_TOKEN_ID=<FILL IN FROM YOUR ACCOUNT INFO IN MATTERPORT>
MATTERPORT_TOKEN_SECRET=<FILL IN FROM YOUR ACCOUNT INFO IN MATTERPORT>
DATA_DIR="habitat-lab/data"

# Link to the HM3D ObjectNav episodes dataset, listed here:
# https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md#task-datasets
# From the above page, locate the link to the HM3D ObjectNav dataset.
# Verify that it is the same as the next two lines.
HM3D_OBJECTNAV=https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip


#Download the HM3D scene datasets
python -m habitat_sim.utils.datasets_download \
  --username $MATTERPORT_TOKEN_ID --password $MATTERPORT_TOKEN_SECRET \
  --uids hm3d_val \
  --data-path $DATA_DIR
```

4) Symlink to PersONAL data (contains episodes)

```bash
#Create directory in habitat-lab
mkdir -p habitat-lab/data/datasets/PersONAL/active/

#Create symlink
ln -s data/split  habitat-lab/data/datasets/PersONAL/active/val
```

5) Update relevant files in habitat-lab to register PersONAL as a valid dataset

```bash
#Dataset info
cp habitat-utils/personalized_object_nav_dataset.py habitat-lab/habitat-lab/habitat/datasets/object_nav/

#Task info
cp habitat-utils/personalized_object_nav_task.py habitat-lab/habitat-lab/habitat/tasks/nav/

#Register PersONAL
cp habitat-utils/register_personalized_dataset.py habitat-lab/habitat-lab/habitat/datasets/object_nav/__init__.py
cp habitat-utils/registration.py habitat-lab/habitat-lab/habitat/datasets/registration.py

#Util : Obtain current habitat position and rotation
cp habitat-utils/RL_Env.py habitat-lab/habitat-lab/habitat/core/env.py
```

(NEED A TEST TO CONFIRM IF ALL WENT )

## Testing Baselines : VLFM

### Setting up the Env

To test the VLFM baseline on the PersONAL dataset, follow the instructions below. Before running, we need to set up a few paths and directories (as required by the source repo). 

If you need an overview of the changes made to the source code, please 
refer to the PersONAL_changes.txt file present in the vlfm directory.

1) Create habitat-lab directory

```bash
#Enter the VLFM directory
cd vlfm

#Symlink habitat-lab (present in parent dir)
ln -s /mnt/PersONAL/habitat-lab habitat-lab
```

2) Clone required repositories (as instructed in the source)

  - GroundingDINO : [https://github.com/IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
  - MobileSAM : [https://github.com/ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
  - yolov7 : [github.com/WongKinYiu/yolov7](github.com/WongKinYiu/yolov7)
  - depth_camera_filtering : [https://github.com/naokiyokoyama/depth_camera_filtering/tree/main/depth_camera_filtering](https://github.com/naokiyokoyama/depth_camera_filtering/tree/main/depth_camera_filtering)
  - frontier_exploration : [https://github.com/naokiyokoyama/frontier_exploration/tree/main/frontier_exploration](https://github.com/naokiyokoyama/frontier_exploration/tree/main/frontier_exploration)

3) Download following weights to the data directory:
  - groundingdino_swint_ogc.pth : [https://github.com/IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
  - mobile_sam.pt : [https://github.com/ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
  - yolov7-e6e.pt : [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)

  NOTE: These should be saved inside the data directory.


### Training

```bash
#Enter the vlfm directory
cd vlfm

#Run on PersONAL
python -m vlfm.run PersONAL_args.log_dir=log/junk
```

### Evaluation

```bash
python -m read_results --log_dir log/junk/ --PersONAL_data_type easy
```


## Testing Baselines : OneMap

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
