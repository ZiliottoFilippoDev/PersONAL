# VLFM: Vision-Language Frontier Maps for Zero-Shot Semantic Navigation

**Paper Reference:** [Link](https://arxiv.org/abs/2312.03275)

**Author:** Naoki Yokoyama, Sehoon Ha, Dhruv Batra, Jiuguang Wang, Bernadette Bucher

**Source Repo:** [Link](https://github.com/bdaiinstitute/vlfm)

## Setting up the directory

To test the VLFM baseline on the PersONAL dataset, follow the instructions below. Before running, we need to set up a few paths and directories (as required by the source repo). 

If you need an overview of the changes made to the source code, please 
refer to the PersONAL_changes.txt file present in the vlfm directory.

### Create habitat-lab directory

```bash
#Enter the VLFM directory
cd PersONAL/baselines/vlfm

#Symlink habitat-lab (present in parent dir)
ln -s \<PATH-TO-PersONAL\>/PersONAL/habitat-labs/v0.2.5/habitat-lab habitat-lab
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

## :black_nib: Source Citation

```
@inproceedings{yokoyama2024vlfm,
  title={VLFM: Vision-Language Frontier Maps for Zero-Shot Semantic Navigation},
  author={Naoki Yokoyama and Sehoon Ha and Dhruv Batra and Jiuguang Wang and Bernadette Bucher},
  booktitle={International Conference on Robotics and Automation (ICRA)},
  year={2024}
}
```
