# Testing Baselines

## Baseline : VLFM

### Setting up the Env

To test the VLFM baseline on the PersONAL dataset, follow the instructions below. Before running, we need to set up a few paths and directories (as required by the source repo). 

If you need an overview of the changes made to the source code, please 
refer to the PersONAL_changes.txt file present in the vlfm directory.

#### Create habitat-lab directory

```bash
#Enter the VLFM directory
cd baselines/vlfm

#Symlink habitat-lab (present in parent dir)
ln -s \<PATH-TO-PersONAL\>/PersONAL/habitat-lab habitat-lab
```

#### Clone required repositories (as instructed in the source)

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

#### Download model weights

The weights for MobileSAM, GroundingDINO, and Yolov7 must be saved to the `data/` directory. The weights can be downloaded from the following links:

  - `groundingdino_swint_ogc.pth` : [https://github.com/IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
  - `mobile_sam.pt` : [https://github.com/ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
  - `yolov7-e6e.pt` : [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)


### Training

For training (and evaluation), make sure the current working directory is in `baselines/vlfm/`.

```bash
#Run on PersONAL
python -m vlfm.run PersONAL_args.log_dir=log/junk
```

### Evaluation

```bash
python -m read_results --log_dir log/junk/ --PersONAL_data_type easy
```