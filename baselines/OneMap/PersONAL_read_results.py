import os
import numpy as np
import gzip
import json
from tqdm import tqdm
from eval.habitat_evaluator import Result
from eval.dataset_utils.hm3d_PersONAL_dataset import load_PersONAL_episodes, load_hm3d_objects

import habitat_sim
from habitat_sim import ActionSpec, ActuationSpec
from eval.habitat_utils import get_closest_dist
from scipy.spatial.transform import Rotation as R

import argparse

def get_obj_pos(scene_name: str, obj_instance: str, data_info_dir: str):
    """"
    Returns the object position and viewpoints

    Args:
        - scene_name
        - obj_instance: Instance ID, made of category name and ID. Ex.: couch_160
        - data_info_dir: Dataset content directory
    """

    scene_info_path = os.path.join(data_info_dir, f"{scene_name}.json.gz")
    with gzip.open(scene_info_path, "r") as f:
        scene_info = json.load(f)

    for goal in scene_info["goals_by_category"]:
        for goal_instance in scene_info["goals_by_category"][goal]:
            if goal_instance["object_name"] == obj_instance:

                return goal_instance["position"], goal_instance["view_points"]

    return None, None

def transform_obs(obs_pos, obs_view_pts):

    #Transforms
    obs_to_pose = lambda obs: [-obs[2], -obs[0], obs[1]]
    quat_to_yaw = lambda orient: R.from_quat([orient[0], orient[1], orient[2], orient[3]]).as_euler("yxz")[0]


    #Transforming viewpoints (positions and rotations)
    view_pts_obs_pos = np.array([pt["agent_state"]["position"] for pt in obs_view_pts])
    view_pts_obs_rot = np.array([pt["agent_state"]["rotation"] for pt in obs_view_pts])

    view_pts_pos = np.apply_along_axis(obs_to_pose, axis=1, arr=view_pts_obs_pos)
    view_pts_rot = np.apply_along_axis(quat_to_yaw, axis=1, arr=view_pts_obs_rot)

    #Transforming object position
    pos = obs_to_pose(obs_pos)

    return pos, view_pts_pos, view_pts_rot

dist_btw_pts = lambda pt_1, pt_2: np.linalg.norm(pt_1 - pt_2)

def get_metrics(results_dir, data_info_dir, 
                success_thresh = 0.2, verbose=False):

    traj_dir = os.path.join(results_dir, "trajectories")
    state_dir = os.path.join(results_dir, "state")
    episodes, scene_data = load_PersONAL_episodes(episodes = [],
                                                    scene_data = {},
                                                    object_nav_path = data_info_dir)

    success_counts = 0
    state_success_counts = 0
    spl = []
    dtg = []

    skipped_eps = []
    ep_counts = 0
    
    for ep_num in tqdm(range(len(episodes))):

        episode = episodes[ep_num]

        traj_file_path = os.path.join(traj_dir, f"poses_{ep_num}.csv")
        if not os.path.exists(traj_file_path): 
            skipped_eps.append(ep_num)
            continue
        
        #Extract relevant info from episode info
        scene_name = episode.scene_id.split("/")[-1].split(".")[0]
        target_obj_instance = episode.extra["object_instance"]
        best_path_len = episode.best_dist

        #Get GT object position and viewpoints. Transform them into pose convention
        obj_pos, obj_vw_pts = get_obj_pos(scene_name, target_obj_instance,
                                          data_info_dir)
        obj_pos, obj_view_pts_pos, obj_view_pts_rot = transform_obs(obj_pos, obj_vw_pts)
        obj_pos, obj_view_pts_pos = obj_pos[:2], obj_view_pts_pos[:, :2]

        #Get (1) Distance travelled by agent, (2) Final position of agent
        traj = np.genfromtxt(traj_file_path, delimiter=",")
        deltas = traj[1:, :2] - traj[:-1, :2]
        traj_dist = np.linalg.norm(deltas, axis=1).sum()

        final_pos = traj[-1][:2]

        #Calculate metrics
        view_pt_dists = np.apply_along_axis(lambda v: dist_btw_pts(v, final_pos), axis=1, arr=obj_view_pts_pos)
        dist_to_closest_view_pt = min(view_pt_dists)

        if dist_to_closest_view_pt <= success_thresh:
            success_counts += 1
            
            spl_curr = best_path_len / max(best_path_len, traj_dist)
            spl.append(spl_curr)

            # if len(traj) < 498: 
            #     print(f"Episode: {ep_num}, Scene: {scene_name}, Num steps: {len(traj)}")
            #     return

        dtg_curr = dist_btw_pts(final_pos, obj_pos)
        dtg.append(dtg_curr)

        ep_counts += 1

        #Check state value for episode
        with open(os.path.join(state_dir, f"state_{ep_num}.txt"), "r") as f:
            state = f.read().strip()

        if int(state) == 1: state_success_counts += 1


    # num_eps = len(episodes)
    num_eps = ep_counts

    SR = np.round((success_counts / num_eps) * 100, 3)
    SPL = np.round(sum(spl) / num_eps, 5)
    DTG = np.round(np.mean(dtg), 3)

    state_SR = np.round(state_success_counts/num_eps * 100, 3)
    
    if verbose:
        print(f"\nSR (thresh: {success_thresh}): {success_counts}/{num_eps} = {SR} %")
        print(f"SPL : {SPL}")
        print(f"DTG : {DTG}\n")

        print(f"State SR: {state_success_counts}/{num_eps} = {state_SR} %\n")

        print(f"Skipped Episodes: {len(skipped_eps)}\nMin: {min(skipped_eps)}, Max: {max(skipped_eps)}")

    # return num_eps, success_counts, spl, dtg, state_success_counts

def get_metrics_all_splits(results_dir, data_info_dir, 
                           success_thresh = 0.2, split_verbose=False):

    num_eps = 0
    success_counts = 0
    spl = []
    dtg = []

    state_success_counts = 0

    results_splits = [d for d in os.listdir(results_dir) if d.__contains__("split")]

    for split in results_splits:
        split_dir = os.path.join(results_dir, split)

        split_num_eps, split_succ, split_spl, split_dtg, split_state_succ = get_metrics(split_dir, data_info_dir, success_thresh, split_verbose)
        print(f"Num Eps done in {split} : {split_num_eps}")

        num_eps += split_num_eps
        success_counts += split_succ
        spl += split_spl
        dtg += split_dtg

        state_success_counts += split_state_succ

    print("\nNum eps: ", num_eps)
    SR = np.round((success_counts / num_eps) * 100, 3)
    SPL = np.round(sum(spl) / num_eps, 5)
    DTG = np.round(np.mean(dtg), 3)
    state_SR = np.round(state_success_counts/num_eps * 100, 3)

    print(f"\nSR (thresh: {success_thresh}): {success_counts}/{num_eps} = {SR} %")
    print(f"SPL : {SPL}")
    print(f"DTG : {DTG}\n")

    print(f"State SR: {state_success_counts}/{num_eps} = {state_SR} %\n")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--PersONAL_data_type", type=str, required=True)
    parser.add_argument("--success_thresh", default=0.2, type=float)
    args = parser.parse_args()

    # ------ Preprocessing Args --------
    assert os.path.exists(args.log_dir) and os.path.isdir(args.log_dir), "Invalid Log Dir."
    assert args.PersONAL_data_type in ["easy", "medium", "hard"], "PersONAL data type should be one of the following : easy, medium, hard"

    if args.PersONAL_data_type == "easy":
        data_content_dir = f"habitat-lab/data/datasets/PersONAL/active/val/{args.PersONAL_data_type}/content"
    else:
        data_content_dir = f"habitat-lab/data/datasets/PersONAL/active/val/test_baselines/{args.PersONAL_data_type + '_filt'}/content"

    # ------ Read results -------
    get_metrics(args.log_dir, data_content_dir, 
            success_thresh = args.success_thresh, verbose=True)