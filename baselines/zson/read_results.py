import os
import json
import gzip
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import argparse


dist_btw_pts = lambda pt_1, pt_2: np.linalg.norm(pt_1 - pt_2)

def get_dtg(final_pos, 
            obj_cat, obj_name,
            scene_name, scene_info,
            to_vw_pt = True):
    """"
    Obtain DTG using the final position of agent and the object's true position. 

    It is possible to get the distance to the object's position or one of its nearest
    viewpoints by switching the to_vw_pt argument. 
    """

    obj_pos, obj_view_pts = [], []

    #Obtain the position of the target object and viewpoints
    goal_key = f"{scene_name}.basis.glb_{obj_cat}"
    for goal_instance in scene_info["goals_by_category"][goal_key]:

        if goal_instance["object_name"] != obj_name: continue   #Skip if not instance

        curr_view_pts = goal_instance["view_points"]
        curr_view_pts = [pt["agent_state"]["position"] for pt in curr_view_pts]

        obj_pos.append(goal_instance["position"])
        obj_view_pts.append(curr_view_pts)

    obj_pos, obj_view_pts = np.array(obj_pos), np.vstack(obj_view_pts)
    obj_pos, obj_view_pts = obj_pos[:, [0, 2]], obj_view_pts[:, [0, 2]]

    #Calculate distance to goal
    if to_vw_pt:
        view_pt_dists = np.apply_along_axis(lambda v: dist_btw_pts(v, final_pos), axis=1, arr=obj_view_pts)
        dtg = min(view_pt_dists)

    else:
        obj_dists = np.apply_along_axis(lambda v: dist_btw_pts(v, final_pos), axis=1, arr=obj_pos)
        dtg = min(obj_dists)
    
    return dtg
    
def get_results(log_dir, data_content_dir,
                    success_thresh = 0.2,
                    dtg_with_vw = True):

    success = 0
    spl_vals = []
    dtg_vals = []

    num_eps = 0

    for f_name in tqdm(os.listdir(log_dir)):

        if not f_name.endswith(".txt"): continue

        #Agent's Trajetory
        f_path = os.path.join(log_dir, f_name)
        with open(f_path, "r") as f:
            traj = f.readlines()

        traj = [[float(s_sub) for s_sub in s.strip().split(",")] for s in traj]
        traj = np.array(traj)
        traj = traj[:, [1, 3]]

        deltas = traj[1:, :] - traj[:-1, :]
        traj_dist = np.linalg.norm(deltas, axis=1).sum()

        #Ground Truth
        scene_id, episode_id = f_name.split("_")[-1][:-4], f_name.split("_")[0]
        scene_info_path = os.path.join(data_content_dir, f"{scene_id}.json.gz")

        with gzip.open(scene_info_path, "r") as f_scene:
            scene_info = json.load(f_scene)

        #Obtain episode info
        found_ep = False
        for ep_info in scene_info["episodes"]:
            if ep_info["episode_id"] == episode_id:
                found_ep = True
                break

        assert found_ep

        #Distance to Goal (nearest viewpoint)
        dtg = get_dtg(final_pos = traj[-1],
                        obj_cat = ep_info["object_category"],
                        obj_name = ep_info["object_id"],
                        scene_name = scene_id,
                        scene_info = scene_info,
                        to_vw_pt = dtg_with_vw)

        dtg_vals.append(dtg)
        if dtg <= success_thresh:

            success += 1
        
            #Success Path Length (SPL)
            geo_dist = ep_info["info"]["geodesic_distance"]
            spl_ratio = geo_dist / max(geo_dist, traj_dist)
            spl_vals.append(spl_ratio)

        num_eps += 1

    print(f"\n Metrics:\n - Success Rate : {success}/{num_eps} -> {(success/num_eps) * 100} %")
    print(f" - SPL : {sum(spl_vals)/num_eps}")
    print(f" - DTG : {sum(dtg_vals)/num_eps}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--PersONAL_data_type", type=str, required=True)
    parser.add_argument("--success_thresh", type=float, default=0.2)
    parser.add_argument("--dtg_with_vw", type=bool, default=True)
    args = parser.parse_args()

    # ------ Preprocessing Args --------
    assert os.path.exists(args.log_dir) and os.path.isdir(args.log_dir), "Invalid Log Dir."
    assert args.PersONAL_data_type in ["easy", "medium", "hard"], "PersONAL data type should be one of the following : easy, medium, hard"

    if args.PersONAL_data_type == "easy":
        data_content_dir = f"data/datasets/PersONAL/val/{args.PersONAL_data_type}/content"
    else:
        data_content_dir = f"data/datasets/PersONAL/val/test_baselines/{args.PersONAL_data_type + '_filt'}/content"

    # ------ Reading results ---------
    get_results(log_dir = args.log_dir,
                data_content_dir = data_content_dir,
                success_thresh = args.success_thresh,
                dtg_with_vw = args.dtg_with_vw)

