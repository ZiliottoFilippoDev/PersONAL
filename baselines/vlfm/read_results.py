import os
import json
import gzip
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import argparse


def read_logs(log_dir):
    results = {}
    invalids = {}

    file_counts = 0

    for file_name in tqdm(os.listdir(log_dir)):
        try:
            if "json" not in file_name: continue

            file_path = os.path.join(log_dir, file_name)
            with open(file_path, "r") as f:
                info = json.load(f)

            info = defaultdict(list, info)

            episode_id = file_name.split("_")[0]
            scene_id = "_".join( file_name.split(".json")[0].split("_")[1:] )

            results[(scene_id, episode_id)] = {
                "dist_to_goal": info["distance_to_goal"],
                "success": info["success"],
                "spl": info["spl"],
                "target_object": info["target_object"],
                "stop_called": info["stop_called"],
                "num_steps": info["num_steps"]
            }

            file_counts += 1

        except Exception as e:
            print(f"Skipping file {file_name} due to Error {e}. Episode Success: {info['success']}")
            invalids[id] = info
            continue

    print(f"Loaded {file_counts}/{file_counts+len(invalids)} files")

    return results, invalids

def get_spl_traj(scene_id, episode_id, 
                 target_obj_descr, traj_dir, data_content_dir, 
                 data_mode=None):

    traj_path = os.path.join(traj_dir, f"{str(episode_id)}_{scene_id}.txt")
    # if not os.path.exists(traj_path): return 0
    assert os.path.exists(traj_path), f"Trajectory file {traj_path} does not exist."
    
    #Get path length from trajectory
    with open(traj_path, "r") as f:
        traj = f.readlines()

    traj = np.array([[float(elem) for elem in row.strip().split(",")] for row in traj])[:-1]
    traj = traj[:, [1, 3]]
    
    deltas = traj[1:, :] - traj[:-1, :]
    traj_dist = np.linalg.norm(deltas, axis=1).sum()

    #Get optimal (GT) path length
    scene_info_path = os.path.join(data_content_dir, f"{scene_id}.json.gz")

    with gzip.open(scene_info_path, "r") as f:
        scene_info = json.load(f)

    geo_dist = None
    for ep in scene_info["episodes"]:

        ep_obj = ep["description"][0] if data_mode == "PersONAL" else ep["object_category"]

        if target_obj_descr == ep_obj:
            geo_dist = ep["info"]["geodesic_distance"]
            break

    if geo_dist is None: return None
    
    spl = geo_dist / max(geo_dist, traj_dist)
    return spl

def read_results(results, dist_thresh=0.1, 
                 with_stop=True, 
                 traj_dir=None, data_content_dir=None,
                 data_mode = None,
                 till_num = -1):

    success = 0
    spl = 0

    keys = []

    spl_ratios = []
    spl_traj = []
    dtg = []

    num_eps = 0

    for k in results:
        scene_id, episode_id = k[0], k[1]

        dist_to_goal = results[k]["dist_to_goal"]
        stop_called = results[k]["stop_called"]
        spl_curr = results[k]["spl"]
        spl_curr = 0 if type(spl) is not int else spl_curr

        steps = results[k]["num_steps"]
 
        if (not with_stop) or (steps > 498): stop_called = True

        if (dist_to_goal < dist_thresh) and (stop_called): #and (steps < 499):
            success += 1

            keys.append(k)

            spl_ratios.append(spl_curr)

            #Calculate SPL from trajectory
            if (traj_dir) and (data_content_dir):
                spl_traj.append(
                    get_spl_traj(scene_id, episode_id, 
                                 target_obj_descr = results[k]['target_object'], 
                                 traj_dir = traj_dir,
                                 data_content_dir = data_content_dir,
                                 data_mode = data_mode
                                 )
                )


        dtg.append(dist_to_goal)

        if (till_num > 0) and (num_eps > till_num):
            break

        num_eps += 1
    
    
    dtg = [elem for elem in dtg if not np.isinf(elem)]

    # num_eps = len(results)
    print(f"\nMetrics: ")
    print(f" - Success Rate : {success}/{num_eps} -> {(success/num_eps) * 100}")
    print(f" - DTG : {np.mean(dtg)}")
    print(f" - SPL (hab) : {(sum(spl_ratios)/num_eps)}")

    if len(spl_traj) > 0:
        print(f" - SPL (traj): {sum(spl_traj)/num_eps}")

    return keys


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--PersONAL_data_type", type=str, required=True)
    args = parser.parse_args()

    # ------ Preprocessing Args --------
    assert os.path.exists(args.log_dir) and os.path.isdir(args.log_dir), "Invalid Log Dir."
    assert args.PersONAL_data_type in ["easy", "medium", "hard"], "PersONAL data type should be one of the following : easy, medium, hard"

    if args.PersONAL_data_type == "easy":
        data_content_dir = f"habitat-lab/data/datasets/PersONAL/active/val/{args.PersONAL_data_type}/content"
    else:
        data_content_dir = f"habitat-lab/data/datasets/PersONAL/active/val/test_baselines/{args.PersONAL_data_type + '_filt'}/content"

    traj_dir = os.path.join(args.log_dir, "trajectory")

    # ------ Reading results ---------
    results, errors = read_logs(args.log_dir)

    keys_success = read_results(results, 
                                dist_thresh = 10,
                                with_stop = True,
                                traj_dir = traj_dir,
                                data_content_dir = data_content_dir,
                                data_mode = "PersONAL")

