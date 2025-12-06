import os
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["GLOG_minloglevel"] = "2"
os.environ["HABITAT_SIM_LOG"] = "quiet"

import habitat_sim
import numpy as np
import argparse
import os
from tqdm import tqdm
import gzip
import json

from eval.dataset_utils.hm3d_PersONAL_dataset import load_PersONAL_episodes
from eval.dataset_utils.common import PersONAL_Episode


class Traj_Metrics:

    def __init__(self,
                 scene_name: str,
                 data_content_dir: str,
                 success_thresh = 0.1,      #Distance to nearest viewpoint
                 scene_dataset_cfg = "habitat-lab/data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json",
                 agent_height = 0.88, agent_radius = 0.18,
                 recomp_navmesh = False
                 ):
        
        self.scene_name = scene_name
        self.success_thresh = success_thresh
        
        #Load scene episodes
        scene_info_path = os.path.join(data_content_dir, f"{scene_name}.json.gz")
        with gzip.open(scene_info_path, "r") as f:
            self.scene_info = json.load(f)
        
        #Initialize Simulator
        self._init_sim(scene_dataset_cfg, 
                       agent_height, agent_radius,
                       recomp_navmesh)

    # ---
    # Habitat Simulator
    # ---

    def _init_sim(self,
                  scene_dataset_cfg,
                  agent_height, agent_radius,
                  recomp_navmesh):

        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_dataset_config_file = scene_dataset_cfg
        sim_cfg.scene_id = self.scene_name

        sim_cfg.enable_physics = False
        sim_cfg.gpu_device_id = 0
        sim_cfg.allow_sliding = False

        sensor_specs = []

        #Buffer variables : Values chosen as per standard ObjectNav Config
        #Reference : ZSON (https://arxiv.org/abs/2206.12403)
        img_size = (480, 640)
        sensor_height = 0.88
        hfov = 79.0

        for name, sensor_type in zip(
            ["color", "depth", "semantic"],
            [
                habitat_sim.SensorType.COLOR,
                habitat_sim.SensorType.DEPTH,
                # habitat_sim.SensorType.SEMANTIC,
            ],
        ):
            sensor_spec = habitat_sim.CameraSensorSpec()
            sensor_spec.uuid = f"{name}_sensor"
            sensor_spec.sensor_type = sensor_type
            sensor_spec.resolution = [img_size[0], img_size[1]]
            sensor_spec.position = [0.0, sensor_height, 0.0]
            sensor_spec.hfov = hfov
            sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            sensor_specs.append(sensor_spec)

        # create agent specifications
        agent_cfg = habitat_sim.agent.AgentConfiguration(
            height=agent_height,
            radius=agent_radius,
            sensor_specifications=sensor_specs,
            action_space = {
                "move_forward": habitat_sim.agent.ActionSpec(
                    "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
                ),
                "turn_left": habitat_sim.agent.ActionSpec(
                    "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
                ),
                "turn_right": habitat_sim.agent.ActionSpec(
                    "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
                ),
            }
        )

        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
        self.sim = habitat_sim.Simulator(cfg)

        if recomp_navmesh:
            # set the navmesh
            print(f"Pathfinder is Loaded: {self.sim.pathfinder.is_loaded}")
            navmesh_settings = habitat_sim.NavMeshSettings()
            navmesh_settings.set_defaults()
            navmesh_settings.agent_height = agent_height
            navmesh_settings.agent_radius = agent_radius
            navmesh_settings.agent_max_climb = 0.2
            navmesh_settings.cell_height = 0.2
            navmesh_success = self.sim.recompute_navmesh(self.sim.pathfinder, navmesh_settings)
            print(f"Navmesh Recompute Success: {navmesh_success}")
            assert navmesh_success, "Failed to build the navmesh!"

    def _close_sim(self):
        self.sim.close()

    # ----
    # Distance Calculation
    # ----

    def _geo_dist(self, agent_pos: np.ndarray, obj_pos: np.ndarray):

        """
        Implementation follows habitat. 
        Refer to habitat-lab/habitat/sims/habitat_simulator.py -> HabitatSim.geodesic_distance
        """

        path = habitat_sim.MultiGoalShortestPath()
        if isinstance(obj_pos[0], np.ndarray):
            path.requested_ends = np.array(obj_pos, dtype=np.float32)
        else:
            path.requested_ends = np.array([np.array(obj_pos, dtype=np.float32)])
        
        path.requested_start = np.array(agent_pos, dtype=np.float32)
        
        self.sim.pathfinder.find_path(path)
        return path.geodesic_distance

    def _euc_dist(self, agent_pos: np.ndarray, obj_pos: np.ndarray):

        assert len(obj_pos.shape) == 2
        assert len(agent_pos.shape) == 1

        dists = np.apply_along_axis(lambda pt : self._dist_btw_pts(pt, agent_pos), axis=1, arr=obj_pos)
        return min(dists)
    
    def _dist_btw_pts(self, pt_1: np.ndarray, pt_2: np.ndarray):
        return np.linalg.norm(pt_1 - pt_2)
        

    # ---
    # Metric : Distance to Goal (Geodesic and Euclidean)
    # ---

    def get_geo_dtg(self,
                    final_pos, 
                    obj_cat, obj_name,
                    ):

        """
        Returns geodesic distance to object position and nearest viewpoint
        """

        obj_pos, obj_view_pts = [], []

        #Obtain the position of the target object and viewpoints
        goal_key = f"{self.scene_name}.basis.glb_{obj_cat}"
        for goal_instance in self.scene_info["goals_by_category"][goal_key]:

            if goal_instance["object_name"] != obj_name: continue   #Skip if not instance

            curr_view_pts = goal_instance["view_points"]
            curr_view_pts = [pt["agent_state"]["position"] for pt in curr_view_pts]

            obj_pos.append(goal_instance["position"])
            obj_view_pts.append(curr_view_pts)

        obj_pos, obj_view_pts = np.array(obj_pos), np.vstack(obj_view_pts)

        # #Transform wrt OneMap implementation. See habitat_evaluator.
        # obs_to_pose = lambda obs: [-obs[2], -obs[0], obs[1]]
        # obj_pos = obs_to_pose(obj_pos)
        # obj_view_pts = np.apply_along_axis(obs_to_pose, axis=1, arr=obj_view_pts)

        #Calculate distance to goal
        dtg_obj = self._geo_dist(final_pos, obj_pos)
        dtg_vw_pt = self._geo_dist(final_pos, obj_view_pts)
        
        return dtg_obj, dtg_vw_pt
    
    def get_euc_dtg(self, 
                    final_pos, 
                    obj_cat, obj_name):

        obj_pos, obj_view_pts = [], []

        #Obtain the position of the target object and viewpoints
        goal_key = f"{self.scene_name}.basis.glb_{obj_cat}"
        for goal_instance in self.scene_info["goals_by_category"][goal_key]:

            if goal_instance["object_name"] != obj_name: continue   #Skip if not instance

            curr_view_pts = goal_instance["view_points"]
            curr_view_pts = [pt["agent_state"]["position"] for pt in curr_view_pts]

            obj_pos.append(goal_instance["position"])
            obj_view_pts.append(curr_view_pts)

        obj_pos, obj_view_pts = np.array(obj_pos), np.vstack(obj_view_pts)

        # #Transform wrt OneMap implementation. See habitat_evaluator.
        # obs_to_pose = lambda obs: [-obs[2], -obs[0], obs[1]]
        # obj_pos = obs_to_pose(obj_pos)
        # obj_view_pts = np.apply_along_axis(obs_to_pose, axis=1, arr=obj_view_pts)

        #Only consider x,y coordinates
        obj_pos, obj_view_pts = obj_pos[:, [0, 2]], obj_view_pts[:, [0, 2]]

        #Calculate distance to goal
        view_pt_dists = np.apply_along_axis(lambda v: self._dist_btw_pts(v, final_pos), axis=1, arr=obj_view_pts)
        dtg_vw_pt = min(view_pt_dists)

        obj_dists = np.apply_along_axis(lambda v: self._dist_btw_pts(v, final_pos), axis=1, arr=obj_pos)
        dtg_obj = min(obj_dists)

        return dtg_obj, dtg_vw_pt

    # ---
    # Calculate Metrics : SR, SPL, DTG
    # ---

    def eval_metrics(self, traj_path: str, episode: PersONAL_Episode):

        #Read trajectory
        traj = np.genfromtxt(traj_path, delimiter=",")
        traj = traj[:, :3]

        #Convert traj to habitat coordinates
        to_hab_coord = lambda pose : [ -pose[1], pose[2], -pose[0]]
        traj = np.apply_along_axis(to_hab_coord, axis=1, arr=traj)

        #Final Position of agent
        final_pos = traj[-1]

        #Geodesic Distance covered by trajectory
        traj_path_dist = sum(
            self._geo_dist(pt_1, pt_2)
            for pt_1, pt_2 in zip(traj[:-1], traj[1:])
        )

        #Get distance to object and nearest viewpoint
        dtg_obj, dtg_vw = self.get_geo_dtg(final_pos = final_pos,
                                            obj_cat = episode.extra["object_category"],
                                            obj_name = episode.extra["object_instance"])
        
        #Calculate metrics
        success, spl = 0, 0
        if dtg_vw <= self.success_thresh:
            success = 1

            #Get SPL
            gt_path_dist = episode.best_dist
            spl = gt_path_dist / max(gt_path_dist, traj_path_dist)


        return success, spl, dtg_obj, dtg_vw
    

get_scene_name = lambda path : path.split("/")[-1].split(".")[0]


def get_metrics(args):

    traj_dir = os.path.join(args.log_dir, "trajectories")
    state_dir = os.path.join(args.log_dir, "state")
    episodes, _ = load_PersONAL_episodes(episodes = [],
                                        scene_data = {},
                                        object_nav_path = args.data_content_dir)

    #Arrange log files by scene
    # This is done to run the sim per scene
    logs_by_scene = {}
    num_eps = 0
    for ep in episodes:

        scene = get_scene_name(ep.scene_id)
        ep_id = ep.episode_id

        
        log_info = {
            "episode" : ep,
            "traj_path" : os.path.join(traj_dir, f"poses_{ep_id}.csv"),
            "state_path" : os.path.join(state_dir, f"state_{ep_id}.txt")
        }

        if (not os.path.exists(log_info["traj_path"])) or (not os.path.exists(log_info["state_path"])): 
            continue

        if scene in logs_by_scene:
            logs_by_scene[scene].append(log_info)
        else:
            logs_by_scene[scene] = [log_info]

        num_eps += 1

    #Initialize variables
    success_vals = []
    spl_vals = []
    dtg_obj_vals = []
    dtg_vw_vals = []

    state_success_vals = []

    # num_eps = 0
    pbar = tqdm(total = num_eps)

    #Obtain metrics per scene and update the above variables
    for scene in logs_by_scene:

        print(f"\n\nScene : {scene}")

        evaluator = Traj_Metrics(scene_name = scene,
                                    data_content_dir = args.data_content_dir,
                                    success_thresh = args.success_thresh,
                                    recomp_navmesh = args.recomp_navmesh,
                                    agent_height = args.agent_height,
                                    agent_radius = args.agent_radius)
        

        for log_info in logs_by_scene[scene]:

            episode, traj_path, state_path = log_info["episode"], log_info["traj_path"], log_info["state_path"]

            success, spl, dtg_obj, dtg_vw = evaluator.eval_metrics(traj_path = traj_path,
                                                                    episode = episode)
            
            success_vals.append(success)
            spl_vals.append(spl)
            if dtg_obj < 1e8: dtg_obj_vals.append(dtg_obj)
            if dtg_vw < 1e8: dtg_vw_vals.append(dtg_vw)

            #Check state value for episode
            with open(state_path, "r") as f:
                state = f.read().strip()

            if int(state) == 1:
                state_success_vals.append(1)
            else:
                state_success_vals.append(0)

            pbar.update()

        evaluator._close_sim()


    print(f"\n\nMetrics : \n")
    print(f" - Success Rate : {sum(success_vals)}/{num_eps} -> {(sum(success_vals)/num_eps) * 100} %")
    print(f" - State Success Rate : {sum(state_success_vals)}/{num_eps} -> {(sum(state_success_vals)/num_eps) * 100} %")
    print(f" - SPL : {sum(spl_vals)/num_eps}")
    print(f" - DTG (obj) : {np.mean(dtg_obj_vals)} (Reacheable : {len(dtg_obj_vals)}/{num_eps})")
    print(f" - DTG (vw_pt) : {np.mean(dtg_vw_vals)} (Reacheable : {len(dtg_vw_vals)}/{num_eps})")

    return num_eps, sum(success_vals), sum(state_success_vals), sum(spl_vals), sum(dtg_obj_vals), len(dtg_obj_vals), sum(dtg_vw_vals), len(dtg_vw_vals)


def get_metrics_all_splits(args):

    num_eps = 0
    success = 0
    spl = 0
    dtg_obj, dtg_obj_eps = 0, 0
    dtg_vw, dtg_vw_eps = 0, 0

    state_success = 0

    splits = [d for d in os.listdir(args.log_dir) if d.__contains__("split")]
    root_log_dir = args.log_dir

    for split in splits:
        split_dir = os.path.join(root_log_dir, split)
        args.log_dir = split_dir

        split_num_eps, split_success, split_state_success, split_spl, \
        split_dtg_obj, split_dtg_obj_eps, \
        split_dtg_vw, split_dtg_vw_eps = get_metrics(args)

        num_eps += split_num_eps
        success += split_success
        state_success += split_state_success
        spl += split_spl
        dtg_obj += split_dtg_obj
        dtg_obj_eps += split_dtg_obj_eps
        dtg_vw += split_dtg_vw
        dtg_vw_eps += split_dtg_vw_eps

    
    print(f"\n\nMetrics (All Splits) : \n")
    print(f" - Success Rate : {success}/{num_eps} -> {(success/num_eps) * 100} %")
    print(f" - State Success Rate : {state_success}/{num_eps} -> {(state_success/num_eps) * 100} %")
    print(f" - SPL : {spl/num_eps}")
    print(f" - DTG (obj) : {dtg_obj/dtg_obj_eps} (Reacheable : {dtg_obj_eps}/{num_eps})")
    print(f" - DTG (vw_pt) : {dtg_vw/dtg_vw_eps} (Reacheable : {dtg_vw_eps}/{num_eps})")





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--data_content_dir", type=str)
    parser.add_argument("--success_thresh", type=float, default=0.1)
    parser.add_argument("--recomp_navmesh", action="store_true")
    parser.add_argument("--agent_height", type=float, default=0.88)
    parser.add_argument("--agent_radius", type=float, default=0.18)


    args = parser.parse_args()


    get_metrics_all_splits(args)
