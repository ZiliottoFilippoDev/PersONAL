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
        # return obj_pos, obj_view_pts

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

    def eval_metrics(self, traj_path: str, hab_results_path: str):

        #Read habitat results log and extract obj_description
        with open(hab_results_path, "r") as f:
            hab_info = json.load(f)
        
        obj_descr = hab_info["target_object"]

        #Read trajectory
        with open(traj_path, "r") as f:
            traj = f.readlines()

        traj = [[float(s_sub) for s_sub in s.strip().split(",")] for s in traj]
        traj = np.array(traj)[:, 1:]

        #Final Position of agent
        final_pos = traj[-1]

        #Geodesic Distance covered by trajectory
        traj_path_dist = sum(
            self._geo_dist(pt_1, pt_2)
            for pt_1, pt_2 in zip(traj[:-1], traj[1:])
        )

        #GT : Obtain episode
        found_ep = False
        for ep_info in self.scene_info["episodes"]:
            if ep_info["description"][0] == obj_descr:
                found_ep = True
                break

        assert found_ep, f"Episode with id {episode_id} and obj descr {obj_descr} not found in scene {self.scene_name}"

        #Get distance to object and nearest viewpoint
        dtg_obj, dtg_vw = self.get_geo_dtg(final_pos = final_pos,
                                            obj_cat = ep_info["object_category"],
                                            obj_name = ep_info["object_id"])
        
        #Calculate metrics
        success, spl = 0, 0
        if dtg_vw <= self.success_thresh:
            success = 1

            #Get SPL
            gt_path_dist = ep_info["info"]["geodesic_distance"]
            spl = gt_path_dist / max(gt_path_dist, traj_path_dist)


        return success, spl, dtg_obj, dtg_vw
    

def get_hab_results_path(traj_path):

    dir_path = "/".join(traj_path.split("/")[:-2])
    f_name = traj_path.split("/")[-1].replace(".txt", ".json")

    return os.path.join(dir_path, f_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--data_content_dir", type=str)
    parser.add_argument("--success_thresh", type=float, default=0.1)
    parser.add_argument("--recomp_navmesh", action="store_true")
    parser.add_argument("--agent_height", type=float, default=0.88)
    parser.add_argument("--agent_radius", type=float, default=0.18)


    args = parser.parse_args()


    #Arrange log files by scene
    # This is done to run the sim per scene
    logs_by_scene = {}
    num_eps = 0
    for f_name in tqdm(os.listdir(args.log_dir)):

        if (not f_name.endswith(".json")) or (not f_name.__contains__("_")): continue

        scene_name = f_name.split("_")[-1].replace(".json", "")
        traj_log_path = os.path.join(args.log_dir, "trajectory", f_name.replace(".json", ".txt"))
        hab_log_path = os.path.join(args.log_dir, f_name)
        log_path = (traj_log_path, hab_log_path)

        if scene_name in logs_by_scene:
            logs_by_scene[scene_name].append(log_path)
        else:
            logs_by_scene[scene_name] = [log_path]

        num_eps += 1

    #Initialize variables
    success_vals = []
    spl_vals = []
    dtg_obj_vals = []
    dtg_vw_vals = []

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
        
    
        for log_path in logs_by_scene[scene]:

            success, spl, dtg_obj, dtg_vw = evaluator.eval_metrics(traj_path = log_path[0],
                                                                    hab_results_path = log_path[1])
            
            success_vals.append(success)
            spl_vals.append(spl)
            if dtg_obj < 1e8: dtg_obj_vals.append(dtg_obj)
            if dtg_vw < 1e8: dtg_vw_vals.append(dtg_vw)

            # num_eps += 1
            pbar.update()

        evaluator._close_sim()
        
    #Print the metrics
    print(f"\n\nMetrics : \n")
    print(f" - Success Rate : {sum(success_vals)}/{num_eps} -> {(sum(success_vals)/num_eps) * 100} %")
    print(f" - SPL : {sum(spl_vals)/num_eps}")
    print(f" - DTG (obj) : {np.mean(dtg_obj_vals)} (Reacheable : {len(dtg_obj_vals)}/{num_eps})")
    print(f" - DTG (vw_pt) : {np.mean(dtg_vw_vals)} (Reacheable : {len(dtg_vw_vals)}/{num_eps})")