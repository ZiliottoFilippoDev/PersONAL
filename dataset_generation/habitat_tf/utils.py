import numpy as np
import habitat_sim
from typing import List, Dict, Any, Tuple
import os
import gzip
import json

def load_merged_scene_data(
    base_path: str,
    scene_id: str,
    splits: List[str] = ("val_seen", "val_seen_synonyms", "val_unseen"),
) -> Dict[str, Any]:
    """
    Loads and merges 'goals' and 'episodes' for a scene across multiple splits.
    Later splits override earlier ones on key conflict.
    """
    # normalize filename
    fname = os.path.splitext(os.path.basename(scene_id.split('/')[-1].split('.')[0]))[0] + ".json.gz"
    merged = {"goals": {}, "episodes": []}
    found = False

    for split in splits:
        path = os.path.join(base_path, split, "content", fname)
        if not os.path.exists(path):
            continue
        found = True
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        merged["goals"].update(data.get("goals", {}))
        merged["episodes"].extend(data.get("episodes", []))

    if not found:
        raise FileNotFoundError(f"No scene data found for '{scene_id}' in splits {splits}")
    return merged

def build_lookups(
    merged: Dict[str, Any],
    use_view_points: bool
) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[int, List[Dict[str, Any]]]]:
    """
    Returns two dicts:
      view_point_lookup[obj_id] -> list of view_point dicts
      start_lookup[obj_id]      -> list of {'pos', 'rot'} dicts
    """
    view_point_lookup: Dict[int, List[Dict[str, Any]]] = {}
    start_lookup: Dict[int, List[Dict[str, Any]]]    = {}

    # Extract view_points per object_id
    if use_view_points:
        for goal_list in merged["goals"].values():
            for item in goal_list:
                obj_id = item.get("object_id")
                if obj_id is not None:
                    view_point_lookup[obj_id] = item.get("view_points", [])

    # Extract start positions & rotations from goat episodes
    for goat_ep in merged["episodes"]:
        pos = goat_ep.get("start_position")
        rot = goat_ep.get("start_rotation")
        for task in goat_ep.get("tasks", []):
            obj_id = task[2]
            start_lookup.setdefault(obj_id, []).append({"pos": pos, "rot": rot})

    return view_point_lookup, start_lookup

def random_yaw_rotation() -> list:
    """
    Generate a random rotation quaternion representing a yaw rotation around the Y-axis.
    
    :return: Quaternion [x, y, z, w].
    """
    theta = np.random.uniform(-np.pi, np.pi)
    half = theta / 2.0
    return [0.0, np.sin(half), 0.0, np.cos(half)]

def euclidean_distance(point_a: np.ndarray, point_b: np.ndarray, dims: tuple = (0, 2)) -> float:
    """
    Compute the Euclidean distance between two points along specified dimensions.
    
    :param point_a: First point as a numpy array [x, y, z].
    :param point_b: Second point as a numpy array [x, y, z].
    :param dims: Dimensions to consider for distance (default x and z).
    :return: Euclidean distance.
    """
    diff = point_a[list(dims)] - point_b[list(dims)]
    return float(np.linalg.norm(diff))

def make_simple_cfg(
    settings: dict,
    use_equirectangular: bool = False
) -> habitat_sim.Configuration:
    """Create a Habitat simulator configuration with basic RGB sensor setup.
    
    Args:
        settings: Configuration parameters including:
            - scene: Scene file path
            - width/height: Sensor resolution
            - sensor_height: Height offset for camera
            - hfov: Horizontal field of view
        use_equirectangular: Enable 360° equirectangular projection
            
    Returns:
        Habitat simulator configuration object
    """
    # Simulator configuration
    sim_config = habitat_sim.SimulatorConfiguration()
    sim_config.scene_id = settings["scene"]

    # RGB sensor configuration
    rgb_sensor = habitat_sim.CameraSensorSpec()
    rgb_sensor.uuid = "rgb"
    rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor.resolution = [settings["height"], settings["width"]]
    rgb_sensor.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor.hfov = settings["hfov"]

    if use_equirectangular:
        rgb_sensor.sensor_subtype = habitat_sim.SensorSubType.EQUIRECTANGULAR

    # Agent configuration
    agent_config = habitat_sim.agent.AgentConfiguration()
    agent_config.sensor_specifications = [rgb_sensor]

    return habitat_sim.Configuration(sim_config, [agent_config])

def sample_additional_viewpoints(
    sim: habitat_sim.Simulator,
    object_pos: list,
    count: int = 10,
    radius: float = 1.5,
    max_tries: int = 200,
    visibility_check: bool = True,
) -> list:
    """
    Samples additional navigable viewpoints near an object's position,
    with optional visibility checks and orientation.
    """
    new_views = []
    object_pos_np = np.array(object_pos, dtype=float)
    object_center_np = object_pos_np + np.array([0, 0.1, 0], dtype=float)

    for _ in range(count):
        # Use a placeholder for the best point found in the tries
        best_vp = None
        for i in range(max_tries):
            vp_nav = sim.pathfinder.get_random_navigable_point_near(
                object_pos, radius=radius, max_tries=20
            )
            if vp_nav is None or np.isinf(vp_nav).any() or np.isnan(vp_nav).any():
                continue

            vp = np.array([vp_nav[0], vp_nav[1], vp_nav[2]], dtype=float)
            best_vp = vp  # Store the last valid point as a fallback

            if not visibility_check:
                break

            dist_to_obj = np.linalg.norm(object_center_np - vp)
    
            ray = habitat_sim.geo.Ray(vp, object_center_np - vp)
            
            # --- CORRECTED LOGIC FOR RaycastResults ---
            raycast_results = sim.cast_ray(ray)
            if (not raycast_results.hits) or (raycast_results.hits[0].ray_distance >= dist_to_obj - 0.2):
                best_vp = vp
                break
            
            # A good viewpoint is one where the ray either hits nothing
            # (the hits list is empty) or the first thing it hits is the object itself.
            if not raycast_results.hits or raycast_results.hits[0].hit_distance >= dist_to_obj - 0.2:
                best_vp = vp # Found a visible point
                break
        
        # If a valid point was found (even a non-visible one), add it
        if best_vp is not None:
            rotation = get_rotation_to_point(best_vp, object_pos_np)
            
            new_views.append({
                "agent_state": {"position": list(best_vp), "rotation": rotation}
            })
            
    return new_views

# Helper function (if you haven't added it yet)
def get_rotation_to_point(source_pos: np.ndarray, target_pos: np.ndarray) -> list:
    direction = target_pos - source_pos
    direction[1] = 0
    yaw = np.arctan2(direction[0], direction[2])
    half_yaw = yaw / 2.0
    return [0.0, np.sin(half_yaw), 0.0, np.cos(half_yaw)]

# Helper: extract goal position from episode
def get_goal(ep):
    vp = ep["view_points"][0]["agent_state"]["position"]
    return np.array(vp, dtype=float)

def get_rotation_to_point(
    source_pos: np.ndarray,
    target_pos: np.ndarray,
    add_random_noise: bool = False,
) -> list:
    """
    Generate a rotation quaternion to face from source to target, with optional noise.
    """
    direction = target_pos - source_pos
    direction[1] = 0  # Project onto the XZ plane for yaw calculation
    
    yaw = np.arctan2(direction[0], direction[2])

    if add_random_noise:
        # Add random noise, e.g., +/- 45 degrees (pi/4 radians)
        noise = np.random.uniform(-np.pi / 4, np.pi / 4)
        yaw += noise

    half_yaw = yaw / 2.0
    return [0.0, np.sin(half_yaw), 0.0, np.cos(half_yaw)]

def all_goals(ep):
    """Yield each possible goal position in this episode."""
    for vp in ep["view_points"]:
        yield np.array(vp["agent_state"]["position"], dtype=float)