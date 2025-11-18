import os
import gzip
import json
import random
from typing import List, Dict, Any, Optional
import habitat_sim
import numpy as np

# Local imports
from habitat_tf.utils import (
    make_simple_cfg, random_yaw_rotation, sample_additional_viewpoints, 
    euclidean_distance, load_merged_scene_data, build_lookups, all_goals, get_rotation_to_point
)

# Setup the Habitat Simulator for the current scene.
SIM_SETTINGS = {
    "scene": None,             # To be set per scene
    "default_agent": 0,        # Index of the default agent
    "sensor_height": 1.0,      # Sensor height (meters)
    "width": 256,              # Observation width
    "height": 256,             # Observation height
    "hfov": 90,                # Horizontal field of view
}


def initialize_simulator(scene_id):
    
    scene_id = get_scene_path("val", scene_id)
    SIM_SETTINGS["scene"] = os.path.join(scene_id)
    sim_cfg = make_simple_cfg(SIM_SETTINGS, use_equirectangular=False)
    sim = habitat_sim.Simulator(sim_cfg)
    return sim

def get_scene_path(split, name):
    # Get the scene path for a given split and name
    split = "val" if "val" in split else "train"
    
    base_path = f"data/scene_datasets/hm3d_v0.2/{split}"
    for folder_name in os.listdir(base_path):
        if folder_name.endswith(f"-{name}"):
            number = folder_name.split('-')[0]
            return f"data/scene_datasets/hm3d_v0.2/{split}/{number}-{name}/{name}.basis.glb"
    return None


def sample_navigable_points(
    sim: habitat_sim.Simulator,
    episodes: list,
    max_tries: int = 100,
    max_geodesic: float = 18.0,  # Minimum geodesic distance from start to goal
    max_euclidean: float = 15.0,  # Minimum planar euclidean distance from start to goal
    min_geodesic: float = 3.0,  # Minimum geodesic distance from start to goal
    max_height_diff: float = 0.5,
    max_closest_height_diff: float = 1.0,
    use_viewpoints: bool = False,
    extra_vp_count: int = 30,
) -> list:
    """
    Samples valid navigable start points for each episode, computes distances,
    and optionally augments view points.
    
    :param sim: Initialized Habitat Simulator instance.
    :param episodes: List of episode dicts containing "view_points" and "object_pos".
    :param scene_dir: Base directory for scene files.
    :param scene_id: Identifier for the scene file.
    :param max_tries: Max attempts for sampling a valid point.
    :param min_geodesic: Minimum required geodesic distance from start to goal.
    :param max_height_diff: Maximum allowed height difference between start and goal.
    :param use_viewpoints: Whether to sample additional viewpoints.
    :param extra_vp_count: Number of extra viewpoints to sample if use_viewpoints is True.
    :return: Updated list of episodes with start positions, rotations, and distances.
    """
    
    # 0b) Initialize the pathfinder
    shortest_path = habitat_sim.ShortestPath()
    
    # 0c) Initialize episodes
    for ep in episodes:
        
        # If no view_point is found sample from object position
        if len(ep.get("view_points", [])) == 0:
            ep["view_points"].extend(
                sample_additional_viewpoints(sim, ep["object_pos"], count=extra_vp_count*2, max_tries=max_tries, radius=1.0)
            )
            ep["closest_view_point"] = min(
                ep["view_points"],
                key=lambda v: euclidean_distance(
                    np.array(v["agent_state"]["position"], dtype=float),
                    np.array(ep["object_pos"], dtype=float),
                    dims = (0,1,2)
                )
            )["agent_state"]["position"]
        
        # skip fully‐valid episodes
        eucl = ep.get("euclidean_distance", float("inf"))
        geo  = ep.get("geodesic_distance",  float("inf"))
        # skip episodes already within the max‐distance envelope            
        if "start_position" in ep and eucl <= max_euclidean and min_geodesic <= geo <= max_geodesic:
            continue

        object_pos = np.array(ep["object_pos"], dtype=float)

        # extract the closest view‐point’s Y once
        closest_y = ep["closest_view_point"][1]

        global_best = {"geo": -1.0, "euclid": -1.0, "pos": None}
        chosen = None

        for goal_pos in all_goals(ep):  # iterate each candidate view_point
            best_for_goal = {"geo": -1.0, "euclid": -1.0, "pos": None}

            for _ in range(max_tries):
                nav = sim.pathfinder.get_random_navigable_point(max_tries=max_tries)
                start_pos = np.array([nav.x, nav.y, nav.z], dtype=float)

                # 1) enforce BOTH height constraints
                if abs(start_pos[1] - goal_pos[1]) > max_height_diff:
                    continue
                if abs(start_pos[1] - closest_y) > max_closest_height_diff:
                    continue

                # 2) geodesic
                shortest_path.requested_start = start_pos.tolist()
                shortest_path.requested_end   = goal_pos.tolist()
                sim.pathfinder.find_path(shortest_path)
                g = shortest_path.geodesic_distance
                if g == float("inf"):
                    continue

                # 3) planar euclidean
                e = float(np.linalg.norm(start_pos[[0,2]] - object_pos[[0,2]]))

                # update best_for_goal (within full [min_geodesic, max_geodesic] ∧ euclid)
                if e <= max_euclidean and min_geodesic <= g <= max_geodesic and g > best_for_goal["geo"]:
                    best_for_goal.update({"geo": g, "euclid": e, "pos": start_pos.copy()})

                # immediate accept if e ≤ max_euclidean AND g ∈ [min_geodesic, max_geodesic]
                if e <= max_euclidean and min_geodesic <= g <= max_geodesic:
                    chosen = {"pos": start_pos, "geo": g, "euclid": e}
                    break

            if chosen:
                break

            # no perfect sample for this goal → update global_best
            if best_for_goal["pos"] is not None and best_for_goal["geo"] > global_best["geo"]:
                global_best = best_for_goal

        # fallback if needed
        if chosen is None:
            if global_best["pos"] is None:
                # --- SECONDARY FALLBACK: drop geodesic+euclidean thresholds entirely, keep height only
                fallback_best = {"geo": -1.0, "pos": None}
                for _ in range(max_tries):
                    nav = sim.pathfinder.get_random_navigable_point(max_tries=max_tries)
                    sp = np.array([nav.x, nav.y, nav.z], dtype=float)
                    # same floor + closest‐view height
                    if abs(sp[1] - goal_pos[1]) > max_height_diff:  
                        continue
                    if abs(sp[1] - closest_y) > max_closest_height_diff:
                        continue

                    shortest_path.requested_start = sp.tolist()
                    shortest_path.requested_end   = goal_pos.tolist()
                    sim.pathfinder.find_path(shortest_path)
                    g = shortest_path.geodesic_distance
                    if g == float("inf"):
                        continue

                    if g > fallback_best["geo"]:
                        fallback_best.update({"geo": g, "pos": sp.copy()})

                if fallback_best["pos"] is None:
                    raise RuntimeError(
                        f"Even fully relaxed sampling failed for ep {ep['scene_id']} obj {ep['object_id']}"
                    )
                chosen = {
                    "pos":   fallback_best["pos"],
                    "geo":   fallback_best["geo"],
                    # recompute planar Euclid once for record:
                    "euclid": float(np.linalg.norm(
                        (fallback_best["pos"][[0,2]] - object_pos[[0,2]])
                    ))
                }
            else:
                chosen = global_best

        # commit
        ep["start_position"]     = chosen["pos"].tolist()
        ep["geodesic_distance"]  = chosen["geo"]
        ep["euclidean_distance"] = chosen["euclid"]
        object_pos_np = np.array(ep["object_pos"], dtype=float)
        ep["start_rotation"] = get_rotation_to_point(
            chosen["pos"],
            object_pos_np,
            add_random_noise=True
        )

        # optionally augment
        if use_viewpoints and len(ep["view_points"]) < 40:
            base_vp = ep["view_points"][0]["agent_state"]["position"]
            ep["view_points"].extend(
                sample_additional_viewpoints(sim, base_vp, count=extra_vp_count, max_tries=max_tries)
            )

    return episodes


def prepare_episode_data(
    sim: habitat_sim.Simulator,
    episodes: List[Dict[str, Any]],
    base_path: str = "data/datasets/goat_bench/hm3d/v1/",
    level="easy",
    use_view_points: bool = False
) -> List[Dict[str, Any]]:
    """
    For each episode in `episodes`, attach:
      - 'start_position' and 'start_rotation' (always)
      - 'view_points' (only if use_view_points=True)

    Loads and merges scene_data from the three splits:
      "val_seen", "val_seen_synonyms", "val_unseen"
    into a single combined scene_data per scene_id.
    """
    assert sim is not None, "Simulator must be initialized before preparing episodes."
    
    if not episodes:
        raise ValueError("episodes list must not be empty")

    splits = ["val_seen", "val_seen_synonyms", "val_unseen"]
    if level == "hard":
        max_geo_dist = 11.0
        min_geo_dist = 3.0
    elif level == "medium":
        max_geo_dist = 7.0
        min_geo_dist = 3.0
    elif level == "easy":
        max_geo_dist = 4.0
        min_geo_dist = 2.0
    
    # Prepare empty merged containers
    merged_goals = load_merged_scene_data(
        base_path=base_path,
        scene_id=episodes[0]["scene_id"],
        splits=splits
    )

    # Build lookups from merged data
    view_point_lookup, start_lookup = build_lookups(
        merged=merged_goals,
        use_view_points=use_view_points
    )
        
    # Finally, attach to each input episode
    for ep in episodes:
        obj_id = ep["object_id"]

        # view_points
        if use_view_points:
            ep["view_points"] = view_point_lookup.get(obj_id)
            if ep["view_points"] is None: ep["view_points"] = []
    
            # Save the closet view_point to the object position
            try:
                closest_vp = min(
                    ep["view_points"],
                    key=lambda v: euclidean_distance(
                        np.array(v["agent_state"]["position"], dtype=float),
                        np.array(ep["object_pos"], dtype=float),
                        dims=(0,1,2)
                    ),
                )
                ep["closest_view_point"] = closest_vp["agent_state"]["position"]
            except ValueError:
                # In the case no view_point is defined
                continue

        # start_position & rotation
        candidates = start_lookup.get(obj_id) or []

        if candidates:
            USE_FURTHERST = False
            if USE_FURTHERST:
                # Check that the starting pos is at least 3 units meters from the object position
                max_distance = -1
                best_choice = None
                for choice in candidates:
                    distance = euclidean_distance(
                        np.array(choice["pos"], dtype=float),
                        np.array(ep["closest_view_point"], dtype=float)
                    )
                    if distance > max_distance:
                        max_distance = distance
                        best_choice = choice
                    
            else:
                # Instead of finding the best choice, select one randomly
                best_choice = random.choice(candidates)
                max_distance = euclidean_distance(
                    np.array(best_choice["pos"], dtype=float),
                    np.array(ep["closest_view_point"], dtype=float)
                )
            
            ep["start_position"] = best_choice["pos"]
            ep["start_rotation"] = best_choice["rot"]
            ep["euclidean_distance"] = max_distance
        
    # Use Haibtat-sim to sample a navigable point
    episodes = sample_navigable_points(
        sim=sim,
        episodes=episodes, 
        max_tries= 200,
        max_geodesic=max_geo_dist,
        min_geodesic=min_geo_dist,
        max_height_diff= 0.5,
        use_viewpoints= use_view_points,
        extra_vp_count = 15,
    )
    
    # Delete closest_view_point
    for ep in episodes:
        if "closest_view_point" in ep:
            del ep["closest_view_point"]
            
    return episodes




