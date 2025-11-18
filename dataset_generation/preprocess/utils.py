import numpy as np
from typing import Dict, List, Optional, Any
import habitat_sim
import shutil
import json
import gzip
import os
import quaternion
# import cv2  # Optional: for advanced image stitching

"""
Json utils
"""
def read_json_gz(file_path: str) -> Any:
    """
    Read and return JSON content from a gzip file.

    Args:
        file_path (str): The path to the gzip file.

    Returns:
        Any: The JSON data loaded from the file.
    """
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        return json.load(f)

def save_filtered_metadata(filtered_goals: Dict[str, List[Dict]], output_dir: str, filename: str) -> None:
    """
    Save filtered goals metadata into a gzip-compressed JSON file.

    Args:
        filtered_goals (Dict[str, List[Dict]]): The filtered goals metadata.
        output_dir (str): The directory where the file will be saved.
        filename (str): The name of the output file.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    with gzip.open(output_path, 'wt', encoding='utf-8') as f:
        json.dump(filtered_goals, f, indent=2)
  
  
"""
Utils to sample points from different
floors since HM3D-sem is not fully annotated
"""
def sample_random_points(
    sim: habitat_sim.Simulator,
    volume_sample_fac: float = 5.0,
    significance_threshold: float = 0.2
) -> Dict[float, np.ndarray]:
    """Sample navigable points and cluster them by floor height using histogram analysis.
    
    Args:
        sim: Habitat simulator instance
        volume_sample_fac: Multiplier for number of points to sample based on scene volume
        significance_threshold: Minimum point percentage to consider a floor level
        
    Returns:
        Dictionary mapping floor heights to corresponding navigable points
    """
    scene = sim.get_active_scene_graph().get_root_node()
    scene_volume = scene.cumulative_bb.size().product()
    
    # Generate navigable points
    num_samples = int(scene_volume * volume_sample_fac)
    points = np.array([sim.pathfinder.get_random_navigable_point() for _ in range(num_samples)])
    
    # Cluster points by vertical height using histogram
    counts, bin_edges = np.histogram(points[:, 1], bins='auto')
    significant_bins = counts / len(points) > significance_threshold
    
    left_edges = bin_edges[:-1][significant_bins]
    right_edges = bin_edges[1:][significant_bins]
    
    points_floors = {}
    for left, right in zip(left_edges, right_edges):
        in_bin = (points[:, 1] >= left) & (points[:, 1] <= right)
        points_floor = points[in_bin]
        height = points_floor[:, 1].mean()
        points_floors[height] = points_floor
        
    return points_floors

def floors_num(
    sim: habitat_sim.Simulator,
    volume_sample_fac: float = 1.0,
    significance_threshold: float = 0.2
) -> int:
    """Count the number of detectable floor levels in the scene."""
    return len(sample_random_points(sim, volume_sample_fac, significance_threshold))

def get_floor_levels(
    current_height: float,
    floor_points: Dict[float, np.ndarray]
) -> Dict[str, List[Optional[float]]]:
    """Identify floors immediately above and below the current height level.
    
    Args:
        current_height: The height to compare against floor levels
        floor_points: Dictionary mapping floor heights to their corresponding points
        
    Returns:
        Dictionary with:
        - upper_level: List containing height of next floor above
        - current_floor: List containing current floor height
        - down_level: List containing height of next floor below
    """
    closest_key = min(floor_points.keys(), key=lambda k: abs(k - current_height))
    
    downstairs_keys = [k for k in floor_points.keys() if k < closest_key]
    upstairs_keys = [k for k in floor_points.keys() if k > closest_key]
    
    down_level_key = max(downstairs_keys) if downstairs_keys else None
    upper_level_key = min(upstairs_keys) if upstairs_keys else None
    
    return {
        'upper_level': [upper_level_key] if upper_level_key is not None else [],
        'current_floor': [closest_key],
        'down_level': [down_level_key] if down_level_key is not None else []
    }

def get_current_floor(
    current_height: float,
    floor_points: Dict[float, np.ndarray]
) -> int:
    """Convert height measurement to ordinal floor number.
    
    Args:
        current_height: The height to compare against floor levels
        floor_points: Dictionary mapping floor heights to their corresponding points
        
    Returns:
        The index of the closest floor level (0-based)
    """
    closest_key = min(floor_points.keys(), key=lambda k: abs(k - current_height))
    return list(floor_points.keys()).index(closest_key)

"""
Image utils
"""
def zip_image_folder(folder_to_zip, remove_unzipped=True):
    
    # Zip the folder
    shutil.make_archive(folder_to_zip, 'zip', folder_to_zip)
    
    # Remove the unzipped version
    if remove_unzipped:
        shutil.rmtree(folder_to_zip)
        
def get_obs_image(max_key, sim_settings, agent_state, agent, sim):
    max_key['position'][1] -= sim_settings["sensor_height"]
    agent_state.position = np.array(max_key['position'])
    agent_state.rotation = np.array(max_key['rotation'])
    agent.set_state(agent_state)
    
    # RGB observation
    return sim.get_sensor_observations()['rgb']


def get_360_composite_image(sim, sim_settings, agent_state, agent, base_rotation, num_views=20):
    """
    Captures multiple images at fixed angular increments around the agent using quaternion rotations 
    and stitches them together.
    
    Args:
        sim: The habitat simulator instance.
        sim_settings: Simulation settings dictionary.
        agent_state: The current state of the agent.
        agent: The agent instance.
        base_rotation (quaternion.quaternion): The starting rotation as a quaternion.
        num_views (int): Number of images to capture around the 360° circle.
    
    Returns:
        np.array: The composite 360° image.
    """
    images = []
    angle_increment = 360.0 / num_views

    # Save the original rotation (make a copy)
    original_rotation = base_rotation.copy()

    for i in range(num_views):
        # Compute the yaw angle in radians for this view.
        yaw_angle_rad = np.deg2rad(angle_increment * i)
        
        # Create a quaternion representing the yaw rotation about the Y-axis.
        # Quaternion format: cos(theta/2) + sin(theta/2)*(0*i + 1*j + 0*k)
        yaw_quat = quaternion.quaternion(np.cos(yaw_angle_rad / 2), 0, 0, np.sin(yaw_angle_rad / 2))
        
        # Calculate the new rotation.
        # The order of multiplication matters. Here, we apply the yaw rotation relative to base.
        new_rotation = base_rotation * yaw_quat
        
        # Update the agent's rotation in agent_state.
        # Convert the quaternion to an array in the order [w, x, y, z].
        agent_state.rotation = np.array([new_rotation.w, new_rotation.x, new_rotation.y, new_rotation.z])
        agent.set_state(agent_state)
        
        # Capture the image from the simulator.
        obs = sim.get_sensor_observations()
        image = obs['rgb']
        images.append(image)
    
    # Restore the original rotation.
    agent_state.rotation = np.array([original_rotation.w, original_rotation.x, original_rotation.y, original_rotation.z])
    agent.set_state(agent_state)
    
    # Stitch the images:
    # Option 1: Simple horizontal concatenation if perspective distortion is minimal.
    composite_image = np.hstack(images)
    
    # Option 2: For more seamless stitching, use OpenCV's stitching module:
    # stitcher = cv2.Stitcher_create()
    # status, composite_image = stitcher.stitch(images)
    # if status != cv2.Stitcher_OK:
    #     print("Error during stitching:", status)
    #     composite_image = None

    return composite_image




