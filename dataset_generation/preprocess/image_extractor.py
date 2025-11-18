# Imports
import os
import gzip
import json
import numpy as np
import habitat_sim
from PIL import Image
from itertools import groupby
from pathlib import Path
from typing import List, Tuple
import argparse

# Simulator settings
import habitat_sim

# Utils imports
from utils import sample_random_points, get_current_floor, get_obs_image, zip_image_folder


def parse_args():
    parser = argparse.ArgumentParser(description="Process Habitat scene files.")
    parser.add_argument("--split", type=str, default="val_seen", help="Dataset split (e.g., val_unseen, train_seen)")
    parser.add_argument("--use_img_annots", action="store_true", default=True, help="Use image annotations")
    parser.add_argument("--save_images", action="store_true", default=True, help="Save images")
    parser.add_argument("--save_json", action="store_true", default=True, help="Save JSON files")
    parser.add_argument("--single_floor_scenes", action="store_true", default=False, help="Process only single-floor scenes")
    parser.add_argument("--filter_unallowed", action="store_true", default=True, help="Filter unallowed objects")
    
    return parser.parse_args()

# Constants (from arguments)
args = parse_args()
USE_IMG_ANNOTS = args.use_img_annots
SAVE_IMAGES = args.save_images
SAVE_JSON = args.save_json
SINGLE_FLOOR_SCENES = args.single_floor_scenes
FILTER_UNALLOWED = args.filter_unallowed
SPLIT = args.split
SN_SPLIT = "val" if "val" in SPLIT else "train"

# Dynamically determine the DATA_PATH relative to the script's location
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
DIR_PATH = repo_root
DATA_PATH = os.path.abspath(os.path.join(DIR_PATH, "../data"))

BASE_SCENES_PATH = os.path.join(DATA_PATH, "scene_datasets", "hm3d_v0.2", SN_SPLIT)
FOLDER_PATH = os.path.join(DATA_PATH, "datasets", "goat_bench", "hm3d", "v3", SPLIT, "content")

#########
# DEBUG #
#########
#SAVE_IMAGES = False
#SAVE_JSON = False
#SINGLE_FLOOR_SCENES = False

# Habitat Simulator settings
SIM_SETTINGS = {
    "scene": None,             # To be set per scene
    "default_agent": 0,        # Index of the default agent
    "sensor_height": 1.0,      # Sensor height (meters)
    "width": 256,              # Observation width
    "height": 256,             # Observation height
    "hfov": 90,                # Horizontal field of view
}

REQUIRED_KEYS = {'position', 'view_points', 'object_category', 'image_goals'}

def main():
    """
    Main function: iterates over scene files, processes each scene, and saves the outputs.
    """
    # Read unallowed objects from a text file.
    unallowed_objects, unallowed_scenes = read_unallowed_objects_and_scenes(
        base_path=os.path.join(DIR_PATH, 'preprocess/unique_objects'), 
        objects_file_name='unallowed_objects.txt',
        scenes_file_name='unallowed_scenes.txt')
    
    n_goals = 0
    for file_name in os.listdir(FOLDER_PATH):
        
        # Skip non-JSON files.
        if not file_name.endswith(".json.gz"):
            continue
        
        scene_id = file_name.split('.')[0]
        file_path = os.path.join(FOLDER_PATH, file_name)
        
        # Process the scene to get grouped scene goals.
        grouped_goals = process_scene_file(file_path, scene_id)
        
        # Remove all goals that contain unallowed objects.
        if FILTER_UNALLOWED:
            
            # Filter scenes that are not in the unallowed scenes list.
            if scene_id in unallowed_scenes:
                print("Skipping unallowed scene:", scene_id)
                continue
            
            # Filter unallowed objects
            grouped_goals = [[goal for goal in floor if goal['object_category'] not in unallowed_objects] for floor in grouped_goals]
            grouped_goals = [floor for floor in grouped_goals if floor]

        # Filter invalid floors based on a text file. The idea is that we can only use floors that we mapped.
        # grouped_goals = filter_invalid_floors(repo_root, grouped_goals, scene_id)
        
        # Skip empty grouped goals. I.e. all floors are invalid so we directly skip the scene.
        if not grouped_goals:
            print("Skipping empty grouped goals for scene:", scene_id)
            continue
                    
        # Skip scenes with more than one floor if the flag is set.
        if SINGLE_FLOOR_SCENES and len(grouped_goals) > 1:
            continue
        
        # Save images if enabled.
        if USE_IMG_ANNOTS and SAVE_IMAGES:
            save_images(DIR_PATH, grouped_goals, scene_id)
        
        # Save JSON files if enabled.
        if SAVE_JSON:
            print(f"Directory: {DIR_PATH}")
            print(f"Save JSON file in root folder: {SPLIT, scene_id}")
            save_json(DIR_PATH, grouped_goals, scene_id)

        n_goals += sum(len(floor) for floor in grouped_goals)
    
    print("N° of goals:", n_goals)
    

def load_basis_glb_file(base_path: str, scene_name: str) -> str:
    """Locates and validates a .basis.glb file path for a given scene.
    
    Args:
        base_path: Root directory to search for scene folders
        scene_name: Name of the scene to load
        
    Returns:
        Full path to the validated .basis.glb file
        
    Raises:
        FileNotFoundError: If folder or file not found
    """
    base_dir = Path(base_path)
    folder_suffix = f"-{scene_name}"
    
    # Find first matching directory
    try:
        scene_folder = next(
            p for p in base_dir.iterdir()
            if p.is_dir() and p.name.endswith(folder_suffix)
        )
        print(scene_folder)
    except StopIteration:
        raise FileNotFoundError(
            f"No folder ending with '{folder_suffix}' found in {base_path}"
        ) from None

    # Validate file existence
    glb_path = scene_folder / f"{scene_name}.basis.glb"
    if not glb_path.exists():
        raise FileNotFoundError(
            f"Scene file {glb_path.name} not found in {scene_folder}"
        )

    return str(glb_path)

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

def create_filtered_objects(target_goal, floor_id, save_view_points=False):
    filtered_objects = {}
    filtered_objects['object_category'] = target_goal['object_category']
    filtered_objects['object_id'] = target_goal['object_id']
    if 'room' in target_goal:
        filtered_objects['room'] = target_goal['room']
    filtered_objects['floor_id'] = floor_id
    filtered_objects['to_discuss'] = False
    
    # Not all objects have a description
    # We can have multiple descriptions for the same object
    if 'lang_desc' in target_goal:
        filtered_objects['description'] = [target_goal['lang_desc'], "", ""]
    else:
        filtered_objects['description'] = ["", "", ""]
        
    filtered_objects['position'] = target_goal['position']
    
    if save_view_points:
        filtered_objects['view_points'] = target_goal['view_points']
    return filtered_objects

def read_unallowed_objects_and_scenes(
    base_path: str = 'preprocess/unique_objects',
    objects_file_name: str = 'unallowed_objects.txt',
    scenes_file_name: str = 'unallowed_scenes.txt'
) -> Tuple[List[str], List[str]]:
    """
    Read lists of unallowed objects and scenes from text files.

    Args:
        base_path (str): The base path where the files are located.
        objects_file_name (str): The name of the text file for unallowed objects.
        scenes_file_name (str): The name of the text file for unallowed scenes.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing two lists:
                                     - A list of unallowed object names.
                                     - A list of unallowed scene names.
    """
    objects_file_path = os.path.join(repo_root, base_path, objects_file_name)
    scenes_file_path = os.path.join(repo_root, base_path, scenes_file_name)

    with open(objects_file_path, 'r') as obj_file:
        unallowed_objects = [line.strip() for line in obj_file]

    with open(scenes_file_path, 'r') as scene_file:
        unallowed_scenes = [line.strip() for line in scene_file]

    return unallowed_objects, unallowed_scenes

def filter_invalid_floors(repo_root, grouped_goals, scene_id, txt_path="valid_floor_scenes.txt"):
    """
    Reads the valid floors from a text file, filters out invalid floors 
    from the 'grouped_goals' for the given 'scene_id', and returns the updated list.
    """
    # Build full path to the valid floors text file
    valid_floors_file = os.path.join(repo_root, 'preprocess/unique_objects', 'files', txt_path)
    
    # Read valid floors from file
    with open(valid_floors_file, 'r') as f:
        valid_floor_scenes = [line.strip() for line in f]
    
    # Collect current valid floors (assuming each nested list shares the same floor_id)
    current_valid_floors = [floor_data[0]['floor_id'] for floor_data in grouped_goals]
    
    # If there are no floors in 'grouped_goals', return an empty list
    if not current_valid_floors:
        print(f"Skipping scene '{scene_id}' with no floors at all.")
        return []
    
    # Build scene-floor strings and identify those that are invalid
    scene_and_floors = [f"{scene_id}_floor_{floor}" for floor in current_valid_floors]
    invalid_floors = [item for item in scene_and_floors if item not in valid_floor_scenes]
    
    # If all floors are invalid, return an empty list
    if len(invalid_floors) == len(scene_and_floors):
        print(f"Skipping scene '{scene_id}' with no valid floors.")
        return []
    
    # Filter out any invalid floors
    if invalid_floors:
        grouped_goals_ = [
            [goal for goal in floor if f"{scene_id}_floor_{goal['floor_id']}" not in invalid_floors]
            for floor in grouped_goals
        ]
        # Remove any empty floor lists
        grouped_goals_ = [fg for fg in grouped_goals_ if fg]
        return grouped_goals_

    return grouped_goals

def process_scene_file(file_path, scene_id):
    """
    Process a single scene file and return the grouped scene goals.
    
    Args:
        file_path (str): Path to the .json.gz file.
        scene_id (str): Identifier for the scene.
    
    Returns:
        list: A list of lists where each inner list contains goals grouped by floor.
    """
    scene_path = load_basis_glb_file(BASE_SCENES_PATH, scene_id)
    print('----------------------------')
    print("Current scene check:", os.path.basename(os.path.dirname(scene_path)))
    print('----------------------------')
    
    # Read the scene objects from the compressed JSON file.
    with gzip.open(file_path, 'rt', encoding='utf-8') as gz_file:
        scene_objects = json.load(gz_file)
    
    # Setup the Habitat Simulator for the current scene.
    SIM_SETTINGS["scene"] = scene_path
    cfg = make_simple_cfg(SIM_SETTINGS)
    
    # Close any previous simulator instance.
    try:
        sim.close()
    except NameError:
        pass
    sim = habitat_sim.Simulator(cfg)
    
    # Initialize the agent if image annotations are used.
    if USE_IMG_ANNOTS:
        agent = sim.initialize_agent(SIM_SETTINGS["default_agent"])
        agent_state = habitat_sim.AgentState()
    else:
        agent = None
        agent_state = None
    
    scene_goals = []
    # Iterate over each goal within the scene.
    for _, goals in scene_objects.items():
        for target_goal in goals:
            try:
                # Use the first view point to determine the floor.
                vp_height = target_goal['view_points'][0]['agent_state']['position'][1]
                floor_id = get_current_floor(vp_height, sample_random_points(sim))
            except Exception as e:
                # Here instead of skipping the image we could take the object position
                # SCENE: 00803-k1cupFYWXJ6 we have to skip due to scene irregularities (it's a fucking castle)
                print("View point for floor extraction not found:", e)
                continue
            
            # Create a filtered object based on the target goal.
            filtered_object = create_filtered_objects(target_goal, floor_id)
            
            # If image annotations are enabled, process and add images.
            if USE_IMG_ANNOTS and agent is not None and agent_state is not None:
                image_goals = target_goal.get('image_goals', [])
                if not image_goals:
                    continue

                # Get images sorted by distance (using Euclidean norm)
                top_x_dist = sorted(
                    image_goals,
                    key=lambda x: np.linalg.norm(np.array(x['position']) - np.array(target_goal['position'])) , reverse=True
                )[:6]  # Select top 4 distant images

                # Get images with highest object and frame coverage.
                top_cov = sorted(top_x_dist, key=lambda x: x.get('object_coverage', 0), reverse=True)[0]
                top_frame = sorted(top_x_dist, key=lambda x: x.get('frame_coverage', 0), reverse=True)[0]
                
                # Ensure that there are at least two images from the distance sorting.
                if len(top_x_dist) < 2:
                    top_x_dist = top_x_dist * 2

                # Retrieve observation images.
                obs = []
                for candidate in [top_x_dist[0], top_x_dist[1], top_cov, top_frame]:
                    obs_image = get_obs_image(candidate, SIM_SETTINGS, agent_state, agent, sim)
                    obs.append(obs_image)
                filtered_object['image'] = obs

            scene_goals.append(filtered_object)
    
    # Group the scene goals by floor_id.
    scene_goals_sorted = sorted(scene_goals, key=lambda x: x['floor_id'])
    grouped_goals = [list(group) for _, group in groupby(scene_goals_sorted, key=lambda x: x['floor_id'])]
    
    sim.close()
    return grouped_goals

def save_images(base_path, scene_goals, scene_id):
    """
    Save observation images to disk and zip the image folder.
    
    Args:
        scene_goals (list): Grouped scene goals.
        scene_id (str): Identifier for the scene.
    """
    images_folder = os.path.join(base_path, "../data", "datasets", "eai_pers", SPLIT, scene_id, "images")
    os.makedirs(images_folder, exist_ok=True)
    
    for floor in scene_goals:
        for obj in floor:
            for i, image in enumerate(obj.get('image', [])):
                pil_image = Image.fromarray(image)
                image_filename = f"{obj['object_id']}_{i}.png"
                image_path = os.path.join(images_folder, image_filename)
                pil_image.save(image_path)
            # Remove image data from the dictionary after saving.
            if 'image' in obj:
                del obj['image']
    
    # Zip the images folder (function assumed to be defined elsewhere).
    zip_image_folder(images_folder, remove_unzipped=False)

def save_json(base_path, scene_goals, scene_id, by_floor=False):
    """
    Save each floor's scene goals to separate JSON files.
    
    Args:
        scene_goals (list): Grouped scene goals.
        scene_id (str): Identifier for the scene.
    """
    scene_folder = os.path.join(base_path, "../data", "datasets", "eai_pers", SPLIT, scene_id)
    os.makedirs(scene_folder, exist_ok=True)
    
    if by_floor:  
        # Check if this is a valid floor & scene
        for i, floor in enumerate(scene_goals):
            json_path = os.path.join(scene_folder, f"{scene_id}_floor_{scene_goals[i][0]['floor_id']}.json")
            with open(json_path, 'w') as f:
                json.dump(floor, f)
    else:
        # convert scene_goal into a single list
        scene_goals = [goal for floor in scene_goals for goal in floor]
        
        # Save all floors in a single JSON file.
        json_path = os.path.join(scene_folder, f"{scene_id}.json")
        with open(json_path, 'w') as f:
            json.dump(scene_goals, f)


if __name__ == "__main__":
    main()