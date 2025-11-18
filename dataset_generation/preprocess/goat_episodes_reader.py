import os
import json
from typing import Dict, List, Optional, Tuple
from utils import read_json_gz, save_filtered_metadata
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Goat episodes reader")
    parser.add_argument("--split", type=str, default="val_unseen", help="Split")
    parser.add_argument("--goat_path", type=str, default="../../data/datasets/goat_bench/hm3d/", help="Goat path")
    parser.add_argument("--version", type=str, default="v1/", help="Version")
    parser.add_argument("--eai_pers_path", type=str, default="data/datasets/eai-pers/", help="EAI pers path")
    parser.add_argument("--create_filtered_files", action="store_true", default=True, help="Enable creation of filtered files")
    parser.add_argument("--create_metadata_file", action="store_true", default=False, help="Enable creation of metadata file")
    return parser.parse_args()
      
def main():
    args = parse_args()
    content_dir = os.path.join(os.path.abspath(args.goat_path), args.version, args.split, 'content')
    filtered_output_dir = os.path.join(args.goat_path, "v2", args.split, 'content')
    metadata_output_dir = os.path.join(args.goat_path, "v2", args.split)
    
    total_objects = 0
    global_metadata = []
    
    objects_name = []
    for scene_file in os.listdir(content_dir):
        scene_path = os.path.join(content_dir, scene_file)
        
        # Process only .json.gz files
        if not scene_path.endswith('.json.gz'):
            continue
        
        print(f"Processing scene: {scene_path}")
        
        filtered_goals, episodes = process_scene(scene_path, args)
        
        # Increment object count from filtered goals
        for objects in filtered_goals.values():
            
            total_objects += len(objects)
            objects_name.append((objects[0]['object_category'], len(objects)))
        
        if args.create_filtered_files:
            save_filtered_metadata(filtered_goals, filtered_output_dir, scene_file)
        
        if args.create_metadata_file and episodes is not None:
            for ep in episodes:
                # Remove the 'tasks' key from each episode if it exists
                if 'tasks' in ep:
                    del ep['tasks']
            global_metadata.extend(episodes)
    
    if args.create_metadata_file:
        if not os.path.exists(metadata_output_dir):
            os.makedirs(metadata_output_dir)
        metadata_file = os.path.join(metadata_output_dir, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(global_metadata, f)

def process_objects(total_objects, objects_list, base_path="eai_pers_dataset/preprocess/unique_objects", split="val_unseen"):
    """
    Process a list of objects with their counts, print totals, and save unique objects to a JSON file.

    Args:
        total_objects (int): Total number of objects.
        objects_list (list of tuple): Each tuple contains (object_name, count).
        base_path (str): The base path where the output directory is located.
        split (str): Identifier for the split, used as the output filename.
    """
    print('Total objects:', total_objects)
    
    # Create a set of unique object names and sum their counts.
    unique_objects = [
        (name, sum(count for n, count in objects_list if n == name))
        for name in set(name for name, count in objects_list)
    ]
    
    # Sort unique objects by total count in descending order.
    unique_objects = sorted(unique_objects, key=lambda x: x[1], reverse=True)
    print('Unique objects:', len(unique_objects))
    
    # Convert list of tuples to list of dictionaries.
    unique_objects_dicts = [{'object_category': name, 'count': count} for name, count in unique_objects]
    
    # Build output file path.
    output_dir = os.path.join(base_path)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{split}.json')
    
    # Write the data to a JSON file.
    with open(output_file, 'w') as f:
        json.dump(unique_objects_dicts, f)
    
    print("Data saved to", output_file)

def filter_goals(goals: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    """
    Filter goal objects to retain only those that contain 'image_goals'.

    Args:
        goals (Dict[str, List[Dict]]): A dictionary where keys are goal identifiers
            and values are lists of goal objects.

    Returns:
        Dict[str, List[Dict]]: A filtered dictionary containing only the goals with 'image_goals'.
    """
    filtered = {
        goal: [obj for obj in objects if 'image_goals' in obj]
        for goal, objects in goals.items()
    }
    # Remove any entries that have become empty after filtering
    return {goal: objs for goal, objs in filtered.items() if objs}

def process_scene(scene_path: str, args) -> Tuple[Dict[str, List[Dict]], Optional[List[Dict]]]:
    """
    Process a single scene file: filter metadata and optionally retrieve episodes.

    Args:
        scene_path (str): The path to the gzipped scene JSON file.

    Returns:
        Tuple[Dict[str, List[Dict]], Optional[List[Dict]]]:
            - A dictionary of filtered goals.
            - A list of episodes if CREATE_METADATA_FILE is True; otherwise, None.
    """
    # Load the full scene JSON content from the gzipped file
    scene_data = read_json_gz(scene_path)

    # Extract and filter "goals" from the scene metadata
    goals = scene_data.get('goals', {})
    filtered_goals = filter_goals(goals)

    # Optionally retrieve episodes based on the global flag
    episodes = scene_data.get('episodes', []) if args.create_metadata_file else None

    return filtered_goals, episodes


if __name__ == "__main__":
    main()

