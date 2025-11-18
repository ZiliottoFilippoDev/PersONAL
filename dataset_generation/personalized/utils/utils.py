import json
import openai
import os
import gzip
from typing import List, Dict, Any

def generate_single_response(prompt, model_type="gpt-4o-mini"):
    """
    Generates direct response to a prompt using the OpenAI Chat API.
    This is NOT ment for BATCH API usage.
    """
    
    response = openai.ChatCompletion.create(
        model=model_type,
        messages=[
            {"role": "system", "content": "You are an helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    
    return response.choices[0].message.content

def get_scene_path(split, name):
    # Get the scene path for a given split and name
    split = "val" if "val" in split else "train"
    
    base_path = f"data/scene_datasets/hm3d_v0.2/{split}"
    for folder_name in os.listdir(base_path):
        if folder_name.endswith(f"-{name}"):
            number = folder_name.split('-')[0]
            return f"hm3d_v0.2/{split}/{number}-{name}/{name}.basis.glb"
    return None

def write_retrieval_episodes(episodes, scene_id, split, level, base_dir="data/datasets/eai_pers"):
    
    # Define the output path for the JSON file
    if split in ["train"]:
        base_dir = f"/Users/filippoziliotto/Desktop/Repos/eai-pers/data/v2_5/splits/train/{level}"
    elif split in ["val_unseen"]:
        base_dir = f"/Users/filippoziliotto/Desktop/Repos/eai-pers/data/v2_5/splits/val/{level}"
        
    # Check if the base directory exists, if not create it      
    output_path = os.path.join(base_dir, scene_id, level, "episodes.json")
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write the episodes to a JSON file
    with open(output_path, 'w') as json_file:
        json.dump(episodes, json_file, indent=2)
        
    return

def write_active_episodes_(
    episodes: List[Dict[str, Any]],
    scene_id: str,
    split: str,
    level: str,
    base_dir: str = "/Users/filippoziliotto/Desktop/Repos/habitat-lab-v0/data/datasets/eai_pers/active"
) -> None:
    episodes = episodes[0]
    
    assert "val" in split
    output_dir = os.path.join(base_dir, "val", level, "content")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{scene_id}.json.gz")

    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        json.dump(episodes, f, indent=2)
        
    return

def groupby_write_active_episode(
    episodes: List[Dict[str, Any]],
    split: str,
    level: str,
    base_dir: str = "/Users/filippoziliotto/Desktop/Repos/habitat-lab-v0/data/datasets/eai_pers/active"
) -> None:
    
    # Groupby scene_id
    grouped_episodes = {}
    for episode in episodes:
        scene_id = episode["episodes"][0]["scene_id"].split("/")[-1].split(".")[0]  # Extract scene_id from the full path
        if scene_id not in grouped_episodes:
            grouped_episodes[scene_id] = []
        grouped_episodes[scene_id].append(episode)
        
    # Write each scene's episodes to a separate file
    for scene_id, scene_episodes in grouped_episodes.items():
        write_active_episodes_(scene_episodes, scene_id, split, level, base_dir)
        
    return

def groupby_write_passive_episode(
    episodes: List[Dict[str, Any]],
    split: str,
    level: str,
    base_dir: str = "/Users/filippoziliotto/Desktop/Repos/habitat-lab-v0/data/datasets/eai_pers/passive"
) -> None:
    
    # Groupby scene_id
    grouped_episodes = {}
    for episode in episodes:
        scene_id = episode["scene_id"].split("/")[-1].split(".")[0]  # Extract scene_id from the full path
        if scene_id not in grouped_episodes:
            grouped_episodes[scene_id] = []
        grouped_episodes[scene_id].append(episode)
        
    # Write each scene's episodes to a separate file
    for scene_id, scene_episodes in grouped_episodes.items():
        write_active_episodes_(scene_episodes, scene_id, split, level, base_dir)
        
    return

def parse_single_batch_output(json_string: str) -> dict:
    """
    Parses the given JSON string and returns a dictionary.
    
    Args:
        json_string (str): The raw JSON output as a string.
    
    Returns:
        dict: Parsed JSON data as a Python dictionary.
    """
    try:
        parsed_data = json.loads(json_string)
        return parsed_data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        raise ValueError("Error decoding JSON data.")
        return {}  # Return an empty dictionary in case of error
