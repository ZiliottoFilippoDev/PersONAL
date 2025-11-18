from personalized.utils.utils import (
    get_scene_path, write_retrieval_episodes, groupby_write_active_episode, groupby_write_passive_episode
)
from personalized.utils.names import NAMES
from habitat_tf.nav_episode import prepare_episode_data, initialize_simulator
import argparse
import os 
import json
import re
import random
import gzip
from collections import defaultdict
from typing import List, Dict, Any, Set


MAX_EPISODES = {
    "easy": 600,
    "medium": 700,
    "hard": 700
}

def main(args):
    """
    This script loads stored batch API output from upload_batch.ipynb
    and generates episodes for the personalized dataset.
    """
    
    # load the batch API output
    batch_response = process_batch_api(
        file_name=f"{os.path.join(args.base_json_path, args.batch_file_name)}_{args.data_split}.jsonl"
    )
    
    # Loop over each scene in the split
    n_episodes = 0
    all_episodes = []
    if args.data_split == "easy":
        assert args.split in ["val_seen"]
    elif args.data_split in ["medium"]:
        assert args.split in ["val_seen_merged"]
    elif args.data_split in ["hard"]:
        assert args.split in ["val"]
        
    # Response before generating episodes
    response_file = []
    all_active_nav_episodes = []
        
    for num_scenes, scene_name in enumerate(os.listdir(os.path.join(args.base_path, args.split))):
        scene_path = os.path.join(args.base_path, args.split, scene_name)
        
        # Delete some scenes
        if scene_name in ["yr17PDCnDDW","eF36g7L6Z9M"]:
            continue
        
        active_nav_episodes = []
        if os.path.isdir(scene_path):
            
            print("-------------")
            print(f"Processing scene {num_scenes + 1}/{len(os.listdir(os.path.join(args.base_path, args.split)))}: {scene_name}")
            print("-------------")
            
            # Initialize simulator:
            if args.add_nav_data:     
                    try: sim.close()
                    except: pass
                    sim = initialize_simulator(
                        scene_id=scene_name,
                    )
            
            # Loop over all the JSON files in the scene folder
            for json_file in os.listdir(scene_path):
                json_path = os.path.join(scene_path, json_file)
                
                # Skip scenes with episode already created
                if json_path.endswith(".json") and "episodes" not in json_path:
                    with open(json_path, 'r') as file:
                        json_content = json.load(file)
                        
                    # Group dicts by floor_id
                    floor_groups = {}
                    for item in json_content:
                        floor = item.get("floor_id")
                        if floor not in floor_groups:
                            floor_groups[floor] = []
                        floor_groups[floor].append(item)
                        
                    #if scene_name == "TEEsavR23oF":
                    #    print("TEEsavR23oF")
                    #    pass

                    # Generate scene id
                    unique_id_base = json_path.split("/")[-1].split(".")[0]
                    multiple_floor_episodes = []
                    for floor, items in floor_groups.items():
                        # Append floor info to unique_id
                        unique_id = f"{unique_id_base}_floor_{floor}"
                        
                        print(f"Processing floor {floor} with unique_id: {unique_id}")
                        
                        # 1) Find ALL splits for this floor
                        matching = [
                            item for item in batch_response
                            if item["custom_id"].startswith(unique_id)
                        ]

                        # 2) Sort by split index (assumes suffix "_split_{idx}")
                        def split_idx_from_id(entry):
                            cid = entry["custom_id"]
                            # if no "_split_", treat as idx=0
                            if "_split_" not in cid:
                                return 0
                            return int(cid.rsplit("_split_", 1)[1])
                        matching = sorted(matching, key=split_idx_from_id)
                    
                        # 3) For each split, process and generate episodes
                        for split_entry in matching:
                            sid = split_entry["custom_id"]
                            print(f" └─ handling {sid}")
                            
                            # Here we associate the current {scene_name}_floor_{floor_id} to the custom_id of the batched response
                            response_text = split_entry['response']['body']['choices'][0]['message']['content']
                            # Convert this response_text str to a dictionary
                            try:
                                response = json.loads(response_text)
                                # append response for statistics
                                response_file.append({
                                    "scene_name": scene_name,
                                    "floor_id": floor,
                                    "custom_id": sid,
                                    "response": response
                                })                                
                            except:
                                continue
                            
                            # append response
                            response_file.append({
                                "scene_name": scene_name,
                                "floor_id": floor,
                                "custom_id": sid,
                                "response": response
                            })

                            # We extract the summary and data from the response and save it to a new dictionary
                            try:
                                process_response = preprocess_response(
                                    response=response, 
                                    object_json=items
                                )
                            except:
                                raise ValueError(
                                    f"Error processing response for scene {scene_name} and floor {floor} and custom_id {sid}. "
                                )
                            
                            # We build the episode list
                            single_floor_episodes = generate_episodes_from_batch(
                                split=args.split,
                                scene_id=scene_name,
                                object_var=process_response,
                                feature_map=None
                            )
                            for ep in single_floor_episodes:
                                print(f"Owner: {ep['owner']}, Object Category: {ep['object_category']}, Object ID: {ep['object_id']}, Position: {ep['object_pos']}")
                            
                            if len(single_floor_episodes) <= 6:
                                print(f"Floor {floor} has too few objects: {len(single_floor_episodes)}")
                                
                            if args.generate_active_data and args.add_nav_data:
                                single_floor_episodes = prepare_episode_data(
                                    sim=sim,
                                    episodes=single_floor_episodes,
                                    level=args.data_split,
                                    use_view_points=True,
                                )
                                
                            # Check to allow multiple instances of the same object owned by same person 
                            if args.generate_active_data:                               
                                single_floor_episodes = allow_multiple_instances(single_floor_episodes)
                            
                            # Overwrite placeholders with random names
                            single_floor_episodes = overwrite_placeholders_names(single_floor_episodes)
                            
                            multiple_floor_episodes.extend(single_floor_episodes)
                            all_episodes.extend(single_floor_episodes)
                            n_episodes += len(single_floor_episodes)
                            print(f"Number of episodes for floor {floor}: {len(single_floor_episodes)}") 
            
            # Add episode_id to each episode in multiple_floor_episodes
            if not args.generate_active_data:
                for i, episode in enumerate(multiple_floor_episodes):
                    episode['episode_id'] = i
            else:
                for i, episode in enumerate(all_episodes):
                    episode['episode_id'] = i
                
            # Convert all_episodes to a list of dictionaries with episode_id
            if args.generate_active_data:
                active_nav_episodes.append(generate_objectgoal_json(multiple_floor_episodes))
                all_active_nav_episodes.extend(active_nav_episodes)
                
            # Save the episodes to a JSON file
            #if args.save_files and not args.generate_active_data:
            #    write_retrieval_episodes(multiple_floor_episodes, scene_name, args.split, args.data_split)
                
            # Save active data
            #if args.save_files and args.generate_active_data:
            #    write_active_episodes(active_nav_episodes, scene_name, args.split, args.data_split)
        
    # Randomly Scrape out to match Max Episodes
    if len(all_episodes) > MAX_EPISODES[args.data_split]:
        if not args.generate_active_data:
            all_episodes = random.sample(
                all_episodes, MAX_EPISODES[args.data_split]
            )
        else:
            episodes_id = random.sample(
                all_episodes, MAX_EPISODES[args.data_split]
            )
            episodes_id = [ep.get("episode_id") for ep in episodes_id]
            filtered_nav_episodes = []
            for k, scene in enumerate(all_active_nav_episodes):
                # Filter episodes in the scene
                filtered_episodes = [
                    ep for ep in scene["episodes"]
                    if int(ep.get("episode_id")) in episodes_id
                ]
                if len(filtered_episodes) == 0:
                    continue
                # Store the filtered episodes for the scene
                filtered_nav_episodes.append({"goals_by_category": scene["goals_by_category"],
                                            "episodes": filtered_episodes})
        # Reindex each episode_id
        for i, episode in enumerate(all_episodes):
            episode['episode_id'] = i
            
    # Save grouped episodes to JSON files
    if args.save_files and args.generate_active_data:
        groupby_write_active_episode(
            episodes=filtered_nav_episodes,
            split=args.split,
            level=args.data_split)
        
    # Save Query-Retrieval Data
    if args.save_files and not args.generate_active_data:
        groupby_write_passive_episode(
            episodes=all_episodes,
            split=args.split,
            level=args.data_split)
        
    # Save response file
    with open(os.path.join(args.base_json_path, f"responses/responses_{args.data_split}.json"), 'w') as f:
        json.dump(response_file, f, indent=2)
    
    # Write the object_category to ID mapping for active navigation
    if args.generate_active_data and args.save_files:
        cat_to_id_mapping(all_episodes, level=args.data_split)
        
    print("Episodes have been written to JSON files.")
    print("Number of episodes:", n_episodes)
    # Print statistics
    print("Mean Geodesic Distance:", sum(ep.get("geodesic_distance", 0) for ep in all_episodes) / len(all_episodes))
    print("Mean Euclidean Distance:", sum(ep.get("euclidean_distance", 0) for ep in all_episodes) / len(all_episodes))
    
    # Calculate average description lenght per word
    avg_description_length = sum(
        len(ep["summary"].split(' ')) for ep in all_episodes
    ) / len(all_episodes) if all_episodes else 0
    print("Average Description Length:", avg_description_length)
    
    

def overwrite_placeholders_names(episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Replaces placeholders like <person1> with random names in episode data.

    This function iterates through a list of episodes. For each episode, it finds
    all unique person placeholders, assigns a unique random name to each, and
    then applies these replacements consistently across the 'summary',
    'extracted_summary', 'query', and 'owner' fields.

    Args:
        episodes: A list of episode dictionaries.

    Returns:
        The list of episodes with placeholders replaced by names.
    """
    # Regex to find all placeholders like <person1>, <person2>, etc.
    placeholder_regex = re.compile(r"<person\d+>")

    for ep in episodes:
        # --- 1. Find all unique placeholders in the current episode ---
        unique_placeholders: Set[str] = set()

        # Search in summary
        unique_placeholders.update(placeholder_regex.findall(ep["summary"]))

        # Search in the list of strings for extracted_summary and query
        for item in ep["extracted_summary"]:
            unique_placeholders.update(placeholder_regex.findall(item))
        for item in ep["query"]:
            unique_placeholders.update(placeholder_regex.findall(item))

        # Ensure the owner placeholder is included
        if ep["owner"] in placeholder_regex.findall(ep["owner"]):
            unique_placeholders.add(ep["owner"])

        # --- 2. Create a mapping from placeholder to a random name ---
        if not unique_placeholders:
            continue # Skip if no placeholders are found

        num_names_needed = len(unique_placeholders)
        if num_names_needed > len(NAMES):
            raise ValueError(
                f"Not enough unique names in NAMES set to replace all "
                f"{num_names_needed} placeholders in episode."
            )

        random_names = random.sample(list(NAMES), num_names_needed)
        name_map = dict(zip(unique_placeholders, random_names))

        # --- 3. Replace placeholders using the created map ---
        
        # Helper function for replacement using re.sub
        def replace_func(match):
            return name_map.get(match.group(0), match.group(0))

        # Replace in 'summary'
        ep["summary"] = placeholder_regex.sub(replace_func, ep["summary"])

        # Replace in 'extracted_summary'
        ep["extracted_summary"] = [
            placeholder_regex.sub(replace_func, item) for item in ep["extracted_summary"]
        ]

        # Replace in 'query'
        ep["query"] = [
            placeholder_regex.sub(replace_func, item) for item in ep["query"]
        ]

        # Replace in 'owner'
        ep["owner"] = name_map.get(ep["owner"], ep["owner"])

    return episodes

def cat_to_id_mapping(episodes: List[Dict[str, Any]],
                      level="easy",
                      base_dir= "/Users/filippoziliotto/Desktop/Repos/habitat-lab-v0/data/datasets/eai_pers/active") -> Dict[str, int]:
    """
    Generates a mapping from object categories to unique IDs based on the episodes.

    Args:
        episodes (list): List of episode dictionaries, each containing 'object_category'.
    """
    object_categories = set()
    for episode in episodes:
        object_categories.add(episode["object_category"])
    object_categories = {cat: idx for idx, cat in enumerate(sorted(object_categories))}
    output_dir = os.path.join(base_dir, "val", level)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{level}.json.gz")
    # Convert to {object_cat: ID} mapping
    file = {}
    file["episodes"] = []
    file["category_to_task_category_id"] = object_categories
    file["category_to_scene_annotation_category_id"] = object_categories

    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        json.dump(file, f, indent=2)
        
    # Now Loop over the json.gz in content fodler and add the two cateory mappings to .json.gz
    for file_name in os.listdir(os.path.join(output_dir, "content")):
        if file_name.endswith(".json.gz"):
            with gzip.open(os.path.join(output_dir, "content", file_name), "rt", encoding="utf-8") as f:
                content = json.load(f)
            # Add the category mappings
            content["category_to_task_category_id"] = object_categories
            content["category_to_scene_annotation_category_id"] = object_categories
            # Write back to the same file
            with gzip.open(os.path.join(output_dir, "content", file_name), "wt", encoding="utf-8") as f:
                json.dump(content, f, indent=2)
    print(f"Object categories to ID mapping saved to {output_path}")
        
    return
        
def allow_multiple_instances(episodes_lst: list):
    """
    Merges multiple instances of the same object for a person into a single episode,
    allowing for multiple object instances in the resulting episode.

    Args:
        episodes (list): List of episode dictionaries, each containing 'owner', 'object_category', 'geodesic_distance', etc.
    Returns:
        list: Flattened list of episodes with multiple instances merged per (owner, object_category).
    """
    assert "geodesic_distance" in episodes_lst[0], "Episodes must contain 'geodesic_distance' field"
    
    # Groupby person and object_category and by summary
    grouped_episodes = {}
    for episode in episodes_lst:
        key = (episode['owner'], episode['object_category'], episode['summary'])
        if key not in grouped_episodes:
            grouped_episodes[key] = []
        grouped_episodes[key].append(episode)
    
    # Merge episodes with the same owner and object_category and summary
    merged_episodes = []
    for (owner, object_category, _), episodes in grouped_episodes.items():
        if len(episodes) > 1:
            
            # Find the episode with the smallest geodesic distance
            closest_idx = min(
                range(len(episodes)),
                key=lambda i: episodes[i]['geodesic_distance']
            )
            # Sort episodes by geodesic distance (ascending)
            sorted_episodes = sorted(episodes, key=lambda ep: ep['geodesic_distance'])

            # Merge relevant fields from all episodes
            merged_viewpoints = [ep["view_points"] for ep in sorted_episodes]
            merged_positions = [ep["object_pos"] for ep in sorted_episodes]
            merged_descriptions = [ep["description"] for ep in sorted_episodes]
            merged_object_ids = [ep["object_id"] for ep in sorted_episodes]

            # Generate a query that allows for multiple instances
            merged_query = generate_queries(
                object_name=object_category,
                owner=owner,
                augment=True,
                multi_instance=True
            )

            # Update the closest episode with merged data
            merged_episode = episodes[closest_idx].copy()
            merged_episode["view_points"] = merged_viewpoints
            merged_episode["object_pos"] = merged_positions
            merged_episode["description"] = merged_descriptions
            merged_episode["object_id"] = merged_object_ids
            merged_episode["query"] = merged_query

            merged_episodes.append(merged_episode)
        else:
            # Only one episode, no merging needed
            merged_episodes.extend(episodes)

    return merged_episodes

def process_batch_api(file_name='batch_api_output.jsonl'):
    
    if file_name.endswith('.jsonl'):
        # Read the JSONL file
        with open(file_name, 'r') as f:
            batch_api_output = f.readlines()
        
        # Read each line as a dict and extract the list of dicts
        batch_api_output = [json.loads(item) for item in batch_api_output if item]
    
    else:
        raise ValueError("Unsupported file format. Please provide a .jsonl file.")
    
    # Return batch_api_output
    return batch_api_output

def preprocess_response(response, object_json):
    
    # Create a dictionary for quick lookup based on object_id for later
    object_lookup = {
        obj["object_id"]: {
            "object_category": obj["object_category"],
            "position": obj.get("position", []),
            "floor_id": obj.get("floor_id", None),
            "description": obj.get("description", [])
        }
        for obj in object_json
    }
    # Check for missing or empty values
    assert all(
        obj["description"] != [] and obj["position"] != [] and obj["floor_id"] is not None
        for obj in object_json
    ), "Description, position or floor_id is empty"

    responses = []
    for scene_summary in response["summaries"]:
        # Create a dictionary to store the response
        response_dict = {}
        
        # Extract selected object items
        response_dict["selected_items"] = scene_summary["selected_items"] # This is a list of dicts
        
        # Extract the summary and data from the response
        response_dict['summary'] = scene_summary['summary']
        
        # Convert the extracted summary to a list of sentences
        response_dict['extracted_summary'] = [sentence for sentence in scene_summary["extracted_summary"]]
    
        # Update selected_items to include extra details
        for item in scene_summary["selected_items"]:
            object_id = item["object_id"]
            obj_data = object_lookup.get(object_id, {})
            
            item.update({
                "object_id": object_id,
                "owner": item["owner"],
                "object_category": obj_data.get("object_category"),
                "position": obj_data.get("position"),
                "floor_id": obj_data.get("floor_id"),
                "description": obj_data.get("description"),
                "room_id": obj_data.get("room_id", None),
            })
    
        # Generate all the possible queries for the selected objects
        response_dict['queries'] = [
            generate_queries(object_lookup[item['object_id']]["object_category"], item['owner'], augment=True)
            for item in scene_summary["selected_items"]
        ]  
    
        responses.append(response_dict)
        
    return responses    

def generate_queries(object_name: str, owner: str, augment=True, multi_instance=False) -> str:
    """
    Generates diverse queries for finding an owner's object.
    
    Args:
        object_name (str): The name of the object.
        owner (str): The name of the owner.
        
    Returns:
        str: A query string.
    """
    
    # Allow for multiple instance of the same object
    if multi_instance:
        if augment:
            query_templates = [
                f"Find one of {owner}'s {object_name}s",
                f"Where is one of {owner}'s {object_name}s?",
                f"Locate one of {owner}'s {object_name}s",
                f"Retrieve one of {owner}'s {object_name}s position",
                f"Identify the position of one of {owner}'s {object_name}s",
                f"Give me the coordinates of one of {owner}'s {object_name}s",
            ]
            return [query for query in query_templates]
        else:
            return [f"Find one of {owner}'s {object_name}s"]

    if augment:
        query_templates = [
            f"Find {owner}'s {object_name}",
            f"Where is {owner}'s {object_name}?",
            f"Locate {owner}'s {object_name}",
            f"Retrieve {owner}'s {object_name} position",
            f"Identify the position of {owner}'s {object_name}",
            f"Give me the coordinates of {owner}'s {object_name}",
        ]
        return [query for query in query_templates]
    else:
        return [f"Find {owner}'s {object_name}"]

def generate_episodes_from_batch(split, scene_id, object_var, feature_map=None):
    """
    Create a single episode from the given object name, description, object position, and feature map.

    Args:
        split (str): The split of the episode (e.g., "val_seen", "val_unseen", "val_seen_syonyms").
        scene_id (str): The unique ID of the scene.
        object_var (dict): A dictionary containing the object category, object ID, person, and floor.
        feature_map (np.array): The feature map of the scene.

    Returns:
        dict: A dictionary containing the object name, description, object position, and feature map.
    """
    
    # OBJECT_VAR example input:
    """[
        {
            "selected_items": [
                {"object_id": "bed_001", "owner": "<person1>", "description": ["king-sized white bed","blablabla"], "position": [1, 2, 3], "floor_id": 1, "object_category": "bed"},
                {"object_id": "refridgerator_24", "owner": "<person2>", "description": ["black stainless-steel refrigerator", "blablabla"], "position": [4, 5, 6], "floor_id": 0, "object_category": "refrigerator"}
            ],
            "summary": "In the first-floor bedroom, there is a king-sized white bed near the window that belongs to <person1>, while in the kitchen, a black stainless-steel refrigerator owned by <person2>.",
            "extracted_summary": [
                "<person1> owns a king-sized white bed near the window in the bedroom",
                "<person2> owns a black stainless steel refrigerator in the kitchen"
            ],
            "queries": [
                "Find <person1>'s bed",
                "Locate <person2>'s refridgerator",
            ]
        },
        # Additional summaries processed similarly...
    ]"""

    # Get scene configurations
    scene_id = get_scene_path(split, scene_id)
    scene_dataset_config = "./data/scene_datasets/hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json"
    
    episodes = []
    # Loop over the different summaries and objects
    for i, response in enumerate(object_var):
        
        items = response["selected_items"]
        extracted_summaries = response["extracted_summary"]
        
        # Loop over the query for each summary
        for j, query in enumerate(response["queries"]):
    
            # Generate the episode dictionary
            episode = {
                
                # Episode information
                "episode_id": None, # Leave None for now
                "scene_id": scene_id, # str
                "scene_dataset_config": scene_dataset_config, # str
                
                # Object information
                "object_category": items[j]["object_category"], # str
                "object_id": items[j]["object_id"], # str
                "object_pos": items[j]["position"], # lst 
                "description": items[j]["description"], # lst of descriptions
                "owner": items[j]["owner"], # str
                "floor_id": items[j]["floor_id"], # str

                # Scene summary and mapping
                "summary": response["summary"], # str
                "extracted_summary": extracted_summaries, # lst of str
                "feature_map": feature_map, # For now, we will leave this as None
                
                # Query
                "query": query, # str
                
            }
            
            episodes.append(episode)

    return episodes

# TODO: fix this function
def check_longest_summary(episodes):
    """
    Check the longest summary in the episodes.
    
    Args:
        episodes (list): List of episode dictionaries.
        
    Returns:
        int: Length of the longest summary.
    """
    max_length = 0
    for episode in episodes:
        length = len(episode["extracted_summary"])
        if length > max_length:
            max_length = length
    return max_length

def extract_object_id_int(object_id_str: str) -> int:
    """Extracts the integer ID from a string like 'shelf_136'."""
    try:
        return int(object_id_str.split("_")[-1])
    except Exception:
        return -1  # fallback
    
def generate_objectgoal_json(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    goals_by_category = defaultdict(list)
    habitat_episodes = []

    for ep in episodes:
        scene_path = ep["scene_id"]
        scene_name = os.path.basename(scene_path)
        object_cat = ep["object_category"]
        object_ids = ep["object_id"]
        object_positions = ep["object_pos"]
        room_id = ep.get("room_id", None)
        view_points_list = ep.get("view_points", [])
        
        # Add other keys
        ep["additional_obj_config_paths"] = []
        ep["goals"] = []
        ep["start_room"] = None
        ep["shortest_paths"] = None

        # Normalize to lists if needed
        if not isinstance(object_ids, list):
            object_ids = [object_ids]
            object_positions = [object_positions]
            view_points_list = [view_points_list]

        goal_key = f"{scene_name}_{object_cat}"

        for i, (obj_id_str, obj_pos) in enumerate(zip(object_ids, object_positions)):
            obj_id_int = extract_object_id_int(obj_id_str)

            existing_ids = {g["object_id"] for g in goals_by_category[goal_key]}
            if obj_id_int not in existing_ids:
                goal_obj = {
                    "position": obj_pos,
                    "radius": None,
                    "object_id": obj_id_int,
                    "object_name": obj_id_str,
                    "object_name_id": None,
                    "object_category": object_cat,
                    "floor_id": ep.get("floor_id", None),
                    "room_id": ep.get("room_id", None),
                    "room_name": None,
                    "view_points": view_points_list[i] if i < len(view_points_list) else [],
                }
                goals_by_category[goal_key].append(goal_obj)

        # Use the first object_id as target
        closest_goal_object_id = extract_object_id_int(object_ids[0])
        ep_dict = dict(ep)
        ep_dict["closest_goal_object_id"] = closest_goal_object_id

        # Add or overwrite the 'info' key
        ep_dict["info"] = {
            "geodesic_distance": ep.get("geodesic_distance", -1),
            "euclidean_distance": ep.get("euclidean_distance", -1),
            "closest_goal_object_id": closest_goal_object_id
        }
        
        # Convert episode_id to str
        ep_dict["episode_id"] = str(ep_dict.get("episode_id", ""))
        
        # Delete unnecessary keys from episode
        del ep_dict["object_pos"]
        del ep_dict["view_points"]
        del ep_dict["geodesic_distance"]
        del ep_dict["euclidean_distance"]
        del ep_dict["feature_map"]
        del ep_dict["closest_goal_object_id"]

        habitat_episodes.append(ep_dict)

    return {
        "goals_by_category": dict(goals_by_category),
        "episodes": habitat_episodes,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch processing script")
    
    parser.add_argument("--base_path", type=str, default="data/datasets/eai_pers", help="Base path for the dataset")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to use")
    parser.add_argument("--data_split", type=str, default="hard", help="Dataset difficulty level")
    parser.add_argument("--save_files", type=bool, default=True, help="Whether to save the generated episodes")
    parser.add_argument("--base_json_path", type=str, default="personalized/io_files", help="Base path for the JSONL files")
    parser.add_argument("--batch_file_name", type=str, default="output_batch", help="Batch API JSONL file name")
    parser.add_argument("--add_nav_data", type=bool, default=True, help="Whether to add navigation data to the episodes")
    parser.add_argument("--use_graph_generator", type=bool, default=True, help="Whether to use episodes generated from graphs")
    parser.add_argument("--generate_active_data", type=bool, default=True, help="Whether to generate active learning data")
    
    args = parser.parse_args()
    main(args)
