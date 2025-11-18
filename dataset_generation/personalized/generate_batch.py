import os
import json
import argparse
import random
from collections import Counter

from personalized.prompts.generate_prompt import generate_prompt, generate_prompt_from_graph

DEBUG = False

if DEBUG:
    print("DEBUG MODE: This script is running in debug mode.\n It will generate a fake batch and save it to a file.")
    
OBJECTS_LIMIT = {
    "chair": 3, # val seen
    "cabinet": 2, # val seen
    "picture": 2, # val unseen
    "kitchen_cabinet": 2, # val unseen
}

def main(args):
    
    """
    This loops over all the scenes of a certain split.
    Then append the LLM prompt + scene objects JSON to a batch list.
    Saves the batch list. In another file we should upload this batch list and retrieve the responses.
    For batch uploading see https://platform.openai.com/docs/guides/batch
    """
    
    batch_list = create_batched_json(
        split_path=os.path.join(args.base_path, args.split),
        model=args.model_type,
        level=args.level,
        max_folders=args.max_folders
    )
    
    if args.save_batch and not DEBUG:
        output_path = f"{os.path.join(args.output_path, args.output_file_name)}_{args.level}"
        with open(f"{output_path}.jsonl", 'w') as file:
            for item in batch_list:
                file.write(json.dumps(item) + "\n")
        print(f"Batch file saved to {output_path}.jsonl")

def create_batched_json(split_path, model, level, max_folders=None):
    """
    Creates a batched file for batch upload for a single scene.
    For batch uploading see https://platform.openai.com/docs/guides/batch
    
    Args:
        split_path (str): Path to the split folder containing the scenes
        model (str): Model identifier to specify which OpenAI model to use
        max_folders (int, optional): Max number of folder to loop through. For debugging purposes. Defaults to None.
    
    Returns:
        list: Lista delle richieste batch create
    """
    batch_api = []
    g_metrics = {}  # To store graph metrics for each scene
    
    # Loop through each scene in the split
    for folder_id, folder_scene in enumerate(os.listdir(split_path)):
        scene_name = folder_scene
            
        # Avoid .DS_Store files
        if scene_name.startswith("."):
            continue
                
        # In each Scene folder load all JSON files
        scene_path = os.path.join(split_path, scene_name)
        for json_file in os.listdir(scene_path):
            json_path = os.path.join(scene_path, json_file)
                        
            # If episodes already exist, skip them
            if json_path.endswith(".json") and "episodes" not in json_path:

                # Open JSON file
                with open(json_path, 'r') as file:
                    json_content = json.load(file)
                    
                # Clean json content
                json_content = clean_json(json_content)
                
                # Limit cabinet and picture object to two per json
                json_content = apply_quota(json_content, limits=OBJECTS_LIMIT)
                    
                # Group dicts by floor_id
                floor_groups = {}
                for item in json_content:
                    floor = item.get("floor_id")
                    if floor not in floor_groups:
                        floor_groups[floor] = []
                    floor_groups[floor].append(item)
                    
                # Remove keys not in ["object_category", "object_id", "floor_id", "description", "position", "room"]
                for floor, items in floor_groups.items():
                    print(len(items))
                    for item in items:
                        keys_to_remove = set(item.keys()) - {"object_category", "object_id", "floor_id", "description", "position", "room"}
                        for key in keys_to_remove:
                            del item[key]

                # Generate a unique ID for the batch
                unique_id_base = json_path.split("/")[-1].split(".")[0]
                for floor, items in floor_groups.items():
                    
                    # Variable chunk based on the number of items
                    chunk_size = get_chunk_size(num_objects=len(items), 
                                                level=args.level,
                                                use_graph_strategy=args.use_graph_strategy)
                    
                    # If no chunking desired, just wrap items in a single list
                    chunks = ([items] if chunk_size is None
                            else [items[i:i + chunk_size]
                                    for i in range(0, len(items), chunk_size)])
                    
                    # If one of the chunks has less than 4 objects, merge it to the previous chunk
                    if len(chunks) > 1 and len(chunks[-1]) < 5:
                        chunks[-2].extend(chunks[-1])
                        chunks = chunks[:-1]
                    
                    # Generate the LLM prompt for this floor
                    count = 0
                    for split_idx, chunk in enumerate(chunks):
                        n_summaries, min_objects, max_objects = check_num_summaries(
                            len(chunk), level=args.level, split=args.split
                        )
                        
                        # Generate the prompt where LLM decides ownership graph
                        if not args.use_graph_strategy:
                            prompt = generate_prompt(
                                chunk,
                                LEVEL=level,
                                N_SUMMARIES=n_summaries,
                                MIN_OBJECTS=min_objects,
                                MAX_OBJECTS=max_objects
                            )
                            # Append floor info to unique_id
                            unique_id = f"{unique_id_base}_floor_{floor}_split_{split_idx}"
                            batch_api.append(
                                generate_single_batch(unique_id, model, prompt)
                            )
                        
                        # Use a graph strategy for generating the prompt  
                        else:                           
                            prompt = []
                            for summary_idx in range(n_summaries):
                                g_infos = generate_prompt_from_graph(
                                    chunk,
                                    LEVEL=level,
                                )
                                unique_id = f"{unique_id_base}_floor_{floor}_split_{count}"
                                batch_api.append(
                                    generate_single_batch(unique_id, model, g_infos["prompt"])
                                )
                                count += 1
                                
                                # Append graph metrics
                                g_metrics[unique_id_base] = g_infos["metrics"]

                
        if max_folders and folder_id >= max_folders - 1:
            break
        
    # Print Graph metrics
    if args.use_graph_strategy:
        print(f"Level: {args.level} - Aggregated Graph Metrics:")
        aggregated_metrics = aggregate_graph_metrics(g_metrics)
        for key, value in aggregated_metrics.items():
            print(f"{key}: {value:.4f}")
                
    return batch_api

def generate_single_batch(unique_id, model, combined_prompt):
    """
    Create a single batch dictionary for OpenAI's batch API call.

    Args:
        unique_id (str): A unique identifier used as the custom ID for the API call.
        model (str): The model identifier to specify which OpenAI model to use.
        combined_prompt (str): The user-provided prompt combined with any other necessary context.

    Returns:
        dict: A dictionary representing a single API call configuration.
    """
    return {
        "custom_id": unique_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": combined_prompt}
            ],
            "max_tokens": 3000,
            "temperature": 0.7
        }
    }    

def check_num_summaries(num_objects, level="easy", split="val_seen"):
    
    if level in ["easy"]:
        assert split in ["val_seen"]
        return 3, 3, 4
    elif level in ["medium"]:
        assert split in ["val_seen_merged"]
        return 3, 4, 7
    elif level in ["hard"]:
        assert split in ["val"]
        return 3, 7, 10    
    else:
        raise ValueError(f"Invalid level: {level}. Choose from ['easy', 'medium', 'hard'].")
   
def clean_json(json_content):
    """
    Given json_content: a list of dicts each containing keys
    ["object_category", "object_id", "room", "floor_id", "to_discuss",
     "description", "position", ...],
    returns a new list of dicts where:
      - "to_discuss" is removed,
      - "description" only contains non-empty strings,
      - only the following keys are kept:
        ["object_category", "object_id", "room", "floor_id",
         "description", "position"].
    """
    cleaned = []
    for entry in json_content:
        # Filter description list to remove empty strings
        desc = entry.get("description", [])
        desc = [s for s in desc if isinstance(s, str) and s.strip()]

        # Build cleaned entry
        new_entry = {
            "object_category": entry.get("object_category"),
            "object_id":       entry.get("object_id"),
            "room":            entry.get("room"),
            "floor_id":        entry.get("floor_id"),
            "description":     desc,
            "position":        entry.get("position"),
        }
        cleaned.append(new_entry)

    # Random shuffle the cleaned entries
    random.shuffle(cleaned)

    return cleaned
    
def get_chunk_size(num_objects, level="easy", use_graph_strategy=False):
    """
    Determines the chunk size based on the number of objects, level, and split.
    
    Args:
        num_objects (int): The number of objects in the scene.
        level (str): The difficulty level of the dataset.
        split (str): The dataset split to use.
    Returns:
        int: The chunk size for the batch.
    """
    if use_graph_strategy and level in ["medium", "hard"]:
        # We take at most 10 objects for the graph strategy
        if num_objects > 10:
            return 10
        else:
            return num_objects
    
    if num_objects <= 10:
        return None
    elif num_objects <= 20:
        return num_objects // 2
    else:
        return num_objects // 3    
    
def aggregate_graph_metrics(metrics: dict):
    """
    Given a dictionary of graph metrics, compute the mean of each metric.
    
    Args:
        metrics (dict): Dictionary containing graph metrics.
        
    Returns:
        dict: Aggregated metrics with mean values.
    """
    aggregated = {}
    for key in next(iter(metrics.values())).keys():
        vals = [m[key] for m in metrics.values()]
        aggregated[f"mean_{key}"] = float(sum(vals) / len(vals)) if vals else 0.0
    return aggregated

def apply_quota(objs, limits=OBJECTS_LIMIT):
    """
    Keep at most `limits[cat]` items for every object_category.
    Items whose category is not in `limits` are kept in full.
    
    Parameters
    ----------
    objs : list[dict]
        Each dict **must** have an 'object_category' key.
    limits : dict[str, int]
        Upper bounds per category.

    Returns
    -------
    list[dict]
        The filtered list, in the same order as the input.
    """
    seen = Counter()
    filtered = []

    for obj in objs:                         # O(n) single pass
        cat = obj.get("object_category")
        # If the category is not limited OR still under the cap, keep it
        if cat not in limits or seen[cat] < limits[cat]:
            filtered.append(obj)
            seen[cat] += 1                   # update count

    return filtered

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch processing script")
    
    parser.add_argument("--save_batch", type=bool, default=True, help="Whether to save the batch output")
    parser.add_argument("--output_path", type=str, default="personalized/io_files", help="Path to save the output files")
    parser.add_argument("--output_file_name", type=str, default="input_batch", help="Name of the output file")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to use")
    parser.add_argument("--model_type", type=str, default="gpt-4.1", help="Model type")
    parser.add_argument("--level", type=str, default="hard", help="Difficulty level of the dataset")
    parser.add_argument("--max_folders", type=int, default=None, help="Maximum number of folders to process")
    parser.add_argument("--base_path", type=str, default="data/datasets/eai_pers", help="Base path for the dataset")
    parser.add_argument("--use_graph_strategy", type=bool, default=True, help="Use graph strategy for generating prompts")
    
    args = parser.parse_args()
    main(args)