
import json
import os
import argparse

from apis.groq_api import generate_image_description
from groq import Groq

DEBUG = False

def process_floor_json(client, json_file_path, subfolder_path, model_name, img_index):
    """
    Processes a single JSON file: updates objects with an empty description slot using the model-generated description.
    
    Parameters:
      - json_file_path (str): Path to the JSON file.
      - subfolder_path (str): Path to the subfolder (needed to locate the images folder).
      - model_name (str): The model used for generating descriptions.
      - img_index (int): Which image index (0, 1, 2, or 3) to use for description.
    """
    with open(json_file_path, "r") as f:
        data = json.load(f)
    
    updated = False
    for obj in data:
        # Ensure the object has a "description" key that is a list
        if "description" not in obj or not isinstance(obj["description"], list):
            continue
        
        # Find the first empty description in the list (an empty string or whitespace)
        try:
            empty_idx = next(i for i, desc in enumerate(obj["description"]) if not desc.strip())
        except StopIteration:
            # All description slots are filled; skip this object
            continue
        
        object_category = obj.get("object_category", "object")
        object_id = obj.get("object_id")
        if not object_id:
            continue
        
        # Construct the image file name (e.g., "bench_1_0.png" if img_index==0)
        image_file_name = f"{object_id}_{img_index}.png"
        image_path = os.path.join(subfolder_path, "images", image_file_name)
        
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            continue
        
        # Generate a short description using the model
        description = generate_image_description(client, image_path, model_name, object_category)
        
        # Update the first empty description slot
        obj["description"][empty_idx] = description
        updated = True
        print(f"Updated {json_file_path}: set description for {object_id} at index {empty_idx}")
        
        if DEBUG:
            break
    
    # Save the JSON file if updates were made
    if updated:
        with open(json_file_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved updated JSON file: {json_file_path}")

def process_subfolder(client, subfolder_path, model_name, img_index):
    """
    Processes a given subfolder by looping over all JSON files related to floors 
    (i.e. those with 'floor' in the filename) and updating object descriptions.
    
    Parameters:
      - subfolder_path (str): Path to the subfolder.
      - model_name (str): The model used for generating descriptions.
      - img_index (int): Which image index (0, 1, 2, or 3) to use for description.
    """
    # Look for JSON files that include "floor" in their filename
    json_files = [f for f in os.listdir(subfolder_path) if f.endswith(".json") and "floor" in f]
    if not json_files:
        print(f"No floor JSON files found in {subfolder_path}")
        return
    
    for json_file in json_files:
        json_file_path = os.path.join(subfolder_path, json_file)
        process_floor_json(client, json_file_path, subfolder_path, model_name, img_index)

def main(model_name, img_index, split):
    base_path = "data/datasets/eai_pers"
    base_path = os.path.join(base_path, split)
    
    # Get the GROQ client
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )
    
    # Loop over every subfolder in the base path
    for entry in os.listdir(base_path):
        subfolder_path = os.path.join(base_path, entry)
        if os.path.isdir(subfolder_path):
            print(f"Processing subfolder: {subfolder_path}")
            process_subfolder(client, subfolder_path, model_name, img_index)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Groq-API script.')
    parser.add_argument('--model_name', type=str, default="llama-3.2-90b", help='Name of the model to use')
    parser.add_argument('--img_index', type=int, default=0, help='Index of the image to use for description')
    parser.add_argument('--split', type=str, default="val_unseen",  help='Dataset split to use (e.g., val_unseen)')

    args = parser.parse_args()
    
    if args.model_name in ['llama-3.2-90b']:
        args.model_name = "llama-3.2-90b-vision-preview"
    elif args.model_name in ['llama-3.2-11b']:
        args.model_name = "llama-3.2-11b-vision-preview"

    main(args.model_name, args.img_index, args.split)