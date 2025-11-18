import os
import json
import math
import numpy as np
from collections import Counter
from wordcloud import STOPWORDS

def load_scenes_objects(base_dir='data/datasets/eai_pers', split="total"):
    if split == "total":
        # Combine all JSONs from both splits into a single dict
        splits = ['val_seen_merged', 'val_unseen']
        merged_jsons = {}
        for s in splits:
            split_dir = os.path.join(base_dir, s)
            for entry in os.listdir(split_dir):
                entry_path = os.path.join(split_dir, entry, f"{entry}.json")
                if not os.path.exists(entry_path):
                    continue
                with open(entry_path, 'r') as f:
                    scene_json = json.load(f)
                # scene_json is a list of objects for this entry
                if entry in merged_jsons:
                    merged_jsons[entry].extend(scene_json)
                else:
                    merged_jsons[entry] = list(scene_json)

        # Now group by scene and floor
        floor_grouped = {}
        for scene_name, objects in merged_jsons.items():
            for obj in objects:
                floor_id = obj.get('floor_id', 0)
                key = f"{scene_name}_floor_{floor_id}"
                if key in floor_grouped:
                    floor_grouped[key].append(obj)
                else:
                    floor_grouped[key] = [obj]
        return floor_grouped

    elif split == "divide":
        # Return separate lists grouped by floor for val_seen and val_unseen
        result = {}
        for s in ['val_seen', 'val_unseen']:
            split_dir = os.path.join(base_dir, s)
            grouped = {}
            for entry in os.listdir(split_dir):
                entry_path = os.path.join(split_dir, entry, f"{entry}.json")
                if not os.path.exists(entry_path):
                    continue
                with open(entry_path, 'r') as f:
                    scene_json = json.load(f)
                for obj in scene_json:
                    floor_id = obj.get('floor_id', 0)
                    key = f"{entry}_floor_{floor_id}"
                    if key in grouped:
                        grouped[key].append(obj)
                    else:
                        grouped[key] = [obj]
            result[s] = grouped
        return result['val_seen'], result['val_unseen']

def load_old_scene_objects(base_dir='data/datasets/goat_bench/hm3d/v2/', split="total"):
    
    if split == "total":
        # Combine all JSONs from both splits into a single dict
        splits = ['val_seen', 'val_unseen', 'val_seen_synonyms']
        merged_jsons = {}
        for s in splits:
            split_dir = os.path.join(base_dir, s)
            for entry in os.listdir(os.path.join(split_dir, "content")):
                entry_path = os.path.join(split_dir, "content", entry)
                if not os.path.exists(entry_path) or entry_path.endswith('.json'):
                    continue
                import gzip
                with gzip.open(entry_path, 'rt', encoding='utf-8') as f:
                    scene_json = json.load(f)
                # scene_json is a list of objects for this entry
                if entry in merged_jsons:
                    merged_jsons[entry].extend(scene_json.values())
                else:
                    merged_jsons[entry] = list(scene_json.values())
                    
    elif split == "divide":
        raise NotImplementedError("The 'divide' option is not implemented for the old dataset.")
    
    descriptions = []
    # This can return only list of descriptions
    for scene_name, objects in merged_jsons.items():
        for obj_infos in objects:
            for infos in obj_infos:
                for k,v in infos.items():
                    if 'lang_desc' in k:
                        # Only keep the first description
                        descriptions.append(infos['lang_desc'])
    
    return descriptions

def extract_descriptions(object_list):
    """
    Extracts the first description from each object in the object_list.

    Args:
        object_list (dict): Dictionary mapping scene names to lists of objects.

    Returns:
        list: List of description strings.
    """
    descriptions = []
    for scene_name, objects in object_list.items():
        for obj in objects:
            descriptions.append(obj.get('description', '')[0])
    return descriptions

def extract_objects(object_list):
    """
    Extracts the object names from the object_list.

    Args:
        object_list (dict): Dictionary mapping scene names to lists of objects.

    Returns:
        list: List of object names.
    """
    objects = []
    for scene_name, objects_list in object_list.items():
        for obj in objects_list:
            objects.append(obj.get('object_category', ''))
    return objects

def extract_object_room(object_list):
    """
    Extracts the object names and their corresponding room names from the object_list.

    Args:
        object_list (dict): Dictionary mapping scene names to lists of objects.

    Returns:
        list: List of tuples containing object names and their corresponding room names.
    """
    objects = []
    for scene_name, objects_list in object_list.items():
        for obj in objects_list:
            objects.append(obj.get('room', ''))
    # Empty room names are filtered out
    objects = [obj for obj in objects if obj != '']
    return objects

def compute_text_metrics(descriptions,
                         noise_tokens=None):
    noise_set = set(noise_tokens) if noise_tokens is not None else set()
    all_tokens = [token.lower() for desc in descriptions for token in desc.split()]
    total_tokens = len(all_tokens)
    vocab_size = len(set(all_tokens))
    lengths = [len(desc.split()) for desc in descriptions]
    mean_len = np.mean(lengths)
    std_len = np.std(lengths)
    freq = Counter(all_tokens)
    probs = [count / total_tokens for count in freq.values()] if total_tokens > 0 else []
    entropy = -sum(p * math.log2(p) for p in probs) if probs else 0.0
    noise_count = sum(count for token, count in freq.items() if token in noise_set)
    noise_fraction = (noise_count / total_tokens) * 100 if total_tokens > 0 else 0.0
    # Additional normalized metric: type-token ratio
    ttr = vocab_size / total_tokens if total_tokens > 0 else 0.0
    print(f"Number of Descriptions: {len(descriptions)}")
    print(f"Total Tokens: {total_tokens}")
    print(f"Vocabulary Size: {vocab_size}")
    print(f"Type-Token Ratio: {ttr:.3f}")
    print(f"Mean Tokens per Description: {mean_len:.2f} ± {std_len:.2f}")
    print(f"Shannon Entropy: {entropy:.2f} bits")
    print(f"Noise Fraction: {noise_fraction:.2f}%")
    return {
        'num_descriptions': len(descriptions),
        'total_tokens': total_tokens,
        'vocab_size': vocab_size,
        'type_token_ratio': ttr,
        'mean_tokens': mean_len,
        'std_tokens': std_len,
        'entropy': entropy,
        'noise_fraction': noise_fraction
    }
    
def compute_metrics_deltas(old_dict, new_dict):
    """
    Compute the difference in metrics between two dictionaries.
    """
    metrics = {}
    for key in old_dict.keys():
        if key in new_dict:
            metrics[key] = new_dict[key] - old_dict[key]
        else:
            metrics[key] = None
            
    # print deltas and clculate percentage
    for key, value in metrics.items():
        if value is not None:
            old_value = old_dict[key]
            new_value = new_dict[key]
            delta = new_value - old_value
            percentage_change = (delta / old_value) * 100 if old_value != 0 else float('inf')
            print(f"{key}: {delta} ({percentage_change:.2f}%)")
        else:
            print(f"{key}: Not Found in New Metrics")
            
    return metrics
    
if __name__ == '__main__':
    # Example usage
    base_dir = 'data/datasets/eai_pers'
    object_list = load_scenes_objects(base_dir, split="total")
    new_descriptions = extract_descriptions(object_list)
    old_descriptions = load_old_scene_objects(split="total")
    
    # strip empty strings
    new_descriptions = [desc for desc in new_descriptions if desc]
    old_descriptions = [desc for desc in old_descriptions if desc]
    
    # Compute metrics for new descriptions
    print("\nOld Descriptions Metrics:")
    old_metrics = compute_text_metrics(old_descriptions)
    print("New Descriptions Metrics:")
    new_metrics = compute_text_metrics(new_descriptions)

    # Compute deltas
    print("\nMetrics Deltas:")
    compute_metrics_deltas(old_metrics, new_metrics)