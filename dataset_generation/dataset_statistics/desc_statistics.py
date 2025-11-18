"""
This is the file where we caluclate the dataset statistics.
"""
import argparse
from dataset_statistics.utils import load_scenes_objects

def calculate_num_objects(objects_list):
    
    # Calculate the average number of objects
    total_objects = 0
    num_lists = 0
    for key, value in objects_list.items():
        num_objects = len(value)
        total_objects += num_objects
        num_lists += 1
    avg_objects = total_objects / num_lists if num_lists > 0 else 0

    # Calculate the standard deviation
    sum_sq_diff = 0
    for key, value in objects_list.items():
        num_objects = len(value)
        sum_sq_diff += (num_objects - avg_objects) ** 2
    std_objects = (sum_sq_diff / num_lists) ** 0.5 if num_lists > 0 else 0
    
    print(f"Average number of objects: {avg_objects}")
    print(f"Standard deviation of number of objects: {std_objects}")
    return avg_objects, std_objects
    
def calculate_num_scene(object_list, per_floor=True):
    """
    Calculate the number of scenes of HM3D.
    """
    if per_floor:
        floor_ids = set()
        for key, value in object_list.items():
            # Append (scene_name, floor_id) to the set
            for obj in value:
                if "floor_id" in obj:
                    floor_ids.add((key, int(obj["floor_id"])))
                else: continue
                
        print(f"Number of scenes, dividing per floor: {len(floor_ids)}")
        return len(floor_ids)
    
    print(f"Number of scenes: {len(object_list)}")
    return len(object_list)

def min_max_objects(object_list):
    """
    Calculate the min and max number of objects in the list.
    """
    min_objects = float('inf')
    max_objects = float('-inf')
    
    for key, value in object_list.items():
        num_objects = len(value)
        if num_objects < min_objects:
            min_objects = num_objects
        if num_objects > max_objects:
            max_objects = num_objects
            
    print(f"Minimum number of objects: {min_objects}")
    print(f"Maximum number of objects: {max_objects}")
    return min_objects, max_objects
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Calculate dataset statistics.")
    parser.add_argument("--base_path", type=str, default="data/datasets/goat_bench/hm3d/v2/", help="Base path to dataset")
    parser.add_argument("--split", type=str, choices=["total", "divide"], default="total", help="Split type to load")    
    args = parser.parse_args()
    
    print("=====================================")
    print("Starting to calculate dataset statistics...")
    print("=====================================")
    
    if args.split=="total":
        print("Loading the total seen + unseen...")
        # Load the objects from the dataset
        object_list = load_scenes_objects(base_dir=args.base_path, split="total")        
        # Calculate the number of objects
        avg_objects, std_objects = calculate_num_objects(object_list)
        # Calculate the number of scenes
        num_scenes = calculate_num_scene(object_list, per_floor=False)
        # Calculate the number of scenes per floor
        num_scenes_per_floor = calculate_num_scene(object_list, per_floor=True)
        # Calculate the min and max number of objects
        min_objects, max_objects = min_max_objects(object_list)
        
    elif args.split=="divide":
        print("Subdiving the dataset into seen and unseen...")
        val_seen_objects, val_unseen_objects = load_scenes_objects(split="divide")
    
        # Calculate the number of objects
        print("---------------")
        print("Val seen objects")
        print("---------------")
        avg_objects_seen, std_objects_seen = calculate_num_objects(val_seen_objects)
        # Calculate the number of scenes per floor
        num_scenes_per_floor_seen = calculate_num_scene(val_seen_objects)
        # Calculate the min and max number of objects
        min_objects_seen, max_objects_seen = min_max_objects(val_seen_objects)
        
        print("---------------")
        print("Val unseen objects")
        print("---------------")
        # Calculate the number of objects
        avg_objects_unseen, std_objects_unseen = calculate_num_objects(val_unseen_objects)
        # Calculate the number of scenes per floor
        num_scenes_per_floor_unseen = calculate_num_scene(val_unseen_objects)
        # Calculate the min and max number of objects
        min_objects_unseen, max_objects_unseen = min_max_objects(val_unseen_objects)

    print("=========")
    print("Finished")
    print("=========")