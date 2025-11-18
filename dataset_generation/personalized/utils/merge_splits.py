import os
import json
import argparse


def main(args):
    
    """
    File to merge togheter the splits. This is useful if we want to 
    merge val_unseen and val_seen_synonyms into a single val_unseen
    split.
    The script will merge the JSON files from the specified splits
    into a single output directory, ensuring that the merged files
    are saved in the correct subfolder structure.
    """
    
    base_dir = 'data/datasets/eai_pers'
    splits = ['val_seen', 'val_seen_synonyms']
    
    # Ensure output directory exists if saving is enabled
    output_base_dir = args.output
    if args.save:
        os.makedirs(output_base_dir, exist_ok=True)
    
    # Determine unique subfolders across splits
    subfolders = set()
    for split in splits:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            print(f"Directory {split_dir} does not exist. Skipping.")
            continue
        for entry in os.listdir(split_dir):
            entry_path = os.path.join(split_dir, entry)
            if os.path.isdir(entry_path):
                subfolders.add(entry)
    
    if not subfolders:
        print("No subfolders found in any splits. Exiting.")
        return
    
    unique_scenes = []
    
    # Process each subfolder
    for subfolder in subfolders:
        print(f"\nProcessing subfolder: {subfolder}")
        
        # Collect all JSON file names present in any split for this subfolder
        file_names = set()
        for split in splits:
            subfolder_dir = os.path.join(base_dir, split, subfolder)
            if os.path.exists(subfolder_dir):
                for fname in os.listdir(subfolder_dir):
                    if fname.endswith('.json'):
                        file_names.add(fname)
        
        if not file_names:
            print(f"No JSON files found in subfolder {subfolder}.")
            continue
        
        # Process each JSON file in this subfolder
        for file_name in file_names:
            merged_data = []
            print(f"  Merging file: {file_name}")
            
            # Loop over each split and merge available JSON file
            for split in splits:
                file_path = os.path.join(base_dir, split, subfolder, file_name)
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                merged_data.extend(data)
                            else:
                                print(f"    Warning: {file_path} does not contain a list; skipping.")
                    except Exception as e:
                        print(f"    Error reading {file_path}: {e}")
                        
            # Append file name
            unique_scenes.append(file_name)
            
            # Save or print merged data
            if args.save:
                output_dir = os.path.join(output_base_dir, subfolder)
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, file_name)
                try:
                    with open(output_path, 'w') as out_f:
                        json.dump(merged_data, out_f, indent=4)
                    print(f"    Saved merged file to: {output_path}")
                except Exception as e:
                    print(f"    Error saving {output_path}: {e}")
            else:
                print(f"    Merged content for {subfolder}/{file_name}:")
                # print(json.dumps(merged_data, indent=4))
                
    # making a set of unique scenes
    unique_scenes = sorted(set(unique_scenes))
    
    # save unique scenes in file in preprocess/unique_objects/files/unique_scenes.txt
    unique_scenes_path = os.path.join('preprocess', 'unique_objects', 'files', 'unique_scenes.txt')
    os.makedirs(os.path.dirname(unique_scenes_path), exist_ok=True)
    
    with open(unique_scenes_path, 'w') as f:
        for scene in unique_scenes:
            f.write(scene + '\n')
    
    print("Merging completed.")
                
def merge_json_lists(json_lists):
    """Concatenates multiple lists into one."""
    merged = []
    for lst in json_lists:
        merged.extend(lst)
    return merged
                    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge JSON files from multiple splits in parallel folders.")
    parser.add_argument('--save', action='store_true', default=False,
                        help="If provided, save merged JSON to the specified output directory.")
    parser.add_argument('--output', type=str, default='data/datasets/eai_pers/val_merged',
                        help="Output directory for merged JSON files (default: data/datasets/eai_pers/val_merged).")
    args = parser.parse_args()
    main(args)
