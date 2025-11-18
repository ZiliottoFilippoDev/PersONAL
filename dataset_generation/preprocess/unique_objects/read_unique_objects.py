import os
import json
import glob

USE_TRAINING_DATA = False

def combine_json_files(directory):
    # Build file search pattern for all json files in the directory
    file_pattern = os.path.join(directory, "*.json")
    json_files = glob.glob(file_pattern)
    combined_data = []

    # Loop over each json file
    for file_path in json_files:
        
        # If total already saved skip it
        if "total.json" in file_path:
            continue
        
        if not USE_TRAINING_DATA:
            if "train" in file_path:
                continue 
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                # Check if the data is a list of dictionaries
                if isinstance(data, list):
                    combined_data.extend(data)
                else:
                    print(f"Skipping file {file_path}: Expected a list, got {type(data).__name__}.")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    return combined_data

def sum_counts(data):
    """
    Sums the 'count' values for each unique 'object_category'.

    Parameters:
        data (list): List of dictionaries with keys 'object_category' and 'count'.

    Returns:
        list: Aggregated list of dictionaries.
    """
    aggregated = {}
    for entry in data:
        category = entry.get("object_category")
        count = entry.get("count", 0)
        if category is not None:
            aggregated[category] = aggregated.get(category, 0) + count

    # Convert aggregated data back to list of dictionaries
    return [{"object_category": cat, "count": cnt} for cat, cnt in aggregated.items()]


def main():
    # Use the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Combine data from all json files
    unique_data = combine_json_files(current_dir)
    
    # Sum counts for each object_category
    aggregated_data = sum_counts(unique_data)
    
    # Sort the data by count in descending order
    aggregated_data = sorted(aggregated_data, key=lambda x: x["count"], reverse=True)
    
    # Save the combined data to a unique json file in the same folder
    output_file = os.path.join(current_dir, "total.json")
    try:
        with open(output_file, "w") as f:
            json.dump(aggregated_data, f, indent=4)
        print(f"Combined data successfully written to {output_file}")
    except Exception as e:
        print(f"Error writing combined file: {e}")

if __name__ == "__main__":
    main()