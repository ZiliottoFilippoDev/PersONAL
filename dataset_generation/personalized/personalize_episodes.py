# Define helper functions
import re
import random
import copy
from typing import Any, Dict, List, Set

OVERLAP = False

# Extract all <personX> placeholders from nested structures
def extract_placeholders(item: Any, found: Set[str]) -> None:
    if isinstance(item, str):
        found.update(re.findall(r"<person\d+>", item))
    elif isinstance(item, list):
        for sub in item:
            extract_placeholders(sub, found)
    elif isinstance(item, dict):
        for sub in item.values():
            extract_placeholders(sub, found)

# Recursively replace placeholders with their mapped names
def replace_in_item(item: Any, mapping: Dict[str, str]) -> Any:
    if isinstance(item, str):
        for ph, name in mapping.items():
            item = item.replace(ph, name)
        return item
    elif isinstance(item, list):
        return [replace_in_item(elem, mapping) for elem in item]
    elif isinstance(item, dict):
        return {k: replace_in_item(v, mapping) for k, v in item.items()}
    return item

# Replace placeholders using a given overlap probability
def replace_person_placeholders(
    episode: Dict[str, Any],
    name_list: List[str] = None,
    overlap_probability: float = 0.0
) -> Dict[str, Any]:
    if name_list is None:
        name_list = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Helen", "Ivy", "Jack"]

    new_episode = copy.deepcopy(episode)
    found_placeholders: Set[str] = set()
    extract_placeholders(new_episode, found_placeholders)

    person_numbers = []
    for ph in found_placeholders:
        m = re.search(r"<person(\d+)>", ph)
        if m:
            person_numbers.append(int(m.group(1)))
    person_numbers = sorted(set(person_numbers))

    mapping = {}
    available_names = name_list.copy()

    for num in person_numbers:
        key = f"<person{num}>"
        if num <= 3:
            if not available_names:
                raise ValueError("Not enough names available for unique mapping.")
            chosen = random.choice(available_names)
            mapping[key] = chosen
            available_names.remove(chosen)
        else:
            if random.random() < (1 - overlap_probability):
                if available_names:
                    chosen = random.choice(available_names)
                    mapping[key] = chosen
                    available_names.remove(chosen)
                else:
                    mapping[key] = random.choice(list(mapping.values()))
            else:
                mapping[key] = random.choice(list(mapping.values()))

    return replace_in_item(new_episode, mapping)

# Main function to apply replacement using dynamic overlap probability
def replace_with_dynamic_overlap(
    episode: Dict[str, Any],
    name_list: List[str] = None
) -> Dict[str, Any]:
    # Mapping from number of people -> overlap probability
    if OVERLAP:
        overlap_prob_by_people = {
            2: 0.0,
            3: 0.0,
            4: 0.01, # 5% chance of overlap
            5: 0.05, # 10% chance of overlap
            6: 0.1 # 20% chance of overlap
        }
    else:  
        overlap_prob_by_people = { k: 0.0 for k in range(2, 7) }

    # Count unique <personX> in the episode
    found_placeholders: Set[str] = set()
    extract_placeholders(episode, found_placeholders)

    person_count = len(set(
        int(re.search(r"<person(\d+)>", p).group(1))
        for p in found_placeholders if re.search(r"<person(\d+)>", p)
    ))

    # Lookup the overlap probability (default to 0.2 if more than 6 people)
    overlap_probability = overlap_prob_by_people.get(person_count, 0.2)

    # Apply the replacement
    return replace_person_placeholders(episode, name_list, overlap_probability)


if __name__ == "__main__":
    fake_episode = {
        "episode_id": None,
        "scene_id": "scene_01",
        "scene_dataset_config": "default",
        "object_category": "bed",
        "object_id": "bed_001",
        "object_pos": [1, 2, 3],
        "description": ["king-sized white bed", "blablabla"],
        "owner": "<person1>",
        "floor_id": 1,
        "summary": "In the bedroom, <person1> owns a bed. <person2> owns a fridge.",
        "extracted_summary": [
            "<person1> owns a king-sized white bed",
            "<person2> owns a black stainless steel refrigerator"
        ],
        "feature_map": None,
        "query": "Find <person1>'s bed"
    }

    new_episode = replace_with_dynamic_overlap(fake_episode)
    print(new_episode)
