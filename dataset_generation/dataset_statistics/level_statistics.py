#!/usr/bin/env python3
"""
level_statistics.py

Compute average per-scene statistics over a JSON file of scene responses:
- Number of owners owning multiple distinct objects
- Number of objects owned by multiple distinct owners
- Number of unique objects per scene
- Number of unique owners per scene
- Average number of selected_items per summary
- Category diversity (distinct object categories) per scene
- Average summary text length (characters) per summary
"""

import argparse
import json
import os
import statistics
from typing import List, Dict, Any, Set


def load_responses(path: str) -> List[Dict[str, Any]]:
    """Load the JSON file of scene responses."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_scene_metrics(scene: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute all desired metrics for a single scene.
    Returns a dict with:
      - multi_object_owners
      - multi_owned_objects
      - unique_objects
      - unique_owners
      - avg_selected_items_per_summary
      - category_diversity
      - avg_summary_length
    """
    summaries = scene.get('response', {}).get('summaries', [])
    owner_to_objects: Dict[str, Set[str]] = {}
    object_to_owners: Dict[str, Set[str]] = {}
    total_selected_items = 0
    total_summary_length = 0

    for summary in summaries:
        items = summary.get('selected_items', [])
        total_selected_items += len(items)
        text = summary.get('summary', '')
        total_summary_length += len(text)

        for item in items:
            obj_id = item['object_id']
            owner = item['owner']
            # map owner → objects
            owner_to_objects.setdefault(owner, set()).add(obj_id)
            # map object → owners
            object_to_owners.setdefault(obj_id, set()).add(owner)

    # Metric calculations
    num_summaries = len(summaries) or 1  # avoid division by zero
    multi_object_owners = sum(1 for objs in owner_to_objects.values() if len(objs) > 1)
    multi_owned_objects = sum(1 for owners in object_to_owners.values() if len(owners) > 1)
    unique_objects = len(object_to_owners)
    unique_owners = len(owner_to_objects)
    avg_selected_items_per_summary = total_selected_items / num_summaries
    avg_summary_length = total_summary_length / num_summaries

    # Category diversity: distinct categories across all selected_items
    categories = {
        item['object_category']
        for summary in summaries
        for item in summary.get('selected_items', [])
    }
    category_diversity = len(categories)

    return {
        'multi_object_owners': multi_object_owners,
        'multi_owned_objects': multi_owned_objects,
        'unique_objects': unique_objects,
        'unique_owners': unique_owners,
        'avg_selected_items_per_summary': avg_selected_items_per_summary,
        'category_diversity': category_diversity,
        'avg_summary_length': avg_summary_length,
    }


def aggregate_metrics(all_scene_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Compute the overall mean of each metric across all scenes."""
    aggregated: Dict[str, float] = {}
    if not all_scene_metrics:
        return aggregated

    # for each metric key, collect its values and compute the mean
    keys = all_scene_metrics[0].keys()
    for key in keys:
        values = [m[key] for m in all_scene_metrics]
        aggregated[key] = statistics.mean(values)

    return aggregated


def main():
    parser = argparse.ArgumentParser(
        description="Compute average scene-level statistics from responses_{level}.json"
    )
    parser.add_argument('--level', type=str,default="hard",help="Level identifier (e.g. '0', '1', ...) to load responses_{level}.json")
    args = parser.parse_args()

    filename = os.path.join( f'dataset_statistics/responses_{args.level}.json')
    if not os.path.isfile(filename):
        parser.error(f"File not found: {filename}")

    scenes = load_responses(filename)
    all_metrics = [compute_scene_metrics(scene) for scene in scenes]
    averages = aggregate_metrics(all_metrics)

    print(f"\nStatistics for file: {filename}\n" + "-" * (len(filename) + 24))
    print(f"• Avg. owners with >1 object per scene:        {averages.get('multi_object_owners', 0):.2f}")
    print(f"• Avg. objects with >1 owner per scene:        {averages.get('multi_owned_objects', 0):.2f}")
    print(f"• Avg. unique objects per scene:               {averages.get('unique_objects', 0):.2f}")
    print(f"• Avg. unique owners per scene:                {averages.get('unique_owners', 0):.2f}")
    print(f"• Avg. selected_items per summary:             {averages.get('avg_selected_items_per_summary', 0):.2f}")
    print(f"• Avg. category diversity per scene:           {averages.get('category_diversity', 0):.2f}")
    print(f"• Avg. summary length (chars) per summary:     {averages.get('avg_summary_length', 0):.2f}")
    print()


if __name__ == '__main__':
    main()
