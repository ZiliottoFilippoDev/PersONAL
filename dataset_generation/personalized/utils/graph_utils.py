from typing import Dict, List
import numpy as np
import networkx as nx


def aggregate_metrics(
    ownership_list: List[Dict[str, List[str]]]
) -> Dict[str, float]:
    """
    Given a list of ownership dicts, compute the mean of each metric.
    """
    all_metrics = [graph_metrics(o) for o in ownership_list]

    # keys are the same for each; just average across runs
    agg = {}
    for key in all_metrics[0].keys():
        vals = [m[key] for m in all_metrics]
        agg[f"mean_{key}"] = float(np.mean(vals))
    return agg


def graph_metrics(ownership: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Costruisce il grafo bipartito e restituisce:
      - num_people, num_objects, num_edges
      - avg_objs_per_person
      - avg_people_per_object
      - density = E / (P*M)
      - overlap_ratio = fraction di oggetti con >1 proprietario
    """
    B = nx.Graph()
    people = list(ownership.keys())
    objects = sorted({o for objs in ownership.values() for o in objs})
    B.add_nodes_from(people, bipartite="person")
    B.add_nodes_from(objects, bipartite="object")
    for p, objs in ownership.items():
        for o in objs:
            B.add_edge(p, o)

    P, M, E = len(people), len(objects), B.number_of_edges()
    # degree arrays
    deg_p = np.array([B.degree(p) for p in people], dtype=float)
    # compute overlap ratio
    shared = sum(1 for o in objects if B.degree(o) > 1)
    overlap_ratio = shared / M if M > 0 else 0.0
    
    # Compute avg. degree per object
    deg_o = np.array([B.degree(o) for o in objects], dtype=float)
    
    return {
        "num_people": P,
        "num_objects": M,
        "num_edges": E,
        "avg_degree_per_person": deg_p.mean() if P else 0.0,
        "avg_degree_per_object": deg_o.mean() if M else 0.0,
        "num_shared_objects": shared,
        "density": E / (P * M) if P and M else 0.0,
        "overlap_ratio": overlap_ratio
    }


def convert_ownership_structure(ownership_dict: Dict) -> Dict:
    """
    Converts a dict of {owner: [object_id, ...]} to
    {"selected_items": [{"object_id": ..., "owner": ...}, ...]}
    """
    selected_items = []
    for owner, objects in ownership_dict.items():
        for obj in objects:
            selected_items.append({"object_id": obj, "owner": owner})
    return {"selected_items": selected_items}