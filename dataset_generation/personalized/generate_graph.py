import numpy as np
import random
from typing import List, Dict, Any, Optional
from personalized.utils.graph_utils import graph_metrics

import numpy as np
import random
import networkx as nx
from typing import List, Dict, Any, Optional


# ---------------------------------------------------
# Parametri globali per ciascun livello di difficoltà
# ---------------------------------------------------
DIFFICULTY_SETTINGS = {
    "easy": {
        # density = E / (P*M) → vogliamo praticamente 0
        "density_range": (0.0, 0.01),
        # avg degree per person ~ 1
        "avg_degree_range": (1.0, 1.1),
        # fraction of shared objects = 0
        "max_overlap_ratio": 0.0
    },
    "medium": {
        # density bassa
        "density_range": (0.02, 0.10),
        # avg degree per person tra 1 e 3
        "avg_degree_range": (1.0, 3.0),
        # 5% di oggetti possono essere condivisi
        "max_overlap_ratio": 0.05
    },
    "hard": {
        # density un po' più alta
        "density_range": (0.10, 0.30),
        # avg degree per person tra 3 e 6
        "avg_degree_range": (2.0, 4.0),
        # allow overlap fino al 50% degli oggetti
        "max_overlap_ratio": 0.3
    }
}


def gen_ownership(
    objects_list: List[Dict[str, Any]],
    mu: float,
    overlap: float,
    difficulty: str = "hard",
    degree_dist: str = "poisson",
    degree_variance: float = 0.0,
    max_tries: int = 10,
    **kwargs
) -> Dict[str, List[str]]:
    """
    Genera ownership secondo tre livelli di difficoltà 'easy', 'medium', 'hard',
    risamplando fino a soddisfare le metriche target, e supporta variabilità
    sul grado con `degree_variance`.

    Parameters aggiuntivi (pass-through): min_number_of_objects, min_number_of_people,
    max_number_of_people, max_objects_per_person.
    """
    # Estraggo configurazioni base e parametri
    min_objs = kwargs.get("min_number_of_objects", 3)
    if len(objects_list) < min_objs:
        raise ValueError(f"Serve almeno {min_objs} oggetti, trovati {len(objects_list)}")

    settings = DIFFICULTY_SETTINGS[difficulty]
    obj_ids = [o["object_id"] for o in objects_list]
    M = len(obj_ids)

    # Helper: genera un singolo ownership con i parametri dati
    def _sample_once() -> Dict[str, List[str]]:
        # 1) numero di persone
        P = random.randint(kwargs.get("min_number_of_people", 2),
                           kwargs.get("max_number_of_people", 8))
        cap = M if kwargs.get("max_objects_per_person") is None else kwargs["max_objects_per_person"]

        # 2) campionare i gradi con rumore gaussiano su mu
        lam = max(mu + degree_variance * np.random.randn(), 0.1)
        if degree_dist == "poisson":
            raw = np.random.poisson(lam=lam, size=P)
        else:
            p = min(lam / cap, 1.0)
            raw = np.random.binomial(n=cap, p=p, size=P)
        di = np.clip(raw, 1, cap).astype(int)

        # 3) costruzione del grafo “hard” (con overlap controllato) o “medium” (no sharing)
        ownership: Dict[str, List[str]] = {}
        if difficulty == "easy":
            P = min(P, M)
            for i in range(P):
                ownership[f"<person{i+1}>"] = [obj_ids[i]]
            return ownership

        if difficulty == "medium":
            # 1) Non posso avere più persone che oggetti
            P = min(P, M)

            # 2) Campiono i gradi
            if degree_dist == "poisson":
                raw = np.random.poisson(lam=lam, size=P)
            else:
                raw = np.random.binomial(n=cap, p=min(lam/cap,1.0), size=P)

            # 3) Consentiamo grado 0 (alcune persone senza oggetti)
            di = np.clip(raw, 0, cap).astype(int)

            # 4) Evito loop infinito: risamplo solo se sum(di) > M,
            #    non vincolo a di>=1!
            while di.sum() > M:
                if degree_dist == "poisson":
                    raw = np.random.poisson(lam=lam, size=P)
                else:
                    raw = np.random.binomial(n=cap, p=min(lam/cap,1.0), size=P)
                di = np.clip(raw, 0, cap).astype(int)

            # 5) Assegno gli oggetti unici
            shuffled = obj_ids.copy()
            random.shuffle(shuffled)
            idx = 0
            ownership = {}
            for i in range(P):
                k = di[i]
                ownership[f"<person{i+1}>"] = shuffled[idx: idx + k]
                idx += k
            return ownership
        
        # difficulty == "hard"
        alpha = (1 - overlap) / max(overlap, 1e-3)
        w = np.random.dirichlet([alpha] * M)
        for i in range(P):
            k = di[i]
            chosen = np.random.choice(M, size=k, replace=False, p=w)
            ownership[f"<person{i+1}>"] = [obj_ids[j] for j in chosen]
        return ownership

    # Risampling fino a soddisfare le metriche desiderate
    for _ in range(max_tries):
        own = _sample_once()
        m = graph_metrics(own)

        # Verifica intervento sulle proprietà
        ok_density = settings["density_range"][0] <= m["density"] <= settings["density_range"][1]
        ok_degree  = settings["avg_degree_range"][0] <= m["avg_degree_per_person"] <= settings["avg_degree_range"][1]
        ok_overlap = m["overlap_ratio"] <= settings["max_overlap_ratio"]

        if ok_density and ok_degree and ok_overlap:
            return own, m

    # Se non riusciamo entro max_tries, ritorniamo l'ultimo campione
    return own, m





# --------------- Example of usage ---------------

if __name__ == "__main__":
    # Suppose your JSON is a list of dicts like in your prompt:
    sample_json = [
        {"object_id": f"obj_{i:03d}"} for i in range(50)
    ]

    # Generate a batch of 10 random ownership graphs
    runs = []
    for _ in range(10):
        own = gen_ownership(
            sample_json,
            mu=8.0,
            overlap=0.2,
            min_number_of_objects=3,
            min_number_of_people=2,
            max_number_of_people=8
        )
        runs.append(own)

    # Compute metrics for each and aggregate
    for idx, own in enumerate(runs, 1):
        m = graph_metrics(own)
        print(f"Run {idx}: {m}")
