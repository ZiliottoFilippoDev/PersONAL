#!/usr/bin/env python3
"""
Create per‑level JSONL files suitable for OpenAI's batch API, one line per
unique episode summary.

Each line is the output of generate_single_batch(unique_id, model, combined_prompt)
where combined_prompt == PROMPT_LLM + raw summary.

Output files produced in summary_refinement/files/:
  ├── easy_summaries.jsonl
  ├── medium_summaries.jsonl
  └── hard_summaries.jsonl
"""

import gzip
import json
from pathlib import Path
from typing import Dict, List, Set

# ---------------------------------------------------------------------
# Parameters you may want to change
# ---------------------------------------------------------------------
MODEL    = "gpt-4.1-mini"            # target model
VAL_ROOT = Path("data/datasets/eai_pers/active/val")
OUT_DIR  = Path("summary_refinement/files")
LEVELS   = ("easy", "medium", "hard")
MAX_TOKS = 2000
# ---------------------------------------------------------------------


def generate_single_batch(unique_id: str, model: str, combined_prompt: str) -> Dict:
    """Return a single object compatible with OpenAI batch‑upload format."""
    return {
        "custom_id": unique_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": combined_prompt}
            ],
            "max_tokens": MAX_TOKS,
            "temperature": 0.7
        }
    }


def collect_unique_summaries(json_gz_path: Path) -> List[Dict[str, str]]:
    """Extract and return list of dicts with 'episode_id' and its 'summary' from one .json.gz file."""
    with gzip.open(json_gz_path, "rt", encoding="utf-8") as fh:
        data = json.load(fh)

    episodes = data.get("episodes", [])
    seen: Set[str] = set()
    results: List[Dict[str, str]] = []
    for ep in episodes:
        ep_id = ep.get("episode_id")
        summary = ep.get("summary")
        if ep_id and summary and ep_id not in seen:
            results.append({"episode_id": ep_id, "summary": summary})
            seen.add(ep_id)
    return results


PROMPT_LLM = """You are a helpful and intelligent assistant.\n
Your task is to rewrite the provided summary to make it sound more natural and fluent.
*Important*:
    - You may freely change the ordering of sentences, rooms, or phrases if it improves clarity and flow.
    - You must preserve the original names, objects, attributes and room names exactly as they appear. Do not modify or replace them.
    - Do not add any new information and don't change owner-object relationships.
    - If the summary already sounds natural, return it unchanged.
Just output the rewritten summary without any additional text or formatting.
INPUT:\n\n"""
    
# TODO: Change the way the summary is arranged --> change the oredering of the phrases keeping the same meaning


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for level in LEVELS:
        level_dir = VAL_ROOT / level / "content"
        if not level_dir.is_dir():
            print(f"[WARN] {level_dir} does not exist – skipping.")
            continue

        out_path = OUT_DIR / f"IN_{level}_summaries.jsonl"
        with out_path.open("w", encoding="utf-8") as out_f:

            # iterate directly over the .json.gz files in .../content
            for json_gz in sorted(level_dir.glob("*.json.gz")):
                name = json_gz.stem                 # file name without .json.gz
                episodes = collect_unique_summaries(json_gz)

                for i, ep in enumerate(episodes, start=1):
                    unique_id       = f"{name.split('.')[0]}_{ep['episode_id']}"
                    combined_prompt = f"{PROMPT_LLM}{ep['summary']}"
                    batch_obj       = generate_single_batch(
                        unique_id, MODEL, combined_prompt
                    )
                    out_f.write(json.dumps(batch_obj, ensure_ascii=False) + "\n")

        print(f"✔ Wrote {out_path.absolute()}")


if __name__ == "__main__":
    main()
