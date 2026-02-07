import argparse
import os
import json
from pathlib import Path

import pandas as pd
from audiobox_aesthetics.infer import initialize_predictor
from tqdm import tqdm

predictor = initialize_predictor()

def categorize_and_score(folder, jsonl_path):
    group_scores = {
        "single": [],
        "multiple": [],
        "all": []
    }

    with open(jsonl_path, "r") as f:
        for line in tqdm(f, desc=os.path.basename(folder)):
            entry = json.loads(line)
            sample_id = entry["id"]
            degradations = entry.get("degradations", [])

            if not isinstance(degradations, list):
                print(f"⚠️ Skipping malformed entry: {sample_id}")
                continue

            group = "multiple" if len(degradations) > 1 else "single"

            file_path = os.path.join(folder, f"{sample_id}"+".flac")

            if not os.path.exists(file_path):
                print(f"⚠️ File not found: {file_path}")
                continue

            try:
                result = predictor.forward([{"path": file_path}])[0]
                group_scores[group].append(result)
                group_scores["all"].append(result)
            except Exception as e:
                print(f"⚠️ Error processing {file_path}: {e}")
                continue

    return {
        k: pd.DataFrame(v).mean().to_dict() if v else {} 
        for k, v in group_scores.items()
    }


def run_aesthetics(jsonl_path, folders):
    rows = []
    for folder in folders:
        grouped_results = categorize_and_score(folder, jsonl_path)
        for group_name, metrics in grouped_results.items():
            if not metrics:
                continue
            row = {"folder": os.path.basename(folder), "group": group_name}
            row.update(metrics)
            rows.append(row)
    return rows

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute aesthetic scores for inference folders")
    parser.add_argument("--jsonl-path", default="/testset_pt.jsonl")
    parser.add_argument("--output-csv", default="/evaluationfinal/aesthetic_summary_models.csv")
    parser.add_argument(
        "--folders",
        nargs="+",
        default=["outputs/run1", "outputs/run2"],
        help="Paths to inference output folders"
    )
    args = parser.parse_args()
    rows = run_aesthetics(args.jsonl_path, args.folders)
    if not rows:
        print("⚠️ No aesthetic results generated.")
    else:
        df = pd.DataFrame(rows)
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"\n✅ Saved grouped aesthetic scores to: {args.output_csv}")
        print(df)
