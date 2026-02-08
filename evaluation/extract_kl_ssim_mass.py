import argparse
import os
import json
from pathlib import Path

import numpy as np
import librosa
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import pandas as pd

def get_log_mel(path, sr=44100, n_fft=2048, hop_length=512, n_mels=128):
    y, _ = librosa.load(path, sr=sr)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft,
        hop_length=hop_length, n_mels=n_mels
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel

def compute_kl(mel1, mel2):
    p = np.mean(mel1, axis=1)
    q = np.mean(mel2, axis=1)
    p = p / (np.sum(p) + 1e-10)
    q = q / (np.sum(q) + 1e-10)
    kl = np.sum(p * np.log((p + 1e-10) / (q + 1e-10)))
    return kl

def compute_ssim(mel1, mel2):
    min_shape = (min(mel1.shape[0], mel2.shape[0]), min(mel1.shape[1], mel2.shape[1]))
    mel1 = mel1[:min_shape[0], :min_shape[1]]
    mel2 = mel2[:min_shape[0], :min_shape[1]]
    mel1 = (mel1 - mel1.min()) / (mel1.max() - mel1.min() + 1e-8)
    mel2 = (mel2 - mel2.min()) / (mel2.max() - mel2.min() + 1e-8)
    return ssim(mel1, mel2, data_range=1.0)

def evaluate_folder(folder, jsonl_path, target_root, out_dir):
    kl_scores = []
    ssim_scores = []

    with open(jsonl_path, "r") as f:
        entries = [json.loads(line) for line in f]

    for i, entry in enumerate(tqdm(entries, desc=f"Folder: {folder}")):
        audio_id = entry.get("id")
        original_id = entry.get("original_id")
        degradations = entry.get("degradations", [])

        # if len(degradations) > 1:
        #     continue
        # if len(degradations) < 2:
        #     continue

        # Handle missing original_id by using audio_id directly (files have matching names in both folders)
        if original_id is None:
            original_id = audio_id

        path1 = os.path.join(folder, f"{audio_id}.flac")
        path2 = os.path.join(target_root, f"{original_id}.flac")

        path1_exists = os.path.exists(path1)
        path2_exists = os.path.exists(path2)
        
        if not (path1_exists and path2_exists):
            if i < 3:  # Only print first 3 entries for debugging
                print(f"\nDebug entry {i}: id={audio_id}, original_id={original_id}")
                print(f"  path1={path1} (exists={path1_exists})")
                print(f"  path2={path2} (exists={path2_exists})")
            continue

        try:
            mel1 = get_log_mel(path1)
            mel2 = get_log_mel(path2)

            kl = compute_kl(mel1, mel2)
            ssim_score = compute_ssim(mel1, mel2)

            kl_scores.append(kl)
            ssim_scores.append(ssim_score)

        except Exception as e:
            print(f"Error processing pair {path1}, {path2}: {e}")

    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(out_dir, f"{folder}_kl_all.npy"), np.array(kl_scores))
        np.save(os.path.join(out_dir, f"{folder}_ssim_all.npy"), np.array(ssim_scores))

    return np.mean(kl_scores), np.mean(ssim_scores)


def run_kl_ssim(jsonl_path, folders, target_root, output_dir=None, save_raw=True):
    summary = []
    for folder in folders:
        print(f"\n🚀 Processing: {folder}")
        kl, ssim_val = evaluate_folder(
            folder,
            jsonl_path,
            target_root,
            out_dir=output_dir if save_raw else None,
        )
        summary.append({
            "folder": os.path.basename(folder),
            "avg_kl": kl,
            "avg_ssim": ssim_val
        })

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(summary)
        df.to_csv(os.path.join(output_dir, "KL_SSIM_summary_all.csv"), index=False)
        print(f"\n✅ KL/SSIM summary saved to: {os.path.join(output_dir, 'KL_SSIM_summary_all.csv')}")
        print(df)

    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute KL/SSIM metrics for inference folders")
    parser.add_argument("--jsonl-path", default="/testset_pt.jsonl")
    parser.add_argument("--target-root", default="/dataset/targets")
    parser.add_argument(
        "--folders",
        nargs="+",
        default=["outputs/run1", "outputs/run2"],
        help="Paths to inference folders"
    )
    parser.add_argument("--output-dir", default="/evaluationfinal/KL_SSIM")
    parser.add_argument("--save-raw", action="store_true", help="Save raw KL/SSIM arrays")
    args = parser.parse_args()
    run_kl_ssim(args.jsonl_path, args.folders, args.target_root, args.output_dir, save_raw=args.save_raw)
