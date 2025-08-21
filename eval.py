import os
import json
import csv
import torch
import numpy as np
import random
from tqdm import tqdm
from collections import Counter
from itertools import islice
from madmom.evaluation.key import key_label_to_class, error_type
from training_utils.dataloader import FMAKeyDataset

from skey.key_detection import (
    load_checkpoint,
    load_model_components,
    infer_key,
    DEFAULT_CHECKPOINT_PATH,
)

# === Set seed ===
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# === CONFIG ===
TEST_SPLIT_PATH = "/home/jovyan/skey/datasets/fma/data/split/test.txt"
# TEST_SPLIT_PATH = "/home/jovyan/skey/datasets/giantsteps/giantsteps.txt"
TEST_AUDIO_ROOT = "/home/jovyan/skey/datasets/fma/data/test"
# TEST_AUDIO_ROOT = "/home/jovyan/skey/datasets/giantsteps/mp3"
ckpt_dir = os.path.dirname(DEFAULT_CHECKPOINT_PATH)
PREDICTIONS_PATH = os.path.join(ckpt_dir, "predictions.json")
METRICS_CSV_PATH = "/home/jovyan/skey/eval/metrics.csv"
model_dir = os.path.dirname(DEFAULT_CHECKPOINT_PATH)
MODEL_NAME = os.path.basename(model_dir)
DATASET_NAME = "FMAKv2"
# DATASET_NAME = "GiantSteps"

# === Count total test tracks ===
with open(TEST_SPLIT_PATH, "r") as f:
    total_tracks = sum(1 for _ in f)

# === Load model ===
ckpt = load_checkpoint(DEFAULT_CHECKPOINT_PATH)
sr = ckpt["audio"]["sr"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hcqt, chromanet, crop_fn = load_model_components(ckpt, device)

# === Load dataset ===
test_dataset = FMAKeyDataset(
    split_file=TEST_SPLIT_PATH,
    root_dir=TEST_AUDIO_ROOT,
    sr=sr,
    # duration=ckpt["audio"]["dur"],
    duration=10,
    batch_size=1,
    device=device,
    dataset_type="supervised",
    shuffle=False
)

test_loader = iter(test_dataset)  # IterableDataset

# === Evaluation loop ===
predictions = []
counter = Counter()
mode_correct = 0

def get_mode(cls):
    return "minor" if cls >= 12 else "major"

for i, batch in enumerate(tqdm(islice(test_loader, total_tracks), total=total_tracks)):
    try:
        waveform = batch["audio"].squeeze(0).to(device)
        label = batch["keymode"][0][0]  # string like "D minor"

        if waveform.shape[0] == 2:  # stereo → mono
            waveform = waveform.mean(dim=0, keepdim=True)

        pred_key = infer_key(hcqt, chromanet, crop_fn, waveform, device)
        true_key = label.title().replace("Minor", "minor").replace("Major", "major")
        pred_key = pred_key.title().replace("Minor", "minor").replace("Major", "major")

        if pred_key == "Error":
            print(f"Invalid prediction at item {i}: {pred_key}")
            continue

        gt_class = key_label_to_class(true_key)
        pred_class = key_label_to_class(pred_key)
        score, category = error_type(pred_class, gt_class, strict_fifth=True)

        if get_mode(gt_class) == get_mode(pred_class):
            mode_correct += 1

        audio_path = batch["path"]
        file_name = os.path.basename(audio_path if isinstance(audio_path, str) else audio_path[0])

        predictions.append({
            "idx": i,
            "file": file_name,
            "true_key": true_key,
            "pred_key": pred_key,
            "category": category
        })
        counter[category] += 1

    except Exception as e:
        print(f"Error at item {i}: {e}")
        continue


# === Metrics ===
def compute_metrics(predictions, model_name="skey", dataset_name="FMAKv2-V2"):
    if not predictions:
        print("⚠️ No predictions to compute metrics.")
        return {
            'Model': model_name,
            'Dataset': dataset_name,
            'Weighted': 0.0,
            'Correct': 0.0,
            'Fifth': 0.0,
            'Relative': 0.0,
            'Parallel': 0.0,
            'Other': 0.0,
            'Mode Accuracy': 0.0
        }

    mode_correct = sum(
        1 for p in predictions
        if p["true_key"].split()[1] == p["pred_key"].split()[1]
    )

    weights = {'correct': 1.0, 'fifth': 0.5, 'relative': 0.3, 'parallel': 0.2, 'other': 0.0}
    counts = Counter(p['category'] for p in predictions)
    total = len(predictions)

    metrics = {
        'Model': model_name,
        'Dataset': dataset_name,
        'Weighted': round(sum(weights[k] * counts.get(k, 0) * 100 / total for k in weights), 1),
        **{k.capitalize(): round(counts.get(k, 0) * 100 / total, 1) for k in weights},
        'Mode Accuracy': round(mode_correct * 100 / total, 1)
    }
    return metrics

metrics = compute_metrics(predictions, MODEL_NAME, DATASET_NAME)

# === Save predictions ===
with open(PREDICTIONS_PATH, "w") as f:
    json.dump(predictions, f, indent=2)
print(f"Saved predictions to {PREDICTIONS_PATH}")

# === Append metrics ===
FIELDNAMES = [
    "Model", "Dataset", "Weighted",
    "Correct", "Fifth", "Relative", "Parallel", "Other", "Mode Accuracy"
]

file_exists = os.path.exists(METRICS_CSV_PATH)
with open(METRICS_CSV_PATH, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    if not file_exists:
        writer.writeheader()
    writer.writerow(metrics)
print(f"Appended metrics to {METRICS_CSV_PATH}")
