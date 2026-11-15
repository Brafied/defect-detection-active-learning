"""Run one ResNet-50 experiment. Launch from the full-dataset notebook."""
import argparse
import csv
import math
import shutil
from torch.utils.data import Subset
from sklearn.metrics import confusion_matrix
import subprocess
import tarfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from IPython.display import display
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import ResNet50_Weights, resnet50
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score, ConfusionMatrixDisplay,
    PrecisionRecallDisplay, RocCurveDisplay,
)

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, required=True)
parser.add_argument("--split", type=int, required=True)
parser.add_argument("--rate", type=float, required=True)
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--data", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
SEED = 42
SPLIT_SEEDS = [args.split]
BACKBONE_RATES = [args.rate]
EPOCHS = args.epochs
HEAD_RATE = 1e-3
BATCH_SIZE = 16
VALIDATION_FRACTION = 0.2
TARGET_RECALL = 0.9
DATA_DIR = Path(args.data)
OUTPUT_DIR = Path(args.output)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)
device = torch.device(f"cuda:{args.gpu}")
torch.cuda.set_device(device)
with (DATA_DIR / "2cls_highshot.csv").open(newline="") as file:
    rows = list(csv.DictReader(file))
print(f"GPU {args.gpu}: split={args.split}, learning rate={args.rate}", flush=True)
weights = ResNet50_Weights.DEFAULT
preprocess = weights.transforms()

class VisaDataset(Dataset):
    def __init__(self, split):
        self.examples = [row for row in rows if row["split"] == split]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        row = self.examples[index]
        with Image.open(DATA_DIR / row["image"]) as image:
            pixels = preprocess(image.convert("RGB"))
        label = torch.tensor({"normal": 0, "anomaly": 1}[row["label"]], dtype=torch.float32)
        return pixels, label

train_data = VisaDataset("train")
test_data = VisaDataset("test")
assert len(train_data) and len(test_data)
assert {r["image"] for r in train_data.examples}.isdisjoint(r["image"] for r in test_data.examples)

def build_model():
    torch.manual_seed(SEED)
    model = resnet50(weights=weights)
    model.requires_grad_(False)
    model.layer4.requires_grad_(True)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model.to(device)

@torch.no_grad()
def predict(model, loader):
    model.eval()
    scores, targets = [], []
    for images, labels in loader:
        logits = model(images.to(device)).squeeze(1)
        scores.append(logits.sigmoid().cpu())
        targets.append(labels)
    return torch.cat(scores).numpy(), torch.cat(targets).numpy().astype(int)

def choose_threshold(labels, scores):
    defect_scores = sorted(scores[labels == 1], reverse=True)
    required = math.ceil(TARGET_RECALL * len(defect_scores))
    return float(defect_scores[required - 1])
strata = [row["object"] + "_" + row["label"] for row in train_data.examples]
run_rows, history_rows = [], []
for split_seed in SPLIT_SEEDS:
    fit_indices, validation_indices = train_test_split(
        range(len(train_data)), test_size=VALIDATION_FRACTION,
        stratify=strata, random_state=split_seed,
    )
    validation_loader = DataLoader(Subset(train_data, validation_indices), batch_size=BATCH_SIZE)
    for backbone_rate in BACKBONE_RATES:
        generator = torch.Generator().manual_seed(SEED)
        fit_loader = DataLoader(
            Subset(train_data, fit_indices), batch_size=BATCH_SIZE,
            shuffle=True, generator=generator,
        )
        model = build_model()
        optimizer = torch.optim.AdamW([
            {"params": model.layer4.parameters(), "lr": backbone_rate},
            {"params": model.fc.parameters(), "lr": HEAD_RATE},
        ], weight_decay=1e-4)
        loss_function = nn.BCEWithLogitsLoss()
        best_ap = -1.0
        checkpoint = OUTPUT_DIR / f"split_{split_seed}_lr_{backbone_rate:g}.pt"
        for epoch in range(1, EPOCHS + 1):
            # Evaluation mode freezes batch normalization, but gradients still update weights.
            model.eval()
            total_loss = 0.0
            for images, labels in fit_loader:
                optimizer.zero_grad()
                logits = model(images.to(device)).squeeze(1)
                loss = loss_function(logits, labels.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(labels)
            scores, labels = predict(model, validation_loader)
            ap = average_precision_score(labels, scores)
            history_rows.append({
                "split_seed": split_seed, "backbone_rate": backbone_rate,
                "epoch": epoch, "train_loss": total_loss / len(fit_indices), "validation_ap": ap,
            })
            print(f"split={split_seed}, lr={backbone_rate:g}, epoch={epoch}, validation AP={ap:.4f}")
            if ap > best_ap:
                best_ap, best_epoch = ap, epoch
                best_scores = scores.copy()
                torch.save(model.state_dict(), checkpoint)
        threshold = choose_threshold(labels, best_scores)
        decisions = best_scores >= threshold
        tn, fp, fn, tp = confusion_matrix(labels, decisions, labels=[0, 1]).ravel()
        run_rows.append({
            "split_seed": split_seed, "backbone_rate": backbone_rate,
            "best_epoch": best_epoch, "validation_ap": best_ap,
            "threshold": threshold, "recall": recall_score(labels, decisions),
            "false_alarms": fp, "missed_defects": fn, "checkpoint": str(checkpoint),
        })
        del optimizer, model

pd.DataFrame(run_rows).to_csv(OUTPUT_DIR / "validation_runs.csv", index=False)
pd.DataFrame(history_rows).to_csv(OUTPUT_DIR / "training_history.csv", index=False)
(OUTPUT_DIR / "finished.txt").write_text("complete\n")
print("FINISHED", flush=True)
