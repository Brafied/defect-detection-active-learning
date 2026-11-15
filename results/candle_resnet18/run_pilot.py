"""Run from the repo root after the notebook's candle dataset is available."""
import csv
import json
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix

out = Path(__file__).resolve().parent
book = json.loads(Path("visa_resnet18_baseline.ipynb").read_text())
cells = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
# Use the notebook's exact preparation, preprocessing, initialization, and optimizer.
torch.set_num_threads(3)
ns = {}
for cell in cells[:3]:
    exec("".join(cell["source"]), ns)
model, optimizer, loss_fn = ns["model"], ns["optimizer"], ns["loss_function"]
device = ns["device"]
train, test = ns["training_split"], ns["testing_split"]
train_paths = [r["image"] for r in train.examples]
test_paths = [r["image"] for r in test.examples]
assert len(set(train_paths)) == len(train_paths) == 660
assert len(set(test_paths)) == len(test_paths) == 440
assert set(train_paths).isdisjoint(test_paths)

# The encoder is frozen in eval mode and preprocessing is deterministic.
# Save one feature vector per image, preserving the order in the split CSV.
head = model.fc
model.fc = nn.Identity()
@torch.no_grad()
def features(dataset):
    vectors, labels = [], []
    for start in range(0, len(dataset), 16):
        batch = [dataset[i] for i in range(start, min(start + 16, len(dataset)))]
        images, targets = zip(*batch)
        vectors.append(model(torch.stack(images).to(device)).cpu())
        labels.append(torch.stack(targets))
    return TensorDataset(torch.cat(vectors), torch.cat(labels))
train_cache, test_cache = features(train), features(test)
train_loader = DataLoader(train_cache, batch_size=16, shuffle=True)
test_loader = DataLoader(test_cache, batch_size=16)
history = []
for epoch in range(1, 11):
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(head(x).squeeze(1), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    with torch.no_grad():
        probabilities, labels = [], []
        for x, y in test_loader:
            probabilities.extend(head(x.to(device)).squeeze(1).sigmoid().cpu().tolist())
            labels.extend(int(label) for label in y)
    predictions = [int(p >= 0.5) for p in probabilities]
    tn, fp, fn, tp = map(int, confusion_matrix(labels, predictions, labels=[0,1]).ravel())
    history.append(dict(epoch=epoch, train_loss=total_loss / len(train), accuracy=(tp+tn)/len(test),
                        average_precision=average_precision_score(labels, probabilities),
                        roc_auc=roc_auc_score(labels, probabilities), tn=tn, fp=fp, fn=fn, tp=tp,
                        precision=tp/(tp+fp) if tp+fp else None, recall=tp/(tp+fn),
                        f1=2*tp/(2*tp+fp+fn)))
    print(history[-1], flush=True)

def write_rows(name, rows):
    with (out/name).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
write_rows("epoch_metrics.csv", history)
write_rows("test_predictions.csv", [dict(image=p, label=y, probability=prob, prediction=pred)
            for p,y,prob,pred in zip(test_paths,labels,probabilities,predictions,strict=True)])
print("Finished", Counter(labels))
