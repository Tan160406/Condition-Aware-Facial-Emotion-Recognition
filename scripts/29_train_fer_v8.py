
from pathlib import Path
import random, time
import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# ================= FER V8 CONFIGURATION =================

SEED = 42
CLASSES = ["anger", "fear", "happiness", "neutral", "sadness", "surprise"]

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Dataset"
IMAGES = DATA / "images"

TRAIN_CSV = DATA / "train_labels.csv"
VAL_CSV = DATA / "val_labels.csv"

CHECKPOINT = DATA / "fer_v8_best.pth"
HISTORY = DATA / "fer_v8_history.csv"

SIZE = 112
BATCH = 32
PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 20
PATIENCE = 7

ImageFile.LOAD_TRUNCATED_IMAGES = True
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

print("=" * 65)
print("FER V8 — SIX CLASS RESNET18")
print("Device:", DEVICE)
print("Classes:", CLASSES)
print("=" * 65)

# ================= TRANSFORMS =================

train_tf = transforms.Compose([
    transforms.Resize((SIZE, SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(
        degrees=8,
        translate=(0.04, 0.04),
        scale=(0.95, 1.05)
    ),
    transforms.ColorJitter(brightness=0.10, contrast=0.10),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

val_tf = transforms.Compose([
    transforms.Resize((SIZE, SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# ================= DATASET =================

class FERDataset(Dataset):
    def __init__(self, csv_path, transform):
        self.df = pd.read_csv(csv_path)
        self.df = self.df.dropna(subset=["Image name", "emotion"]).copy()

        self.df["Image name"] = self.df["Image name"].astype(str).str.strip()
        self.df["emotion"] = (
            self.df["emotion"].astype(str).str.strip().str.lower()
        )

        self.df = self.df[
            self.df["emotion"].isin(CLASSES)
        ].reset_index(drop=True)

        self.df["label"] = self.df["emotion"].map(
            {name: i for i, name in enumerate(CLASSES)}
        )

        self.transform = transform

        missing = [
            name for name in self.df["Image name"]
            if not (IMAGES / name).is_file()
        ]

        if missing:
            raise FileNotFoundError(
                f"{len(missing)} images missing. Examples: {missing[:5]}"
            )

        print(f"\n{Path(csv_path).name}: {len(self.df)} samples")
        print(
            self.df["emotion"].value_counts()
            .reindex(CLASSES, fill_value=0)
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        with Image.open(IMAGES / row["Image name"]) as image:
            image = image.convert("L").convert("RGB")
            image = self.transform(image)

        return image, int(row["label"])


train_ds = FERDataset(TRAIN_CSV, train_tf)
val_ds = FERDataset(VAL_CSV, val_tf)

counts = (
    train_ds.df["label"]
    .value_counts()
    .reindex(range(6), fill_value=0)
    .to_numpy()
)

if np.any(counts == 0):
    raise ValueError("A selected class has no training samples.")

# Moderate square-root inverse-frequency sampling.
# No additional class weights are applied to the loss.

sample_weights = 1.0 / np.sqrt(
    counts[train_ds.df["label"].to_numpy()]
)

sampler = WeightedRandomSampler(
    torch.as_tensor(sample_weights, dtype=torch.double),
    num_samples=len(train_ds),
    replacement=True
)

train_loader = DataLoader(
    train_ds, batch_size=BATCH,
    sampler=sampler, num_workers=0
)

val_loader = DataLoader(
    val_ds, batch_size=BATCH,
    shuffle=False, num_workers=0
)

# ================= MODEL =================

print("\nLoading ImageNet-pretrained ResNet18...")

model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

model.fc = nn.Sequential(
    nn.Dropout(0.35),
    nn.Linear(model.fc.in_features, 6)
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.03)


def configure_phase(phase):
    for p in model.parameters():
        p.requires_grad = False

    for p in model.layer4.parameters():
        p.requires_grad = True

    for p in model.fc.parameters():
        p.requires_grad = True

    if phase == 2:
        for p in model.layer3.parameters():
            p.requires_grad = True


def make_optimizer(phase):
    if phase == 1:
        groups = [
            {"params": model.layer4.parameters(), "lr": 3e-5},
            {"params": model.fc.parameters(), "lr": 2e-4}
        ]
    else:
        groups = [
            {"params": model.layer3.parameters(), "lr": 1e-5},
            {"params": model.layer4.parameters(), "lr": 4e-5},
            {"params": model.fc.parameters(), "lr": 2e-4}
        ]

    return torch.optim.AdamW(groups, weight_decay=1e-4)


# ================= VALIDATION =================

def evaluate():
    model.eval()
    actual, predicted = [], []
    total_loss, total = 0.0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * len(labels)
            total += len(labels)

            actual.extend(labels.cpu().tolist())
            predicted.extend(outputs.argmax(1).cpu().tolist())

    accuracy = accuracy_score(actual, predicted)

    macro_f1 = f1_score(
        actual, predicted,
        labels=list(range(6)),
        average="macro",
        zero_division=0
    )

    return total_loss / total, accuracy, macro_f1, actual, predicted


# ================= TRAINING =================

best_f1 = -1.0
best_accuracy = 0.0
best_epoch = 0
stale = 0
epoch_number = 0
history = []

start = time.time()

for phase, max_epochs in [
    (1, PHASE1_EPOCHS),
    (2, PHASE2_EPOCHS)
]:

    configure_phase(phase)
    optimizer = make_optimizer(phase)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-7
    )

    print(f"\nSTARTING PHASE {phase}")

    for phase_epoch in range(1, max_epochs + 1):

        epoch_number += 1
        model.train()

        running_loss = 0.0
        train_true, train_pred = [], []

        for batch, (images, labels) in enumerate(train_loader, 1):

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                2.0
            )

            optimizer.step()

            running_loss += loss.item() * len(labels)

            train_true.extend(labels.detach().cpu().tolist())
            train_pred.extend(
                outputs.argmax(1).detach().cpu().tolist()
            )

            if batch % 150 == 0:
                print(
                    f"Epoch {epoch_number}: "
                    f"batch {batch}/{len(train_loader)}"
                )

        train_loss = running_loss / len(train_ds)

        train_accuracy = accuracy_score(
            train_true, train_pred
        )

        train_f1 = f1_score(
            train_true, train_pred,
            labels=list(range(6)),
            average="macro",
            zero_division=0
        )

        val_loss, val_accuracy, val_f1, _, _ = evaluate()

        scheduler.step(val_f1)

        record = {
            "epoch": epoch_number,
            "phase": phase,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_f1
        }

        history.append(record)
        pd.DataFrame(history).to_csv(HISTORY, index=False)

        print("\n" + "-" * 60)
        print("Epoch:", epoch_number)
        print(f"Train accuracy: {train_accuracy:.4f}")
        print(f"Train macro-F1: {train_f1:.4f}")
        print(f"Validation accuracy: {val_accuracy:.4f}")
        print(f"Validation macro-F1: {val_f1:.4f}")
        print(f"Validation loss: {val_loss:.4f}")

        if val_f1 > best_f1:

            best_f1 = val_f1
            best_accuracy = val_accuracy
            best_epoch = epoch_number
            stale = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": CLASSES,
                "num_classes": 6,
                "image_size": SIZE,
                "imagenet_mean": MEAN,
                "imagenet_std": STD,
                "architecture": "resnet18",
                "source_image_size": 48,
                "best_epoch": best_epoch,
                "best_val_accuracy": best_accuracy,
                "best_val_macro_f1": best_f1,
                "seed": SEED
            }, CHECKPOINT)

            print("BEST CHECKPOINT SAVED:", CHECKPOINT)

        else:
            stale += 1
            print(f"No improvement: {stale}/{PATIENCE}")

        if stale >= PATIENCE:
            print("Early stopping.")
            break

    if stale >= PATIENCE:
        break


# ================= FINAL VALIDATION REPORT =================

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])

_, accuracy, macro_f1, actual, predicted = evaluate()

print("\n" + "=" * 65)
print("FER V8 TRAINING COMPLETE")
print("=" * 65)

print("Best epoch:", best_epoch)
print(f"Validation accuracy: {accuracy:.4f}")
print(f"Validation macro-F1: {macro_f1:.4f}")
print(f"Elapsed minutes: {(time.time()-start)/60:.1f}")

print("\nCLASSIFICATION REPORT")

print(classification_report(
    actual, predicted,
    labels=list(range(6)),
    target_names=CLASSES,
    digits=4,
    zero_division=0
))

print("\nCONFUSION MATRIX")

print(confusion_matrix(
    actual, predicted,
    labels=list(range(6))
))

print("\nTest split has NOT been used.")
print("Checkpoint:", CHECKPOINT)
print("History:", HISTORY)
