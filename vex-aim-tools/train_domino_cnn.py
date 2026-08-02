import os
import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from DominoCNN import DominoCNN
from utils import *

# ============================================================
# Label mapping helpers
# ============================================================

def make_domino_class_maps():
    """
    Canonical class ordering:
        0-0, 1-0, 1-1, 2-0, 2-1, 2-2, ..., 6-6
    where each pair is stored as (hi, lo).
    """
    pair_to_class = {}
    class_to_pair = {}
    class_id = 0
    for hi in range(0, 7):
        for lo in range(0, hi + 1):
            pair = (hi, lo)
            pair_to_class[pair] = class_id
            class_to_pair[class_id] = pair
            class_id += 1
    return pair_to_class, class_to_pair


def class_id_to_string(class_id: int) -> str:
    _, class_to_pair = make_domino_class_maps()
    hi, lo = class_to_pair[class_id]
    return f"{hi}-{lo}"

# ============================================================
# Dataset loading
# ============================================================

class AugmentedSubset(Dataset):
    """
    Wrap a Subset (or any dataset returning image, label) and optionally
    apply a transform to the image.
    """
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class TensorImageDataset(Dataset):
    """
    Dataset backed by cached tensors:
        images: [N, C, H, W]
        labels: [N]
    """
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels.long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]



def load_cached_dataset(pt_file: str) -> Tuple[list, list]:
    data = torch.load(pt_file, map_location="cpu", weights_only=True)
    images = data["images"]
    labels = data["labels"]

    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.long)
    else:
        labels = labels.long()

    return TensorImageDataset(images, labels)


# ============================================================
# Utility functions
# ============================================================

def set_seed(seed: int = 1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total


def evaluate(model, dataloader, criterion, device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_examples += images.size(0)

    avg_loss = total_loss / total_examples
    avg_acc = total_correct / total_examples
    return avg_loss, avg_acc


# ----------------------------
# Configuration
# ----------------------------

data_file = "domino_dataset.pt"
batch_size = 64
learning_rate = 3e-4
num_epochs = 60
seed = 1234
model_output_file = "best_domino_cnn.pt"

set_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------------------------------
# Load the unified dataset
# ------------------------------------------------
full_dataset = load_cached_dataset("domino_dataset.pt")

# reproducible stratified split
train_frac=0.7
val_frac=0.15
test_frac=0.15
labels = full_dataset.labels
train_idx, val_idx, test_idx = stratified_split_indices(labels, train_frac, val_frac, test_frac)

train_base = Subset(full_dataset, train_idx)
val_base = Subset(full_dataset, val_idx)
test_base = Subset(full_dataset, test_idx)

def print_counts(name, counts):
    print(name, " ".join(f"{count:2d}" for count in counts))

print("Stratified split:")
print_counts("  train counts:", class_counts(train_base))
print_counts("    val counts:", class_counts(val_base))
print_counts("   test counts:", class_counts(test_base))

class RandomRotate180:
    """
    Rotate tensor image 180 degrees with probability p.
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            image = torch.rot90(image, 2, dims=(1, 2))  # rotate H,W
        return image

train_transform = transforms.Compose([
    transforms.RandomAffine(
        degrees=8,
        translate=(0.08, 0.08),
        scale=(0.95, 1.05),
        interpolation=InterpolationMode.BILINEAR,
        fill=1.0,
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    RandomRotate180(p=0.5),
])

eval_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    RandomRotate180(p=0.5),
])

# Apply augmentation only to training split
train_dataset = AugmentedSubset(train_base, transform=train_transform)
val_dataset = AugmentedSubset(val_base, transform=eval_transform)
test_dataset = AugmentedSubset(test_base, transform=eval_transform)

# ------------------------------------------------
# DataLoaders
# ------------------------------------------------
batch_size = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
)

print(f"Total:      {len(full_dataset)}")
print(f"Train:      {len(train_dataset)}")
print(f"Validation: {len(val_dataset)}")
print(f"Test:       {len(test_dataset)}")

# ----------------------------
# Build model
# ----------------------------

model = DominoCNN(num_classes=28).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# ============================================================
# Training
# ============================================================

def main():
    # ----------------------------
    # Training loop
    # ----------------------------
    best_val_acc = 0.0
    best_val_loss = np.inf
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        model.train()

        running_loss = 0.0
        running_correct = 0
        running_examples = 0

        for images_batch, labels_batch in train_loader:
            images_batch = images_batch.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()

            logits = model(images_batch)
            loss = criterion(logits, labels_batch)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images_batch.size(0)
            preds = torch.argmax(logits, dim=1)
            running_correct += (preds == labels_batch).sum().item()
            running_examples += images_batch.size(0)

        train_loss = running_loss / running_examples
        train_acc = running_correct / running_examples

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:2d}/{num_epochs} | "
            f"train loss {train_loss:.4f} | train acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} | val acc {val_acc:.4f}"
        )

        if val_acc >= best_val_acc and val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "epoch": epoch,
                },
                model_output_file,
            )
            #print(f"  Saved best model to {model_output_file}")

    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best epoch: {best_epoch}")

    # ----------------------------
    # Final test evaluation
    # ----------------------------
    checkpoint = torch.load(model_output_file, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print("\nFinal test results using best validation checkpoint:")
    print(f"  test loss: {test_loss:.4f}")
    print(f"  test acc:  {test_acc:.4f}")
    
    test_predictions, test_labels = collect_predictions(model, test_loader, device)
    cm = confusion_matrix(test_predictions, test_labels)
    #print_confusion_matrix(cm)
    #plot_confusion_matrix(cm)
    top_confusions(cm)

if __name__ == "__main__":
    main()