import cv2
import numpy as np
import random
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import torch

def class_id_to_string(class_id):
    # Convert torch scalar tensor -> Python int
    if isinstance(class_id, torch.Tensor):
        class_id = class_id.item()

    class_to_pair = {}
    cid = 0
    for hi in range(7):
        for lo in range(hi + 1):
            class_to_pair[cid] = (hi, lo)
            cid += 1

    hi, lo = class_to_pair[class_id]
    return f"{hi}-{lo}"

def stratified_split_indices(labels, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=0):
    if isinstance(labels, torch.Tensor):
        labels = labels.tolist()

    by_class = defaultdict(list)

    for i, y in enumerate(labels):
        by_class[int(y)].append(i)
    g = torch.Generator().manual_seed(seed)

    train_idx = []
    val_idx = []
    test_idx = []

    for cls, idxs in by_class.items():
        idxs = torch.tensor(idxs)
        idxs = idxs[torch.randperm(len(idxs), generator=g)].tolist()

        n = len(idxs)
        n_train = int(train_frac * n)
        n_val = int(val_frac * n)

        train_idx += idxs[:n_train]
        val_idx += idxs[n_train:n_train+n_val]
        test_idx += idxs[n_train+n_val:]

    return train_idx, val_idx, test_idx

def class_counts(dataset):
    counts = Counter()
    for i in range(len(dataset)):
        _, y = dataset[i]
        y = y.item() if torch.is_tensor(y) else y
        counts[y] += 1
    return [x[1] for x in sorted(counts.items())]
 

def show(dataset, i):
    """
    Display the i-th image from a PyTorch dataset using OpenCV.

    Args:
        dataset : PyTorch dataset returning (image, label)
        i       : index of sample to display
    """

    image, label = dataset[i]

    # Convert tensor to numpy
    if isinstance(image, torch.Tensor):
        img = image.detach().cpu().numpy()
    else:
        raise TypeError("Dataset image must be a torch.Tensor")

    # Convert from CHW â†’ HWC
    img = np.transpose(img, (1, 2, 0))

    # Convert from float [0,1] â†’ uint8 [0,255]
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)

    # Convert RGB â†’ BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    window_name = f"sample {i} label={label}"
    cv2.imshow(window_name, img)

    print(f"Index: {i}, label: {class_id_to_string(label)}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_grid(dataset, indices=None, rows=4, cols=6, scale=2):
    """
    Display a grid of images from a dataset using OpenCV.

    Args:
        dataset : PyTorch dataset returning (image, label)
        indices : optional list of dataset indices
        rows    : number of rows in the grid
        cols    : number of columns in the grid
        scale   : scaling factor for display
    """

    n = rows * cols

    if indices is None:
        indices = random.sample(range(len(dataset)), n)

    tiles = []

    for idx in indices:
        image, label = dataset[idx]

        img = image.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))

        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        label_str = class_id_to_string(label)

        cv2.putText(
            img,
            label_str,
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        tiles.append(img)

    # Build rows
    grid_rows = []
    for r in range(rows):
        row = np.hstack(tiles[r * cols:(r + 1) * cols])
        grid_rows.append(row)

    grid = np.vstack(grid_rows)

    # Optional scaling
    if scale != 1:
        grid = cv2.resize(
            grid,
            (grid.shape[1] * scale, grid.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST
        )

    cv2.imshow("dataset grid", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def collect_predictions(model, dataloader, device):
    model.eval()

    preds = []
    labels = []

    with torch.no_grad():
        for images, y in dataloader:
            images = images.to(device)

            logits = model(images)
            p = torch.argmax(logits, dim=1).cpu()

            preds.append(p)
            labels.append(y.cpu())

    preds = torch.cat(preds)
    labels = torch.cat(labels)

    return preds, labels

def confusion_matrix(preds, labels, num_classes=28):
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for p, t in zip(preds, labels):
        cm[t, p] += 1
    return cm


def print_confusion_matrix(cm):
    class_labels = [class_id_to_string(i) for i in range(28)]
    print("Confusion matrix (rows=true, cols=predicted)\n")
    print("      ", " ".join(f"{l:>4}" for l in class_labels))
    for i in range(28):
        row = " ".join(f"{cm[i,j]:4d}" for j in range(28))
        print(f"{class_labels[i]:>4}  {row}")

def plot_confusion_matrix(cm):
    class_labels = [class_id_to_string(i) for i in range(28)]

    plt.figure(figsize=(10,10))

    plt.imshow(cm, cmap="Blues")
    plt.colorbar()

    plt.xticks(range(28), class_labels, rotation=90)
    plt.yticks(range(28), class_labels)

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.title("Domino Confusion Matrix")

    plt.tight_layout()
    plt.show()

def top_confusions(cm, k=10):
    conf = []
    for t in range(28):
        for p in range(28):
            if t != p and cm[t,p] > 0:
                conf.append((cm[t,p].item(), t, p))
    conf.sort(reverse=True)
    if len(conf) == 0:
        print("New evalution: no confusions on new test set.")
    else:
        print("New evaluation -- top confusions:")
        for count, t, p in conf[:k]:
            print(f"{class_id_to_string(t)} â†’ {class_id_to_string(p)} : {count}")
        