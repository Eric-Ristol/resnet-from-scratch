#Evaluation -- loads the best checkpoint and reports:
#  - top-1 test accuracy
#  - per-class accuracy
#  - a confusion matrix saved to plots/confusion_matrix.png
#Run with: python evaluate.py
#Or from the main.py menu.

import os
import numpy as np
import torch

import data as data_module
from data import CLASSES
from model import resnet20
from train import BEST_PATH, pick_device


PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")


def load_best(device):
    #Loads the best checkpoint into a fresh ResNet-20.
    if not os.path.exists(BEST_PATH):
        raise FileNotFoundError(
            f"No trained model at {BEST_PATH}. "
            "Train one first (option II in main.py)."
        )
    ckpt = torch.load(BEST_PATH, map_location=device)
    model = resnet20(num_classes=10).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def compute_confusion(model, loader, device, num_classes=10):
    #Returns a [C x C] numpy array where cm[i, j] is the count of
    #examples with true label i predicted as j.
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    correct_per_class = np.zeros(num_classes, dtype=np.int64)
    total_per_class   = np.zeros(num_classes, dtype=np.int64)

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb).argmax(dim=1)
            for t, p in zip(yb.cpu().numpy(), preds.cpu().numpy()):
                cm[t, p] += 1
                total_per_class[t]   += 1
                if t == p:
                    correct_per_class[t] += 1

    per_class_acc = correct_per_class / np.maximum(total_per_class, 1)
    overall_acc   = correct_per_class.sum() / max(total_per_class.sum(), 1)
    return cm, per_class_acc, overall_acc


def plot_confusion(cm, out_path, classes=CLASSES):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("  (skipping plot -- matplotlib not available:", e, ")")
        return

    #Row-normalise so the diagonal is "% of this true class predicted correctly".
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("ResNet-20 on CIFAR-10 -- confusion matrix (row-normalised)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = cm_norm[i, j]
            if v > 0.02:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run_evaluation(download=True):
    device = pick_device()
    print("Device:", device)

    _, test_loader = data_module.get_loaders(
        batch_size=256, num_workers=2, download=download
    )
    model, ckpt = load_best(device)
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(reported best_acc={ckpt['best_acc']*100:.2f}%).")

    cm, per_class, overall = compute_confusion(model, test_loader, device)

    print(f"\nTest accuracy: {overall*100:.2f}%")
    print("\nPer-class accuracy:")
    for name, acc in zip(CLASSES, per_class):
        bar = "#" * int(acc * 40)
        print(f"  {name:12s} {acc*100:6.2f}%  {bar}")

    out = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plot_confusion(cm, out)
    print(f"\nSaved confusion matrix to {out}")

    return overall, per_class, cm


def predict_one(image_path):
    #Classify a single image file using the best checkpoint.
    from PIL import Image
    from torchvision import transforms
    from data import CIFAR10_MEAN, CIFAR10_STD

    device = pick_device()
    model, _ = load_best(device)

    tfm = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    img = Image.open(image_path).convert("RGB")
    x   = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    top_idx  = int(probs.argmax().item())
    top_prob = float(probs[top_idx].item())
    return CLASSES[top_idx], top_prob, probs.cpu().tolist()


if __name__ == "__main__":
    run_evaluation()
