#Training loop for ResNet-20 on CIFAR-10.
#
#Recipe (from the original ResNet paper, Sec. 4.2):
#  - SGD, momentum 0.9, weight decay 1e-4
#  - mini-batch size 128
#  - learning rate 0.1, divided by 10 at epochs 80 and 120
#  - train for 160 epochs total
#  - weight init: He / Kaiming (done in model.py)
#
#For this educational reimplementation I default to:
#  - epochs=80 (gets within ~1% of full recipe, half the time)
#  - cosine LR schedule (slightly simpler than the paper's step decay,
#    a tiny bit better accuracy, and a nicer smooth curve to show off
#    in the README plot).
#
#On CPU a single epoch is ~4-6 minutes. On a modest GPU it's seconds.

import argparse
import os
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim

import data as data_module
from model import resnet20, count_parameters


MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
PLOTS_DIR  = os.path.join(os.path.dirname(__file__), "plots")
BEST_PATH     = os.path.join(MODELS_DIR, "best.pt")
HISTORY_PATH  = os.path.join(MODELS_DIR, "history.json")


def pick_device():
    #Use CUDA if available, then Apple MPS, else CPU. Same preference
    #order as the LLM-finetune project, for consistency.
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def accuracy(logits, targets):
    #Top-1 classification accuracy as a float in [0, 1].
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def run_epoch(model, loader, device, criterion, optimizer=None):
    #One full pass over `loader`. If `optimizer` is given we train,
    #otherwise we evaluate. Returns (mean_loss, mean_accuracy).
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_acc  = 0.0
    total_n    = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb)
            loss   = criterion(logits, yb)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_n     = xb.size(0)
            total_loss += loss.item() * batch_n
            total_acc  += accuracy(logits, yb) * batch_n
            total_n    += batch_n

    return total_loss / total_n, total_acc / total_n


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_acc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch":     epoch,
        "best_acc":  best_acc,
    }, path)


def plot_history(history, out_path):
    #Draws two stacked subplots: loss over epochs, and accuracy over epochs.
    #Saved as a PNG so it can be embedded in the README.
    try:
        import matplotlib
        matplotlib.use("Agg")  #no display needed — saving to file only
        import matplotlib.pyplot as plt
    except Exception as e:
        print("  (skipping plot — matplotlib not available:", e, ")")
        return

    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6))

    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["test_loss"],  label="test")
    ax1.set_ylabel("loss")
    ax1.set_title("ResNet-20 on CIFAR-10")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["test_acc"],  label="test")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run_training(epochs=80, batch_size=128, lr=0.1, momentum=0.9,
                 weight_decay=1e-4, num_workers=2, tiny=False,
                 device=None, download=True):
    #Full training routine. `tiny=True` runs on a small subset — useful
    #for unit tests and CI.
    device = device or pick_device()
    print("Device:", device)

    if tiny:
        train_loader, test_loader = data_module.get_tiny_subset_loaders(
            n_train=256, n_test=128, batch_size=batch_size,
            num_workers=0, download=download,
        )
    else:
        train_loader, test_loader = data_module.get_loaders(
            batch_size=batch_size, num_workers=num_workers, download=download,
        )

    model = resnet20(num_classes=10).to(device)
    print("Parameters:", count_parameters(model))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum,
        weight_decay=weight_decay, nesterov=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "train_loss": [], "train_acc": [],
        "test_loss":  [], "test_acc":  [],
        "lr": [],
    }
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, device, criterion, optimizer=optimizer
        )
        test_loss,  test_acc  = run_epoch(
            model, test_loader, device, criterion, optimizer=None
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        took = time.time() - t0
        print(
            f"epoch {epoch:3d}/{epochs}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc*100:5.2f}%  "
            f"test_loss={test_loss:.4f} test_acc={test_acc*100:5.2f}%  "
            f"lr={history['lr'][-1]:.4f}  ({took:.1f}s)"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(BEST_PATH, model, optimizer, scheduler, epoch, best_acc)

    #Final history + plot.
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f)

    plot_history(history, os.path.join(PLOTS_DIR, "learning_curves.png"))

    print(f"\nDone. Best test accuracy: {best_acc*100:.2f}%  saved to {BEST_PATH}")
    return history, best_acc


def main():
    p = argparse.ArgumentParser(description="Train ResNet-20 on CIFAR-10.")
    p.add_argument("--epochs",       type=int,   default=80)
    p.add_argument("--batch-size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=0.1)
    p.add_argument("--momentum",     type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers",  type=int,   default=2)
    p.add_argument("--tiny", action="store_true",
                   help="Train on a small subset (smoke test / CI).")
    p.add_argument("--no-download", action="store_true",
                   help="Assume CIFAR-10 is already present in ./data.")
    args = p.parse_args()

    run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        tiny=args.tiny,
        download=not args.no_download,
    )


if __name__ == "__main__":
    main()
