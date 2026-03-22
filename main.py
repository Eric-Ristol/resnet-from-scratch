#Main entry point — small Roman-numeral menu, same style as the other
#projects in this folder.
#
#Options:
#  I.   Download CIFAR-10 (first-time setup)
#  II.  Train ResNet-20
#  III. Train a tiny smoke-run (~256 images, 1 epoch)
#  IV.  Evaluate the best checkpoint
#  V.   Predict on a single image file
#  VI.  Quit

import os
import sys


def print_menu():
    print("")
    print("================================================")
    print("   RESNET-20 / CIFAR-10 — Main menu")
    print("================================================")
    print("  I.   Download CIFAR-10")
    print("  II.  Train ResNet-20 (80 epochs)")
    print("  III. Smoke-train (tiny subset, 1 epoch)")
    print("  IV.  Evaluate best checkpoint")
    print("  V.   Predict on a single image")
    print("  VI.  Launch API server (web demo)")
    print("  VII. Quit")
    print("------------------------------------------------")


def option_download():
    import data as d
    print("\n>>> Downloading CIFAR-10 into ./data (this can take a minute)...")
    d.get_datasets(download=True)
    print("    Done.")


def option_train():
    import train as t
    epochs = input("Number of epochs (default 80): ").strip()
    epochs = int(epochs) if epochs.isdigit() else 80
    t.run_training(epochs=epochs)


def option_smoke():
    import train as t
    t.run_training(epochs=1, batch_size=64, num_workers=0, tiny=True)


def option_evaluate():
    import evaluate as e
    try:
        e.run_evaluation()
    except FileNotFoundError as err:
        print("  [ERR]", err)


def option_predict():
    path = input("Path to an image file: ").strip()
    if not path:
        print("  [ERR] No path given.")
        return
    if not os.path.exists(path):
        print(f"  [ERR] File not found: {path}")
        return
    import evaluate as e
    try:
        label, prob, _ = e.predict_one(path)
    except FileNotFoundError as err:
        print("  [ERR]", err)
        return
    print(f"\nPredicted: {label}  (confidence {prob*100:.1f}%)")


def option_api():
    print("\n>>> Starting API server at http://localhost:8000")
    print("    Open that URL in your browser to classify images.")
    print("    Press Ctrl+C to stop the server.\n")
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=False)


def main():
    while True:
        print_menu()
        choice = input("Choose an option: ").strip().upper()
        if choice == "I":
            option_download()
        elif choice == "II":
            option_train()
        elif choice == "III":
            option_smoke()
        elif choice == "IV":
            option_evaluate()
        elif choice == "V":
            option_predict()
        elif choice == "VI":
            option_api()
        elif choice == "VII" or choice in ("Q", "QUIT", "EXIT"):
            print("Bye!")
            sys.exit(0)
        else:
            print("  [ERR] Unknown option. Try I, II, III, IV, V, VI or VII.")


if __name__ == "__main__":
    main()
