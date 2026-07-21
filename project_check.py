from pathlib import Path
import json

import config


def check():

    experiment = Path(config.OUTPUT_DIR) / config.EXPERIMENTS [0][0]

    print("=" * 60)
    print("PROJECT CHECK")
    print("=" * 60)

    folders = [
        "checkpoints",
        "history",
        "plots",
        "predictions"
    ]

    print("\nFolders:")

    for folder in folders:

        p = experiment / folder

        print(f"{folder:15}", "OK" if p.exists() else "MISSING")

    print("\nFiles:")

    files = [

        "checkpoints/best_model.pth",

        "history/history.json",

        "history/history.csv",

        "plots/loss.png",

        "plots/miou.png",

        "plots/dice.png",

        "plots/pixel_accuracy.png",

        "plots/class_iou.png",

        "plots/class_dice.png"

    ]

    for file in files:

        p = experiment / file

        print(f"{file:35}", "OK" if p.exists() else "MISSING")

    history = experiment / "history/history.json"

    if history.exists():

        with open(history) as f:

            hist = json.load(f)

        print("\nEpochs:", len(hist["epoch"]))

        print("Final Train Loss :", hist["train_loss"][-1])

        print("Final Val Loss   :", hist["val_loss"][-1])

        print("Final mIoU       :", hist["mean_iou"][-1])

        print("Final Dice       :", hist["mean_dice"][-1])

        print("Final Accuracy   :", hist["pixel_accuracy"][-1])

    print("\nPredictions:")

    preds = list((experiment / "predictions").glob("*.png"))

    print(len(preds), "images")

    print("=" * 60)


if __name__ == "__main__":

    check()