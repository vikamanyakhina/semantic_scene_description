from pathlib import Path
import json

import config


def check_experiment(name):

    print("=" * 60)
    print(name)
    print("=" * 60)

    exp = Path(config.OUTPUT_DIR) / name

    folders = [
        "checkpoints",
        "history",
        "plots",
        "predictions"
    ]

    for folder in folders:

        ok = (exp / folder).exists()

        print(f"{folder:<15} {'OK' if ok else 'MISSING'}")

    history = exp / "history" / "history.json"

    if history.exists():

        with open(history) as f:

            hist = json.load(f)

        print()

        print("Epochs:", len(hist["epoch"]))

        print("Final mIoU :", hist["mean_iou"][-1])

        print("Final Dice :", hist["mean_dice"][-1])

        print("Final Acc  :", hist["pixel_accuracy"][-1])

    print()


if __name__ == "__main__":

    for experiment_name, _ in config.EXPERIMENTS:

        check_experiment(experiment_name)