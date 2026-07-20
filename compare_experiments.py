"""
compare_experiments.py

Сравнение результатов различных экспериментов.

Поддерживает:

RGB

RGB + LBP

RGB + Entropy

RGB + Variance

RGB + Gabor

"""

from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

import config

def load_history(experiment_name):

    history_file = (
        Path(config.OUTPUT_DIR)
        / experiment_name
        / "history"
        / "history.json"
    )

    if not history_file.exists():
        return None

    with open(history_file, "r") as f:
        history = json.load(f)

    return history

def get_last_metrics(history):

    return {

        "Train Loss":
            history["train_loss"][-1],

        "Validation Loss":
            history["val_loss"][-1],

        "Pixel Accuracy":
            history["pixel_accuracy"][-1],

        "Mean IoU":
            history["mean_iou"][-1],

        "Mean Dice":
            history["mean_dice"][-1]

    }

def compare():

    rows = []

    for experiment, _ in config.EXPERIMENTS:

        history = load_history(experiment)

        if history is None:
            print(f"{experiment}: not found")
            continue

        metrics = get_last_metrics(history)

        metrics["Experiment"] = experiment

        rows.append(metrics)

    df = pd.DataFrame(rows)

    df = df[
        [
            "Experiment",
            "Train Loss",
            "Validation Loss",
            "Pixel Accuracy",
            "Mean IoU",
            "Mean Dice"
        ]
    ]

    return df

def print_table(df):

    print()

    print("="*70)

    print(df.to_string(index=False))

    print("="*70)

def save_table(df):

    save_dir = (
        Path(config.OUTPUT_DIR)
        / "comparison"
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(

        save_dir / "comparison.csv",

        index=False

    )

    return save_dir

def plot_metric(
        df,
        metric,
        save_dir
):

    plt.figure(figsize=(8,5))

    plt.bar(

        df["Experiment"],

        df[metric]

    )

    plt.ylabel(metric)

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(

        save_dir /

        f"{metric}.png",

        dpi=300

    )

    plt.close()

def build_plots(df, save_dir):

    plot_metric(df, "Pixel Accuracy", save_dir)

    plot_metric(df, "Mean IoU", save_dir)

    plot_metric(df, "Mean Dice", save_dir)

    plot_metric(df, "Train Loss", save_dir)

    plot_metric(df, "Validation Loss", save_dir)

def main():

    df = compare()

    print_table(df)

    save_dir = save_table(df)

    build_plots(df, save_dir)

    print()

    print("Comparison finished.")

    print(save_dir)

if __name__ == "__main__":

    main()