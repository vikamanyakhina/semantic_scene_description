"""
utils/visualization.py

Построение графиков обучения модели.

Создает:

- train_loss.png
- val_loss.png
- loss.png
- miou.png
- dice.png
- pixel_accuracy.png
- learning_rate.png
- epoch_time.png
- class_iou.png
- class_dice.png

"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class TrainingVisualizer:

    def __init__(self, experiment_dir):

        self.experiment_dir = Path(experiment_dir)

        self.plot_dir = self.experiment_dir / "plots"

        self.plot_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # ---------------------------------------------------------

    def _save_plot(self,
                   x,
                   y,
                   title,
                   xlabel,
                   ylabel,
                   filename):

        plt.figure(figsize=(8, 5))

        plt.plot(
            x,
            y,
            linewidth=2
        )

        plt.grid(True)

        plt.title(title)

        plt.xlabel(xlabel)

        plt.ylabel(ylabel)

        plt.tight_layout()

        plt.savefig(
            self.plot_dir / filename,
            dpi=300
        )

        plt.close()

    # ---------------------------------------------------------

    def plot_train_loss(self, history):

        self._save_plot(

            history["epoch"],

            history["train_loss"],

            "Training Loss",

            "Epoch",

            "Loss",

            "train_loss.png"

        )

    # ---------------------------------------------------------

    def plot_val_loss(self, history):

        self._save_plot(

            history["epoch"],

            history["val_loss"],

            "Validation Loss",

            "Epoch",

            "Loss",

            "val_loss.png"

        )

    # ---------------------------------------------------------

    def plot_loss(self, history):

        plt.figure(figsize=(8, 5))

        plt.plot(

            history["epoch"],

            history["train_loss"],

            label="Train"

        )

        plt.plot(

            history["epoch"],

            history["val_loss"],

            label="Validation"

        )

        plt.grid(True)

        plt.legend()

        plt.title("Loss")

        plt.xlabel("Epoch")

        plt.ylabel("Loss")

        plt.tight_layout()

        plt.savefig(

            self.plot_dir / "loss.png",

            dpi=300

        )

        plt.close()

    # ---------------------------------------------------------

    def plot_miou(self, history):

        self._save_plot(

            history["epoch"],

            history["mean_iou"],

            "Mean IoU",

            "Epoch",

            "mIoU",

            "miou.png"

        )

    # ---------------------------------------------------------

    def plot_dice(self, history):

        self._save_plot(

            history["epoch"],

            history["mean_dice"],

            "Mean Dice",

            "Epoch",

            "Dice",

            "dice.png"

        )

    # ---------------------------------------------------------

    def plot_accuracy(self, history):

        self._save_plot(

            history["epoch"],

            history["pixel_accuracy"],

            "Pixel Accuracy",

            "Epoch",

            "Accuracy",

            "pixel_accuracy.png"

        )

    # ---------------------------------------------------------

    def plot_learning_rate(self, history):

        self._save_plot(

            history["epoch"],

            history["learning_rate"],

            "Learning Rate",

            "Epoch",

            "LR",

            "learning_rate.png"

        )

    # ---------------------------------------------------------

    def plot_epoch_time(self, history):

        self._save_plot(

            history["epoch"],

            history["epoch_time"],

            "Epoch Time",

            "Epoch",

            "Seconds",

            "epoch_time.png"

        )

    # ---------------------------------------------------------

    def plot_class_iou(self,
                       class_names,
                       class_iou):

        plt.figure(figsize=(10, 5))

        plt.bar(

            np.arange(len(class_names)),

            class_iou

        )

        plt.xticks(

            np.arange(len(class_names)),

            class_names,

            rotation=30

        )

        plt.ylim(0, 1)

        plt.ylabel("IoU")

        plt.title("IoU for each class")

        plt.tight_layout()

        plt.savefig(

            self.plot_dir / "class_iou.png",

            dpi=300

        )

        plt.close()

    # ---------------------------------------------------------

    def plot_class_dice(self,
                        class_names,
                        class_dice):

        plt.figure(figsize=(10, 5))

        plt.bar(

            np.arange(len(class_names)),

            class_dice

        )

        plt.xticks(

            np.arange(len(class_names)),

            class_names,

            rotation=30

        )

        plt.ylim(0, 1)

        plt.ylabel("Dice")

        plt.title("Dice for each class")

        plt.tight_layout()

        plt.savefig(

            self.plot_dir / "class_dice.png",

            dpi=300

        )

        plt.close()

    # ---------------------------------------------------------

    def build_all_plots(self,
                        history,
                        class_names=None,
                        class_iou=None,
                        class_dice=None):

        self.plot_train_loss(history)

        self.plot_val_loss(history)

        self.plot_loss(history)

        self.plot_miou(history)

        self.plot_dice(history)

        self.plot_accuracy(history)

        self.plot_learning_rate(history)

        self.plot_epoch_time(history)

        if class_names is not None:

            self.plot_class_iou(

                class_names,

                class_iou

            )

            self.plot_class_dice(

                class_names,

                class_dice

            )