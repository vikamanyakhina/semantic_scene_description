"""
utils/logger.py

Логирование процесса обучения.

Сохраняет:

- историю обучения;
- метрики;
- время эпох;
- csv;
- json.

"""

import json
import csv
from pathlib import Path


class TrainingLogger:

    def __init__(self, experiment_dir):

        self.experiment_dir = Path(experiment_dir)

        self.history_dir = self.experiment_dir / "history"

        self.history_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.history = {

            "epoch": [],

            "train_loss": [],

            "val_loss": [],

            "pixel_accuracy": [],

            "mean_iou": [],

            "mean_dice": [],

            "learning_rate": [],

            "epoch_time": []
        }

    # -------------------------------------------------------------

    def log_epoch(
            self,
            epoch,
            train_loss,
            val_loss,
            metrics,
            learning_rate,
            epoch_time
    ):

        self.history["epoch"].append(epoch)

        self.history["train_loss"].append(train_loss)

        self.history["val_loss"].append(val_loss)

        self.history["pixel_accuracy"].append(
            metrics["pixel_accuracy"]
        )

        self.history["mean_iou"].append(
            metrics["mean_iou"]
        )

        self.history["mean_dice"].append(
            metrics["mean_dice"]
        )

        self.history["learning_rate"].append(
            learning_rate
        )

        self.history["epoch_time"].append(
            epoch_time
        )

    # -------------------------------------------------------------

    def save_json(self):

        file = self.history_dir / "history.json"

        with open(
                file,
                "w",
                encoding="utf-8"
        ) as f:

            json.dump(
                self.history,
                f,
                ensure_ascii=False,
                indent=4
            )

    # -------------------------------------------------------------

    def save_csv(self):

        file = self.history_dir / "history.csv"

        with open(
                file,
                "w",
                newline="",
                encoding="utf-8"
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                "epoch",
                "train_loss",
                "val_loss",
                "pixel_accuracy",
                "mean_iou",
                "mean_dice",
                "learning_rate",
                "epoch_time"
            ])

            for i in range(len(self.history["epoch"])):

                writer.writerow([

                    self.history["epoch"][i],

                    self.history["train_loss"][i],

                    self.history["val_loss"][i],

                    self.history["pixel_accuracy"][i],

                    self.history["mean_iou"][i],

                    self.history["mean_dice"][i],

                    self.history["learning_rate"][i],

                    self.history["epoch_time"][i]

                ])

    # -------------------------------------------------------------

    def save(self):

        self.save_json()

        self.save_csv()