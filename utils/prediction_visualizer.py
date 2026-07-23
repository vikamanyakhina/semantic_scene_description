"""
utils/prediction_visualizer.py

Визуализация результатов сегментации.

Сохраняет:

RGB
Ground Truth
Prediction

"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


class PredictionVisualizer:

    def __init__(self, experiment_dir):

        self.output_dir = (
            Path(experiment_dir) /
            "predictions"
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # -----------------------------------------------------

    @staticmethod
    def decode_mask(mask):

        colors = np.array([
            [0, 0, 0],          # Background
            [255, 0, 0],        # Building
            [128,128,128],      # Road
            [0,0,255],          # Water
            [255,165,0],        # Barren
            [34,139,34],        # Forest
            [255,255,0],        # Agricultural
            [0,255,0]           # Urban Green
        ], dtype=np.uint8)

        return colors[mask]

    # -----------------------------------------------------

    @staticmethod
    def tensor_to_image(image):

        image = image.cpu().numpy()

        image = np.transpose(image, (1, 2, 0))

        image = image[:, :, :3]

        image = np.clip(image, 0, 1)

        return image

    # -----------------------------------------------------

    def save_prediction(
            self,
            image,
            gt,
            prediction,
            epoch,
            index=0
    ):

        image = self.tensor_to_image(image)

        gt = gt.cpu().numpy()

        prediction = prediction.cpu().numpy()

        gt = self.decode_mask(gt)

        prediction = self.decode_mask(prediction)

        fig, ax = plt.subplots(
            1,
            3,
            figsize=(15, 5)
        )

        ax[0].imshow(image)
        ax[0].set_title("RGB")

        ax[1].imshow(gt)
        ax[1].set_title("Ground Truth")

        ax[2].imshow(prediction)
        ax[2].set_title("Prediction")

        for a in ax:
            a.axis("off")

        plt.tight_layout()

        plt.savefig(

            self.output_dir /

            f"epoch_{epoch:03d}_{index}.png",

            dpi=300

        )

        plt.close()

    # -----------------------------------------------------

    @torch.no_grad()
    def save_batch(
            self,
            images,
            masks,
            outputs,
            epoch,
            number=3
    ):

        prediction = torch.argmax(
            outputs,
            dim=1
        )

        number = min(
            number,
            len(images)
        )

        for i in range(number):

            self.save_prediction(

                images[i],

                masks[i],

                prediction[i],

                epoch,

                i

            )