"""
inference.py

Запуск обученной модели
и формирование семантического описания сцены.
"""

from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torchvision.transforms import ToTensor

import config

from models.unet import UNet

from scene_description import SceneDescription

from utils.prediction_visualizer import PredictionVisualizer

def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

def load_model(device, experiment_name):

    model = UNet(
        in_channels=config.IN_CHANNELS,
        num_classes=config.NUM_CLASSES
    )

    checkpoint = (
        Path(config.OUTPUT_DIR)
        / experiment_name
        / "checkpoints"
        / "best_model.pth"
    )

    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    return model

def get_test_images():

    images = []

    for area in ["Urban", "Rural"]:

        folder = (

            Path(config.DATASET_PATH)

            / "Test"

            / area

            / "images_png"

        )

        if folder.exists():

            images.extend(

                sorted(folder.glob("*.png"))

            )

    return images

@torch.no_grad()

def predict(

        model,

        image_path,

        device

):

    image = Image.open(

        image_path

    ).convert("RGB")

    image_np = np.array(image)

    tensor = ToTensor()(image)

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)

    output = model(tensor)

    prediction = torch.argmax(

        output,

        dim=1

    )

    prediction = prediction.squeeze().cpu().numpy()

    return image_np, prediction

def main(
        experiment_name,
        texture
      ):

    device = get_device()

    print()

    print("=" * 60)

    print("Inference")

    print("=" * 60)

    print()

    model = load_model(
        device,
        experiment_name
    )
    visualizer = PredictionVisualizer(

        Path(config.OUTPUT_DIR)

        / experiment_name

        / "inference"

    )

    images = get_test_images()

    print(f"Найдено изображений: {len(images)}")

    for image_path in images:

        print(image_path.name)

        image, prediction = predict(

            model,

            image_path,

            device

        )

        save_dir = (

            Path(config.OUTPUT_DIR)

            / experiment_name

            / "inference"

            / image_path.stem

        )

        save_dir.mkdir(

            parents=True,

            exist_ok=True

        )

        Image.fromarray(

            image

        ).save(

            save_dir /

            "image.png"

        )

        Image.fromarray(

            prediction.astype(np.uint8)

        ).save(

            save_dir /

            "prediction.png"

        )

        color = visualizer.decode_mask(

            prediction

        )

        Image.fromarray(

            color

        ).save(

            save_dir /

            "prediction_color.png"

        )

        scene = SceneDescription(

            save_dir

        )

        description = scene.describe(

            prediction

        )

        print(description)

        print("-" * 60)

    print()

    print("Инференс завершен.")

if __name__ == "__main__":

    for experiment_name, texture in config.EXPERIMENTS:

        print()
        print("=" * 70)
        print(experiment_name)
        print("=" * 70)

        main(
            experiment_name,
            texture
        )