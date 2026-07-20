"""
train.py

Обучение модели семантической сегментации.

Проект:
Semantic Scene Description
"""

from pathlib import Path
import time
import random

import numpy as np
import torch
from models.losses import CombinedLoss

from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from dataset.augmentations import (
    get_train_augmentation,
    get_val_augmentation
)

# --------------------------------------------------------
# Конфигурация проекта
# --------------------------------------------------------

import config

# --------------------------------------------------------
# Dataset
# --------------------------------------------------------

from my_datasets.loveda_dataset import LoveDADataset

# --------------------------------------------------------
# Model
# --------------------------------------------------------

from models.unet import UNet

# --------------------------------------------------------
# Utils
# --------------------------------------------------------

from utils.metrics import SegmentationMetrics

from utils.logger import TrainingLogger

from utils.visualization import TrainingVisualizer

from utils.prediction_visualizer import PredictionVisualizer

# --------------------------------------------------------
# Оптимизация
# --------------------------------------------------------

from torch.optim import AdamW

from torch.optim.lr_scheduler import ReduceLROnPlateau

# --------------------------------------------------------
# Для воспроизводимости
# --------------------------------------------------------

SEED = 42

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

torch.cuda.manual_seed_all(SEED)

# --------------------------------------------------------
# Device
# --------------------------------------------------------

def get_device():

    if torch.cuda.is_available():

        device = torch.device("cuda")

    elif torch.backends.mps.is_available():

        device = torch.device("mps")

    else:

        device = torch.device("cpu")

    print()

    print("=" * 60)

    print("Device:", device)

    print("=" * 60)

    print()

    return device

# --------------------------------------------------------
# Создание папок эксперимента
# --------------------------------------------------------

def create_experiment():

    experiment_name = config.EXPERIMENT_NAME

    output_dir = (

        Path(config.OUTPUT_DIR)

        / experiment_name

    )

    folders = [

        output_dir,

        output_dir / "plots",

        output_dir / "history",

        output_dir / "metrics",

        output_dir / "predictions",

        output_dir / "checkpoints"

    ]

    for folder in folders:

        folder.mkdir(

            parents=True,

            exist_ok=True

        )

    return output_dir

# --------------------------------------------------------
# Dataset
# --------------------------------------------------------

def build_dataloaders():

    train_transform = get_train_augmentation(
        config.IMAGE_SIZE
    )

    val_transform = get_val_augmentation(
        config.IMAGE_SIZE
    )

    train_dataset = LoveDADataset(

        root_dir=config.DATASET_PATH,

        split="Train",

        use_texture=config.USE_TEXTURE,

        texture=config.TEXTURE_TYPE,

        transform=train_transform

    )

    val_dataset = LoveDADataset(

        root_dir=config.DATASET_PATH,

        split="Val",

        use_texture=config.USE_TEXTURE,

        texture=config.TEXTURE_TYPE,

        transform=val_transform

    )

    print()
    print(f"Train images      : {len(train_dataset)}")
    print(f"Validation images : {len(val_dataset)}")
    print()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda")
    )

    return train_loader, val_loader
# --------------------------------------------------------
# Model
# --------------------------------------------------------

def build_model(device):

    model = UNet(

        in_channels=config.IN_CHANNELS,

        num_classes=config.NUM_CLASSES

    )

    model.to(device)

    return model

# --------------------------------------------------------
# Optimizer
# --------------------------------------------------------

def build_optimizer(model):

    optimizer = AdamW(

        model.parameters(),

        lr=config.LEARNING_RATE,

        weight_decay=1e-4

    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

      optimizer,

      T_max=config.NUM_EPOCHS,

      eta_min=1e-6

    )

    return optimizer, scheduler

# --------------------------------------------------------
# Loss
# --------------------------------------------------------

def build_loss():

    criterion = CombinedLoss(
        ce_weight=0.5,
        dice_weight=0.5,
        focal_weight=0.3
    )
    return criterion

# --------------------------------------------------------
# Utils
# --------------------------------------------------------

def build_utils(output_dir):

    logger = TrainingLogger(

        output_dir

    )

    metrics = SegmentationMetrics(

        config.NUM_CLASSES

    )

    visualizer = TrainingVisualizer(

        output_dir

    )

    prediction_visualizer = PredictionVisualizer(

        output_dir

    )

    return (

        logger,

        metrics,

        visualizer,

        prediction_visualizer

    )

# --------------------------------------------------------
# Обучение одной эпохи
# --------------------------------------------------------

def train_one_epoch(
        model,
        loader,
        optimizer,
        criterion,
        metrics,
        device
        ):
    """
    Обучение модели в течение одной эпохи.

    Parameters
    ----------
    model : nn.Module

    loader : DataLoader

    optimizer : torch.optim

    criterion : Loss

    metrics : SegmentationMetrics

    device : torch.device

    Returns
    -------
    train_loss : float

    epoch_time : float
    """

    model.train()

    metrics.reset()

    running_loss = 0.0

    start_time = time.time()

    progress = tqdm(
        loader,
        desc="Train",
        leave=False
    )

    for images, masks in progress:

        images = images.to(device)

        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            masks
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        metrics.update(
            outputs,
            masks
        )

        progress.set_postfix(

            loss=f"{loss.item():.4f}"

        )

    epoch_time = time.time() - start_time

    train_loss = (

        running_loss /

        len(loader)

    )

    return train_loss, epoch_time

# --------------------------------------------------------
# Validation
# --------------------------------------------------------

@torch.no_grad()
def validate(
        model,
        loader,
        criterion,
        metrics,
        prediction_visualizer,
        device,
        epoch
        ):
    """
    Валидация модели после эпохи обучения.

    Returns
    -------
    dict
        Все вычисленные метрики.
    """

    model.eval()

    metrics.reset()

    running_loss = 0.0

    saved_predictions = False

    progress = tqdm(
        loader,
        desc="Validation",
        leave=False
    )

    for images, masks in progress:

        images = images.to(device)

        masks = masks.to(device)

        outputs = model(images)

        loss = criterion(
            outputs,
            masks
        )

        running_loss += loss.item()

        metrics.update(
            outputs,
            masks
        )

        progress.set_postfix(

            loss=f"{loss.item():.4f}"

        )

        # --------------------------------------------
        # сохраняем несколько предсказаний
        # только один раз за эпоху
        # --------------------------------------------

        if not saved_predictions:

            prediction_visualizer.save_batch(

                images,

                masks,

                outputs,

                epoch,

                number=3

            )

            saved_predictions = True

    summary = metrics.summary()

    summary["val_loss"] = (

        running_loss /

        len(loader)

    )

    return summary

# --------------------------------------------------------
# Красивый вывод результатов эпохи
# --------------------------------------------------------

def print_epoch_summary(
        epoch,
        total_epochs,
        train_loss,
        results,
        learning_rate,
        epoch_time
        ):

    print()

    print("=" * 65)

    print(f"Epoch {epoch}/{total_epochs}")

    print("-" * 65)

    print(f"Train Loss      : {train_loss:.4f}")

    print(f"Validation Loss : {results['val_loss']:.4f}")

    print(f"Pixel Accuracy  : {results['pixel_accuracy']:.4f}")

    print(f"Mean IoU        : {results['mean_iou']:.4f}")

    print(f"Mean Dice       : {results['mean_dice']:.4f}")

    print(f"Learning Rate   : {learning_rate:.6f}")

    print(f"Epoch Time      : {epoch_time:.1f} sec")

    print("=" * 65)

    print()

# --------------------------------------------------------
# Main
# --------------------------------------------------------

def main():

    device = get_device()

    output_dir = create_experiment()

    logger, metrics, visualizer, prediction_visualizer = build_utils(
        output_dir
    )

    train_loader, val_loader = build_dataloaders()

    model = build_model(device)

    optimizer, scheduler = build_optimizer(model)

    criterion = build_loss()

    best_iou = 0.0

    patience_counter = 0

    history = {

        "epoch": [],

        "train_loss": [],

        "val_loss": [],

        "pixel_accuracy": [],

        "mean_iou": [],

        "mean_dice": [],

        "learning_rate": [],

        "epoch_time": []

    }

    print()

    print("Training started...")

    print()

    for epoch in range(

            1,

            config.NUM_EPOCHS + 1

    ):

        train_loss, epoch_time = train_one_epoch(

            model,

            train_loader,

            optimizer,

            criterion,

            metrics,

            device

        )

        results = validate(

            model,

            val_loader,

            criterion,

            metrics,

            prediction_visualizer,

            device,

            epoch

        )

        scheduler.step(

            results["mean_iou"]

        )

        lr = optimizer.param_groups[0]["lr"]
        history["epoch"].append(epoch)

        history["train_loss"].append(train_loss)

        history["val_loss"].append(

            results["val_loss"]

        )

        history["pixel_accuracy"].append(

            results["pixel_accuracy"]

        )

        history["mean_iou"].append(

            results["mean_iou"]

        )

        history["mean_dice"].append(

            results["mean_dice"]

        )

        history["learning_rate"].append(lr)

        history["epoch_time"].append(epoch_time)

        print_epoch_summary(

            epoch,

            config.NUM_EPOCHS,

            train_loss,

            results,

            lr,

            epoch_time

        )

        logger.log_epoch(
          epoch=epoch,
          train_loss=train_loss,
          val_loss=results["val_loss"],
          metrics=results,
          learning_rate=lr,
          epoch_time=epoch_time
        )

        if results["mean_iou"] > best_iou:

            best_iou = results["mean_iou"]

            patience_counter = 0

            torch.save(

                model.state_dict(),

                output_dir /

                "checkpoints" /

                "best_model.pth"

            )

            print(

                "Best model updated."

            )

        else:

            patience_counter += 1

            print(

                f"No improvement ({patience_counter})"

            )

        torch.save(

            model.state_dict(),

            output_dir /

            "checkpoints" /

            "last_model.pth"

        )

        if patience_counter >= config.PATIENCE:

            print()

            print(

                "Early stopping."

            )

            break

    logger.save_csv()

    logger.save_json()

    visualizer.build_all_plots(

        history,

        class_names=config.CLASS_NAMES,

        class_iou=results["iou"],

        class_dice=results["dice"]

    )

    print()

    print("=" * 65)

    print("Training completed.")

    print()

    print(

        f"Best Mean IoU : {best_iou:.4f}"

    )

    print()

    print(

        "Results saved to"

    )

    print(output_dir)

    print("=" * 65)

if __name__ == "__main__":

    main()