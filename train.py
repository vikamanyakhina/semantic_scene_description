"""
train.py

Обучение модели U-Net для семантической сегментации.
"""

from pathlib import Path

import random
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from datasets.loveda_dataset import LoveDADataset
from models.unet import UNet
from utils.losses import get_loss

import config


# ==========================================================
# Фиксация генераторов случайных чисел
# ==========================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================================
# Один проход обучения
# ==========================================================

def train_one_epoch(model,
                    loader,
                    optimizer,
                    criterion,
                    device):

    model.train()

    running_loss = 0

    progress = tqdm(loader)

    for images, masks in progress:

        images = images.to(device)

        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, masks)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        progress.set_description(

            f"Loss: {loss.item():.4f}"

        )

    return running_loss / len(loader)


# ==========================================================
# Валидация
# ==========================================================

@torch.no_grad()

def validate(model,
             loader,
             criterion,
             device):

    model.eval()

    running_loss = 0

    for images, masks in loader:

        images = images.to(device)

        masks = masks.to(device)

        outputs = model(images)

        loss = criterion(outputs, masks)

        running_loss += loss.item()

    return running_loss / len(loader)


# ==========================================================
# Основная функция
# ==========================================================

def main():

    set_seed(config.SEED)

    device = torch.device(config.DEVICE)

    print(f"Device: {device}")

    # ------------------------------------------------------

    train_dataset = LoveDADataset(

        root_dir=config.DATASET_PATH,

        split="Train",

        texture=config.TEXTURE

    )

    val_dataset = LoveDADataset(

        root_dir=config.DATASET_PATH,

        split="Val",

        texture=config.TEXTURE

    )

    # ------------------------------------------------------

    train_loader = DataLoader(

        train_dataset,

        batch_size=config.BATCH_SIZE,

        shuffle=True,

        num_workers=config.NUM_WORKERS

    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=config.BATCH_SIZE,

        shuffle=False,

        num_workers=config.NUM_WORKERS

    )

    # ------------------------------------------------------

    in_channels = 3

    if config.TEXTURE is not None:

        in_channels = 4

    # ------------------------------------------------------

    model = UNet(

        in_channels=in_channels,

        num_classes=config.NUM_CLASSES

    )

    model.to(device)

    # ------------------------------------------------------

    criterion = get_loss()

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=config.LEARNING_RATE

    )

    # ------------------------------------------------------

    Path(config.CHECKPOINT_DIR).mkdir(

        parents=True,

        exist_ok=True

    )

    # ------------------------------------------------------

    best_loss = float("inf")

    # ------------------------------------------------------

    for epoch in range(config.NUM_EPOCHS):

        print()

        print("=" * 50)

        print(

            f"Epoch {epoch+1}/{config.NUM_EPOCHS}"

        )

        train_loss = train_one_epoch(

            model,

            train_loader,

            optimizer,

            criterion,

            device

        )

        val_loss = validate(

            model,

            val_loader,

            criterion,

            device

        )

        print(

            f"Train Loss: {train_loss:.5f}"

        )

        print(

            f"Val Loss:   {val_loss:.5f}"

        )

        if val_loss < best_loss:

            best_loss = val_loss

            torch.save(

                model.state_dict(),

                Path(config.CHECKPOINT_DIR) /

                "best_model.pth"

            )

            print("Best model saved.")

    print()

    print("Training finished.")


if __name__ == "__main__":

    main()