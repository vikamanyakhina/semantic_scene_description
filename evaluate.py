"""
evaluate.py

Оценка качества модели семантической сегментации.
"""

from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config

from dataset.loveda_dataset import LoveDADataset

from models.unet import LightUNet

from models.losses import CombinedLoss

from utils.metrics import SegmentationMetrics

from utils.prediction_visualizer import PredictionVisualizer

def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

def load_model(device):

    model = LightUNet(

        in_channels=config.IN_CHANNELS,

        num_classes=config.NUM_CLASSES

    )

    checkpoint = (

        Path(config.OUTPUT_DIR)

        / config.EXPERIMENT_NAME

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

def build_loader():

    dataset = LoveDADataset(

        split="Val",

        use_texture=config.USE_TEXTURE,

        texture_type=config.TEXTURE_TYPE

    )

    loader = DataLoader(

        dataset,

        batch_size=config.BATCH_SIZE,

        shuffle=False,

        num_workers=config.NUM_WORKERS,

        pin_memory=True

    )

    return loader

