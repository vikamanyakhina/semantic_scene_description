"""
config.py

Единый конфигурационный файл проекта.
Изменяя параметры здесь, можно запускать разные эксперименты
без изменения остального кода.
"""

import torch
from pathlib import Path


# ============================================================
# Пути
# ============================================================

DATASET_PATH = Path("data/LoveDA_small")

OUTPUT_DIR = Path("outputs")

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

RESULTS_DIR = OUTPUT_DIR / "results"

JSON_DIR = OUTPUT_DIR / "json"

TXT_DIR = OUTPUT_DIR / "txt"


# ============================================================
# Данные
# ============================================================

IMAGE_SIZE = 512

NUM_CLASSES = 8

TEXTURE = None
# None
# "lbp"
# "entropy"
# "variance"
# "gabor"


# ============================================================
# Обучение
# ============================================================

MODEL_NAME = "light_unet"

SAVE_BEST_ONLY = True

# ============================================================
# Устройство
# ============================================================


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

SEED = 42

BATCH_SIZE = 4

NUM_EPOCHS = 3

LEARNING_RATE = 1e-4

NUM_WORKERS = 0

PATIENCE = 8

EXPERIMENTS = [
    ("RGB", None),
    ("RGB_LBP", "lbp"),
    ("RGB_ENTROPY", "entropy"),
    ("RGB_VARIANCE", "variance"),
    ("RGB_GABOR", "gabor"),
]
