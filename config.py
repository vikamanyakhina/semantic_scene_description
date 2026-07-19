"""
config.py

Конфигурационный файл проекта
Semantic Scene Description
"""

from pathlib import Path
import torch

# ==========================================================
# Пути
# ==========================================================

PROJECT_ROOT = Path(__file__).parent

DATASET_PATH = Path("/content/drive/MyDrive/MyProject/data/LoveDA_small")

OUTPUT_DIR = Path("/content/drive/MyDrive/MyProject/outputs")

# ==========================================================
# Эксперименты
# ==========================================================

EXPERIMENT_NAME = "RGB"

EXPERIMENTS = [
    ("RGB", None),
    ("RGB_LBP", "lbp"),
    ("RGB_ENTROPY", "entropy"),
    ("RGB_VARIANCE", "variance"),
    ("RGB_GABOR", "gabor"),
]

# ==========================================================
# Текстурные признаки
# ==========================================================

USE_TEXTURE = False

TEXTURE_TYPE = None

# ==========================================================
# Данные
# ==========================================================

IMAGE_SIZE = 512

NUM_CLASSES = 8

IN_CHANNELS = 3

CLASS_NAMES = [
    "Background",
    "Building",
    "Road",
    "Water",
    "Barren",
    "Forest",
    "Agricultural",
    "Unknown"
]

# ==========================================================
# Обучение
# ==========================================================

MODEL_NAME = "LightUNet"

BATCH_SIZE = 4

NUM_EPOCHS = 30

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 2

PATIENCE = 8

SAVE_BEST_ONLY = True

SEED = 42

# ==========================================================
# Устройство
# ==========================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# ==========================================================
# Визуализация
# ==========================================================

SAVE_PREDICTIONS = True

NUM_PREDICTIONS = 3

SAVE_PLOTS = True

# ==========================================================
# Оценка
# ==========================================================

SAVE_CONFUSION_MATRIX = True

SAVE_CLASS_METRICS = True

# ==========================================================
# Формирование описания сцены
# ==========================================================

MIN_OBJECT_AREA = 100

SAVE_JSON = True

SAVE_TXT = True

VERBOSE = True