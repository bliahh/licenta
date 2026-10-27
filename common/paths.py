import os
import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_ROOT = os.path.join(PROJECT_DIR, "dataset", "topaneu_release")
LABELS_DIR = os.path.join(PROJECT_DIR, "dataset", "dataset_labels")
BINARY_LABELS_PATH = os.path.join(LABELS_DIR, "binary_labels.json")
SPLIT_PATH = os.path.join(LABELS_DIR, "dataset_split.json")

# the same split is used by every project
SPLIT = {"val_ratio": 0.16, "test_ratio": 0.20, "seed": 42}

# patient table (spacing, center, modality) produced by dataset_analysis and read by the evaluation of both models
PATIENTS_CSV = os.path.join(PROJECT_DIR, "dataset_analysis", "out", "patients.csv")

# MR volumes with the thickest spacing <= THIN_MM are "thin"
THIN_MM = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
