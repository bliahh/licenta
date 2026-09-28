import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

SIZE = (128, 128, 128)

LABELS_DIR = os.path.join(
    PROJECT_DIR,
    "dataset",
    "dataset_labels"
)

BINARY_LABELS_PATH = os.path.join(
    LABELS_DIR,
    "binary_labels.json"
)

SPLIT_PATH = os.path.join(
    LABELS_DIR,
    "dataset_split.json"
)

DATASET_SPLIT_PATH = "dataset_split.json"
DATASET_LABELS_PATH = LABELS_DIR

DATASET_ROOT = os.path.join(
    PROJECT_DIR,
    "dataset",
    "topaneu_release"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARCHITECTURES = [
    "resnet10",
    "resnet18",
    "resnet34",
    "resnet50",
    "densenet121",
    "densenet169",
    "efficientnet-b0",
    "efficientnet-b1"
]

BATCH_PER_ARCH = {
    "resnet10": 2,
    "resnet18": 2,
    "resnet34": 2,
    "densenet121": 1,
    "densenet169": 1,
    "resnet50": 1,
    "efficientnet-b0": 1,
    "efficientnet-b1": 1,
}

LOSS_NAME = "both"

N_POS_TRAIN = 194
N_NEG_TRAIN = 71
N_TRIALS = 20
FINAL_EPOCHS = 60
PATIENCE = 10
BATCH_SIZE = 2
NUM_WORKERS = 2
BETA = 2

OUTPUT_DIR = os.path.join(PROJECT_DIR, "output", "outputs_vesselmask")
SAVE_DIR = os.path.join(OUTPUT_DIR, "models")
TUNED_EVAL_DIR = os.path.join(OUTPUT_DIR, "evaluation_tuned")

RESULTS_PATH = os.path.join(OUTPUT_DIR, "comparison_results.json")
TUNED_RESULTS_PATH = os.path.join(OUTPUT_DIR, "comparison_results_tuned.json")

OUTPUT_SAVE_PATH = os.path.join(PROJECT_DIR,"public_out", "sliding")

OUTPUT_SAVE_DIR = os.path.join(PROJECT_DIR,"public_out","sliding_with_batch_norm8")
BEST_MODEL_PATH = os.path.join(OUTPUT_SAVE_DIR,"best_model.pth")
EVAL_DIR = os.path.join(OUTPUT_SAVE_DIR,"evaluation")
PREPROCESSED_DIR = os.path.join(PROJECT_DIR,"dataset","preprocessed_dataset")


POSITIVE = 1
WEAK_POSITIVE = -1
NEG_BIF = 2
NEGATIVE = 0
