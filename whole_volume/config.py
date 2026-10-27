import os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
MODELS_DIR = os.path.join(OUT_DIR, "models")                   # models/<loss>/<arch>.pth
EVAL_DIR = os.path.join(OUT_DIR, "evaluation")                 # evaluation/<loss>/<arch>/
TUNED_EVAL_DIR = os.path.join(OUT_DIR, "evaluation_tuned")
OPTUNA_DIR = os.path.join(OUT_DIR, "optuna")                   # optuna_<arch>_<loss>.db + trial checkpoints
FN_FP_LOG = os.path.join(OUT_DIR, "fn_fp.txt")

# volumes are padded / cropped to SIZE after the vessel crop
SIZE = (128, 128, 128)

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

# the large architectures run at batch 1 so they do not go OOM
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

# "bce", "focal" or "both"
LOSS_NAME = "both"

N_TRIALS = 20
TRIAL_EPOCHS = 20
FINAL_EPOCHS = 60
PATIENCE = 10
NUM_WORKERS = 2
CACHE_RATE = 0.3
BETA = 2
