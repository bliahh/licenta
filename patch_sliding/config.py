import os

from common.paths import PROJECT_DIR

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
RUNS_DIR = os.path.join(OUT_DIR, "runs")                       # runs/run_<date>/ : checkpoints, config.json, evaluation/
TABLES_DIR = os.path.join(OUT_DIR, "tables")

# image.pt, aneurysm_mask.pt, bifurcations.pt and metadata.pt for every patient
PREPROCESSED_DIR = os.path.join(PROJECT_DIR, "dataset", "preprocessed_dataset")

NUM_WORKERS = 2

# patch categories
POSITIVE = 1
WEAK_POSITIVE = -1      # ambiguous: part of an aneurysm, ignored by the loss
NEG_BIF = 2             # negative with at least PATCH["min_bif"] bifurcations
NEGATIVE = 0

# how volumes are cut into patches and how patches are labelled
PATCH = {
    "patch_size": (64, 64, 64),
    "overlap": 0.25,
    "min_bif": 10,
    "th_fraction": 0.0,
}

# sampler of the training loader: "hard" (HardPosNegSampler), "posneg" (PosNegSampler) or None (all patches, shuffled)
SAMPLER = {"name": "hard", "neg_ratio": 3, "hard_ratio": 0.5}

MODEL = {"norm": ("group", {"num_groups": 8}), "conv1_stride": 2}

# hyperparameters tried so far; train() uses ACTIVE_PRESET
PRESETS = {
    "lr1e-4_pw1.5": {"batch_size": 8, "epochs": 30, "lr": 1e-4, "weight_decay": 1e-4, "dropout": 0.3662004056693042, "pos_weight": 1.5, "patience": 10},
    "lr1.7e-5_pw6.8_bs1": {"batch_size": 1, "epochs": 50, "lr": 1.7432210823763044e-05, "weight_decay": 1.791471589306398e-05, "dropout": 0.3662004056693042, "pos_weight": 6.8, "patience": 10},
    "lr1.7e-5_pw6.8": {"batch_size": 8, "epochs": 50, "lr": 1.7432210823763044e-05, "weight_decay": 1.791471589306398e-05, "dropout": 0.3662004056693042, "pos_weight": 6.8, "patience": 10},
    "lr1.7e-5_wd1e-4_pw4": {"batch_size": 8, "epochs": 50, "lr": 1.7432210823763044e-05, "weight_decay": 1.01471589306398e-04, "dropout": 0.3662004056693042, "pos_weight": 4, "patience": 10},
}
ACTIVE_PRESET = "lr1.7e-5_pw6.8"
