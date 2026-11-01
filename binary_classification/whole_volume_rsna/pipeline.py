import os

import pandas as pd
import torch

from binary_classification.whole_volume_rsna.models.test import test
from binary_classification.whole_volume_rsna.models.train import train, DEVICE

SEEDS = (0, 1, 2)
RUNS_DIR = "out"
EVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"


def run_experiment(name, only_mr=False, only_ct=False, seeds=SEEDS):
    results = []
    for seed in seeds:
        run_dir = os.path.join(RUNS_DIR, f"{name}_seed{seed}")
        train(seed, only_mr=only_mr, only_ct=only_ct, run_dir=run_dir)
        metrics = test(run_dir, seed, only_mr=only_mr, only_ct=only_ct,device=DEVICE, use_amp=USE_AMP)
        results.append(metrics)
        results.append({"seed": seed, **metrics})

    results = pd.DataFrame(results)
    results.to_csv(os.path.join(RUNS_DIR, f"{name}_all_seeds.csv"), index=False)
    print(f"[ {name} ] mean and stddev {len(seeds)} seeds")
    print(results.drop(columns="seed").agg(["mean", "std"]).round(3))
    return results


def run_only_cta():
    return run_experiment("cta", only_ct=True)


def run_only_mra():
    return run_experiment("mra", only_mr=True)


def run_joint():
    return run_experiment("combined")

