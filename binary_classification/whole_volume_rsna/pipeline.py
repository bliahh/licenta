import os

import pandas as pd

from binary_classification.whole_volume_rsna.models.test import test
from binary_classification.whole_volume_rsna.models.train import train, DEVICE, USE_AMP

SEEDS = (0, 1, 2)
RUNS_DIR = "out"


def run_experiment(name, only_mr=False, only_ct=False, seeds=SEEDS, **train_options):
    results = []
    for seed in seeds:
        run_dir = os.path.join(RUNS_DIR, f"{name}_seed{seed}")
        train(seed, only_mr=only_mr, only_ct=only_ct, run_dir=run_dir, **train_options)
        metrics = test(run_dir, seed, only_mr=only_mr, only_ct=only_ct, device=DEVICE, use_amp=USE_AMP)
        results.append({"seed": seed, **metrics})

    results = pd.DataFrame(results)
    results.to_csv(os.path.join(RUNS_DIR, f"{name}_all_seeds.csv"), index=False)
    print(f"[ {name} ] mean and stddev {len(seeds)} seeds")
    print(results.drop(columns="seed").agg(["mean", "std"]).round(3))
    return results


def run_only_mra_roi96_lambd0():
    return run_experiment("mra_roi96_lambd0", only_mr=True, seeds=(0,), multitask=True, roi_size=96, lambd=0.0)


def run_only_mra_roi96_heat():
    return run_experiment("mra_roi96_heat_modif", only_mr=True, seeds=(0,), multitask=True, roi_size=96,
                          roi_jitter=12, pos_weight=10.0, lambd=2.0, sigma=0.7)


def run_only_mra_05mm():
    return run_experiment("mra_05mm_p30", only_mr=True, seeds=(0,), multitask=True, roi_size=None, patience=30)


def run_more():
    run_experiment("mra_roi96_jitter_p30", only_mr=True, seeds=(0,), multitask=True, roi_size=96,
                   roi_jitter=12, patience=30)
    run_experiment("mra_roi96_p30", only_mr=True, seeds=(0,), multitask=True, roi_size=96,
                   patience=30)
    run_experiment("mra_roi96_jitter_lr3e-4_p30", only_mr=True, seeds=(0,), multitask=True, roi_size=96,
                   roi_jitter=12, lr=3e-4, patience=30)

PRETRAINED_PATH = os.path.abspath("pretrained/resnet_18_23dataset.pth")

def run_only_mra_pretrained():
    return run_experiment("mra_roi96_pretrained_p30", only_mr=True, seeds=(0,), multitask=True, roi_size=96,
                              patience=30, pretrained_path=PRETRAINED_PATH)