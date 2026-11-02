import torch

from binary_classification.whole_volume_rsna.models.network import ResNet18MultiTask
from binary_classification.whole_volume_rsna.models.train import train
from binary_classification.whole_volume_rsna.pipeline import run_only_mra, run_only_mra_roi96, run_only_mra_roi96_lambd0

if __name__ == "__main__":
     run_only_mra_roi96_lambd0()

