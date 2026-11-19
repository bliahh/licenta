import torch

from binary_classification.whole_volume_rsna.models.network import ResNet18MultiTask
from binary_classification.whole_volume_rsna.models.train import train
from binary_classification.whole_volume_rsna.pipeline import run_only_mra_roi96_lambd0, run_only_mra_roi96_heat


if __name__ == "__main__":
     from binary_classification.whole_volume_rsna.data.loaders import DATA_DIR
     from binary_classification.whole_volume_rsna.pipeline import run_only_mra_05mm

     print("[ DATA ]", DATA_DIR)
     run_only_mra_05mm()
