
import os
import json

from src.config import ARCHITECTURES, OUTPUT_DIR
from src.data.loaders import build_loaders
from src.evaluation.evaluator import  evaluate_models
from src.utils.helpers import (
    get_model_dir,
    get_eval_dir,
    get_results_path,
)



def main():

    print("\n" + "=" * 70)
    print("BINARY CLASSIFICATION")
    print("TEST + THRESHOLD TUNING")
    print("=" * 70)


    focal_results = evaluate_models("focal")


    bce_results = evaluate_models("bce")


    all_results = {
        "focal": focal_results,
        "bce": bce_results,
    }

    all_results_path = os.path.join(OUTPUT_DIR,"comparison_results_TEST_TUNED_ALL.json")

    with open(all_results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print("ALL EVALUATIONS FINISHED")
    print("=" * 70)

    print("\nAll results saved to:")
    print(all_results_path)


if __name__ == "__main__":
    main()

