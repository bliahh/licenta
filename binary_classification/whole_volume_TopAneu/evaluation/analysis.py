import json
import os
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

from common.paths import DEVICE
from common.data.patient_info import read_spacing, center_of, patient_groups
from common.evaluation.groups import group_metrics, print_groups
from common.evaluation.thresholds import youden_mr_threshold
from whole_volume.config import ARCHITECTURES, OUT_DIR
from whole_volume.data.loaders import volume_loaders, volume_test_loader
from whole_volume.models.networks import load_model
from whole_volume.models.run import predict
from whole_volume.models.tuning.hyperparams import find_threshold
from whole_volume.utils.helpers import get_model_dir, get_results_path


def evaluate_volume_models(loss_name, architectures=ARCHITECTURES, batch_size=2):
    """
    For every whole-volume architecture trained with loss_name: thresholds chosen on VAL (0.5, F1, F2, Youden on MR),
    metrics on TEST for every patient group. Saves per-patient scores and comparison_<loss>_TEST_GROUPS.csv / .json.
    """
    print("\n" + "=" * 70)
    print(f"WHOLE-VOLUME EVALUATION PER GROUP - {loss_name.upper()}")
    print("=" * 70)

    params_path = get_results_path(loss_name)
    if not os.path.exists(params_path):
        print(f"[ERROR] Results file not found: {params_path}")
        return {}
    with open(params_path, "r") as f:
        original_results = json.load(f)

    os.makedirs(OUT_DIR, exist_ok=True)
    json_path = os.path.join(OUT_DIR, f"comparison_{loss_name}_TEST_GROUPS.json")
    csv_path = os.path.join(OUT_DIR, f"comparison_{loss_name}_TEST_GROUPS.csv")

    val_loader, test_loader = volume_loaders(batch_size=batch_size)
    val_samples, test_samples = val_loader.dataset.data, test_loader.dataset.data
    test_images, test_acq = [s["image"] for s in test_samples], [s["type_acq"] for s in test_samples]
    test_thick = [max(read_spacing(p)[0]) for p in test_images]
    groups = patient_groups(test_images, test_acq, test_thick)
    print("[GROUPS] " + " | ".join(f"{g}={int(m.sum())}" for g, m in groups.items()))

    results, rows = {}, []
    for name in architectures:
        print("\n" + "=" * 70)
        print(f"EVALUATING {name} / {loss_name}")
        print("=" * 70)

        save_path = os.path.join(get_model_dir(loss_name), f"{name}.pth")
        if not os.path.exists(save_path) or "best_params" not in original_results.get(name, {}):
            print(f"[MISSING] model or best_params for {name}")
            continue
        params = original_results[name]["best_params"]

        threshold_f1, _ = find_threshold(name, loss_name, params, val_loader, beta=1)
        threshold_f2, _ = find_threshold(name, loss_name, params, val_loader, beta=2)

        model = load_model(name, params, save_path, DEVICE)
        val_probs, val_labels = predict(model, val_loader, DEVICE)
        test_probs, test_labels = predict(model, test_loader, DEVICE)
        assert all(int(s["label"]) == l for s, l in zip(test_samples, test_labels)), "test loader order differs from samples"
        threshold_youden = youden_mr_threshold(val_probs, val_labels, [s["type_acq"] for s in val_samples])
        pd.DataFrame({"image": test_images, "label": test_labels, "type_acq": test_acq, "center": [center_of(p) for p in test_images], "thickest": test_thick, "score": test_probs}).to_csv(os.path.join(OUT_DIR, f"scores_test_{name}_{loss_name}.csv"), index=False)

        results[name] = {}
        for threshold_name, threshold in (("normal", 0.5), ("F1opt", threshold_f1), ("F2opt", threshold_f2), ("youden_mr", threshold_youden)):
            print(f"\n--- {name}/{loss_name} [{threshold_name}] threshold = {threshold:.4f} ---")
            gm = group_metrics(test_probs, test_labels, groups, threshold)
            print_groups(gm)
            results[name][threshold_name] = {"threshold": threshold, "best_params": params, "groups": gm}
            rows += [{"model": name, "loss": loss_name, "threshold_name": threshold_name, "threshold": threshold, "group": g, **r} for g, r in gm.items()]

        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=float)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"[DONE] {name}")

    print(f"\n[SAVED]\n{json_path}\n{csv_path}")
    return results


def thick_slice_experiment(loss_name, name, thick_mm=0.7, threshold=0.5, batch_size=2):
    """
    Scores of the same TEST patients with the original slices and with slices simulated at thick_mm.
    Saves one row per patient and prints, for the thin MR patients, the score change per label and the AUC before / after.
    """
    with open(get_results_path(loss_name), "r") as f:
        params = json.load(f)[name]["best_params"]
    model = load_model(name, params, os.path.join(get_model_dir(loss_name), f"{name}.pth"), DEVICE)

    test_loader, thick_loader = volume_test_loader(batch_size=batch_size), volume_test_loader(thick_mm=thick_mm, batch_size=batch_size)
    samples = test_loader.dataset.data
    images, acq = [s["image"] for s in samples], [s["type_acq"] for s in samples]
    thickest = [max(read_spacing(p)[0]) for p in images]

    print(f"[1/2] original slices, {len(samples)} test patients...")
    probs, labels = predict(model, test_loader, DEVICE)
    print(f"[2/2] slices simulated at {thick_mm} mm...")
    probs_thick, _ = predict(model, thick_loader, DEVICE)
    df = pd.DataFrame({"image": images, "label": labels, "type_acq": acq, "center": [center_of(p) for p in images], "thickest": thickest, "score": probs, "score_thick": probs_thick})
    df["delta"] = df["score_thick"] - df["score"]
    out_csv = os.path.join(OUT_DIR, f"thick_slice_{name}_{loss_name}_{thick_mm}.csv")
    df.to_csv(out_csv, index=False)

    thin = df[patient_groups(images, acq, thickest)["mr_thin"]]
    print(f"\n=== THICK SLICE EXPERIMENT {name}/{loss_name} | simulated {thick_mm} mm | thin MR test patients: {len(thin)} ===")
    for label, label_name in ((0, "healthy"), (1, "aneurysm")):
        g = thin[thin["label"] == label]
        p_value = wilcoxon(g["delta"]).pvalue if len(g) >= 6 and (g["delta"] != 0).any() else float("nan")
        print(f"  [{label_name:<8}] n={len(g):3d} | score {g['score'].mean():.3f} -> {g['score_thick'].mean():.3f} | mean delta {g['delta'].mean():+.3f} | went up {(g['delta'] > 0).sum()}/{len(g)} | above {threshold}: {(g['score'] >= threshold).sum()} -> {(g['score_thick'] >= threshold).sum()} | Wilcoxon p={p_value:.4f}")
    if thin["label"].nunique() == 2:
        print(f"  AUC mr_thin: original {roc_auc_score(thin['label'], thin['score']):.3f} | simulated {roc_auc_score(thin['label'], thin['score_thick']):.3f}")
    print(f"[SAVED] {out_csv}")
    return df
