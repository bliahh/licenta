import os

import pandas as pd
from sklearn.metrics import roc_auc_score

from common.data.labels import build_bin_samples
from common.paths import DATASET_ROOT
from dataset_analysis.meta_analysis.TopAneu.metadata_analysis import build_dataset_table, attach_descrip_info
from dataset_analysis.meta_analysis.TopAneu.model import meta_build_train_test_split, train_xgboost, test_classifier, map_center_labels, test_classifier_multiclass

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../out")
TABLE_CSV = os.path.join(OUT_DIR, "metadata_topaneu.csv")
DESCRIP_CSV = os.path.join(OUT_DIR, "descrip_topaneu.csv")
INTENSITY_CSV = os.path.join(OUT_DIR, "intensity.csv")


CENTER_LABELS = {1: 0, 2: 1, 4: 2, 5: 3}
CENTER_NAMES = ["center1", "center2", "center4", "center5"]
CONVERSION_COLUMNS = ["datatype", "dim_info", "qform_code", "sform_code", "has_scaling", "xyzt_units", "has_descrip"]
NOT_FEATURES = ["file", "patient", "scan", "label"]



def load_or_build_table(csv_path=TABLE_CSV):
    if os.path.exists(csv_path):
        table = pd.read_csv(csv_path)
        for column in ["modality", "datatype"]:
            table[column] = table[column].astype("category")
        return table
    samples = build_bin_samples()
    table = build_dataset_table(samples, include_center=True, include_intensity=True, intensity_csv=INTENSITY_CSV)
    table.to_csv(csv_path, index=False)
    return table


def check_descrip(csv_path=DESCRIP_CSV):
    files = sorted(f for f in os.listdir(os.path.join(DATASET_ROOT, "images")) if f.endswith((".nii", ".nii.gz")))
    descrip_table = attach_descrip_info(files)
    descrip_table.to_csv(csv_path, index=False)
    not_empty = descrip_table[descrip_table["descrip"] != ""]
    print(not_empty.groupby("center").head(3).to_string())


SEEDS = [0, 1, 2, 3, 4]
INTENSITY_COLUMNS = ["int_min", "int_max", "int_mean", "int_std", "int_p1", "int_p50", "int_p99", "int_range", "frac_min", "has_negative", "n_unique"]




def run_meta_analysis_without_center(table):
    center1 = table[table["center"] == 1]
    results = []
    for seed in SEEDS:
        results.append(run_xgboost(table, "tot setul cu metadate", seed=seed, drop_columns=INTENSITY_COLUMNS))
        results.append(run_xgboost(center1, "center1 cu metadate", seed=seed, drop_columns=INTENSITY_COLUMNS))
        results.append(run_xgboost(table, "tot setul cu metadate + intensitate", seed=seed))
        results.append(run_xgboost(center1, "center1 cu metadate + intensitate", seed=seed))

    all_runs = pd.DataFrame(results).drop(columns=["model"])
    all_runs.to_csv(os.path.join(OUT_DIR, "binary_classif_no_center_all_runs.csv"), index=False)
    summary = all_runs.groupby("experiment")["auc_test"].agg(["mean", "std"]).round(3)
    summary.to_csv(os.path.join(OUT_DIR, "binary_classif_no_center.csv"))
    print("\n", summary.to_string())
    return results


def run_xgboost(table, name, include_center=False, seed=42, drop_columns=(), multiclass=False):
    table = table.drop(columns=[c for c in drop_columns if c in table])
    x_train, y_train, x_test, y_test = meta_build_train_test_split(table, include_center, seed)
    model, best_params = train_xgboost(x_train, y_train, multiclass)
    if multiclass:
        p, balanced_acc = test_classifier_multiclass(model, x_test, y_test, CENTER_NAMES)
        auc = roc_auc_score(y_test, p, multi_class="ovr")
    else:
        p = test_classifier(model, x_test, y_test)
        auc, balanced_acc = roc_auc_score(y_test, p), None
    return {"experiment": name, "seed": seed, "n": len(table), "n_test": len(y_test),
            "auc_test": auc, "balanced_acc": balanced_acc, "model": model}


def run_meta_analysis_classify_centers(table):
    table = table.copy()
    table["label"] = map_center_labels(table)
    table = table.drop(columns=["center"])
    not_conversion = [c for c in table.columns if c not in CONVERSION_COLUMNS + NOT_FEATURES]

    results = []
    for seed in SEEDS:
        results.append(run_xgboost(table, "centre  cu metadate", seed=seed, drop_columns=INTENSITY_COLUMNS, multiclass=True))
        results.append(run_xgboost(table, "centre  doar cu conversie", seed=seed, drop_columns=not_conversion, multiclass=True))
        results.append(run_xgboost(table, "centre cu metadate + intensitate", seed=seed, multiclass=True))

    all_runs = pd.DataFrame(results).drop(columns=["model"])
    all_runs.to_csv(os.path.join(OUT_DIR, "classif_centers_all_runs.csv"), index=False)
    summary = all_runs.groupby("experiment")[["auc_test", "balanced_acc"]].agg(["mean", "std"]).round(3)
    summary.to_csv(os.path.join(OUT_DIR, "classif_centers_summary.csv"))
    print("\n", summary.to_string())
    return results




def main():
    table = load_or_build_table()
    print(table.shape)
    print(pd.crosstab(table["center"], table["label"]))


    results_without_center = run_meta_analysis_without_center(table)
    results_center_classifier =run_meta_analysis_classify_centers(table)


if __name__ == "__main__":
    main()