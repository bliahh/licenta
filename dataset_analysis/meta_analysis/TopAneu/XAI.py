import os
import pandas as pd
import shap
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


def shap_explanation(model, x, out_dir, name, class_index=None):
    contribs = model.get_booster().predict(xgb.DMatrix(x, enable_categorical=True), pred_contribs=True)
    if class_index is not None:
        contribs = contribs[:, class_index, :]

    data = x.copy()
    for c in data.select_dtypes("category"):
        data[c] = data[c].cat.codes
    sv = shap.Explanation(values=contribs[:, :-1],
                          base_values=contribs[:, -1],
                          data=data.values,
                          feature_names=list(x.columns))

    shap_plot_bar_values(sv, os.path.join(out_dir, f"shap_bar_{name}.png"))
    shap_plot_beeswarm(sv, os.path.join(out_dir, f"shap_beeswarm_{name}.png"))

    importance = pd.Series(abs(sv.values).mean(axis=0), index=x.columns).sort_values(ascending=False)
    return importance


def shap_plot_bar_values(sv, out_path):
    shap.plots.bar(sv, max_display=10, show=False)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close("all")


def shap_plot_beeswarm(sv, out_path):
    shap.plots.beeswarm(sv, max_display=10, show=False)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close("all")