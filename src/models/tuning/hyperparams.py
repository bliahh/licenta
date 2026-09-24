import json
import os
import numpy as np
import optuna
import torch
from sklearn.metrics import fbeta_score
from src.config import N_TRIALS, DEVICE, ARCHITECTURES, OUTPUT_DIR
from src.models.networks import build_model
from src.models.tuning.optuna_search import objective
from src.data.loaders import build_loaders
from src.utils.helpers import get_tuned_eval_dir, get_tuned_results_path, get_model_dir


def optimize(name, loss_name, train_loader, val_loader):
    study_name = f"{name}_{loss_name}"
    storage = f"sqlite:///optuna_{name}_{loss_name}.db"

    print("\n" + "<>" * 60)
    print(f"<> OPTUNA {name} / {loss_name}")
    print("<>" * 60)

    study = optuna.create_study(
        study_name=study_name, storage=storage, load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5))

    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, N_TRIALS - done)

    if remaining > 0:
        print(f"[{study_name}] {done}/{N_TRIALS} completed.")
        study.optimize(lambda trial: objective(trial, name, train_loader, val_loader, loss_name),
                       n_trials=remaining)
    else:
        print(f"[{study_name}] already complete.")

    if len(study.trials) == 0:
        raise RuntimeError(f"No Optuna trials for {study_name}")

    print(f"[{study_name}] best AUC={study.best_value:.4f}")
    print(f"[{study_name}] params={study.best_params}")
    return study.best_params


def find_threshold(name, loss_name, params, val_loader, beta=1):
    """
    find the best threshold to maximize F_beta score for a given loss name
    :param name: the name of the arhitecture
    :param loss_name: name of the loss function
    :param params: the optimization parameters
    :param val_loader: validation loader
    :param beta: 1 for F1, 2 for F2
    :return: the best threshold computed in order to maximize F_beta score
    """
    save_path = os.path.join(get_model_dir(loss_name), f"{name}.pth")
    model = build_model(name, params["dropout"]).to(DEVICE)
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            x = batch["image"].to(DEVICE)
            y = batch["label"].to(DEVICE)
            logits = model(x)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())

    probs = np.concatenate(all_probs).ravel()
    labels = np.concatenate(all_labels).ravel()

    print(f"\n[INFO] Searching threshold for {name}/{loss_name}")
    best_thresh, best_metric = 0.5, -1.0
    for threshold in np.arange(0.0, 1.0, 0.001):
        preds = (probs >= threshold).astype(int)
        score = fbeta_score(labels, preds, beta=beta, zero_division=0)
        if score > best_metric:
            best_metric = score
            best_thresh = float(threshold)

    print(f"[THRESHOLD {name}/{loss_name}] best={best_thresh:.3f} | F{beta}={best_metric:.3f}")
    return best_thresh, best_metric


def compute_all_thresh(loss_name, results, beta):
    """
    compute all thresholds for all the arhitectures given in ARHITECTURES config file
    :param loss_name: name of the loss function
    :param results:
    :param beta:
    :return:
    """
    from src.evaluation.evaluator import evaluate

    tuned = {}
    for name in ARCHITECTURES:
        if name not in results:
            continue
        params = results[name]["best_params"]
        save_path = os.path.join(get_model_dir(loss_name), f"{name}.pth")
        if not os.path.exists(save_path):
            continue

        _, val_loader, test_loader = build_loaders(name)
        best_thresh, val_fbeta = find_threshold(name, loss_name, params, val_loader, beta=beta)
        test_metrics = evaluate(name, loss_name, save_path, params, test_loader, threshold=best_thresh, eval_dir=get_tuned_eval_dir(loss_name))

        tuned[name] = {
            "loss": loss_name, "best_params": params, "threshold": best_thresh,
            "val_fbeta": val_fbeta, "val_auc": results[name]["val_auc"],
            "test_auc": test_metrics["auc"], "test_pr_auc": test_metrics["pr_auc"],
            "test_recall": test_metrics["recall"], "test_precision": test_metrics["precision"],
            "test_f1": test_metrics["f1"], "test_acc": test_metrics["acc"]}

    path = get_tuned_results_path(loss_name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(tuned, f, indent=2)
    return tuned


