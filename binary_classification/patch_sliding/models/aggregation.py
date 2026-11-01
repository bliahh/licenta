from functools import partial

import numpy as np


def aggregate_patches(probs_patch, patient_labels, all_sample_index, fn):
    """Groups the patch probabilities by patient and reduces them with fn; patients are returned sorted by index."""
    patient_proba = {}
    patient_label = {}
    for proba, label, sample_index in zip(probs_patch, patient_labels, all_sample_index):
        patient_proba.setdefault(sample_index, []).append(proba)
        patient_label[sample_index] = label

    patient_proba = {index: fn(probas) for index, probas in patient_proba.items()}
    patient_idx = sorted(patient_proba.keys())
    probs = np.array([patient_proba[idx] for idx in patient_idx])
    labels = np.array([patient_label[idx] for idx in patient_idx])
    return probs, labels


def arg_max_prob(probas):
    return float(np.max(probas))

def top_k_prob(probas,k):
    return float(np.average(sorted(probas,reverse=True)[:k]))

def mean_prob(probas):
    return float(np.average(probas))

def median_prob(probas):
    return float(np.median(probas))

def k_percentile_prob(probas,k):
    return float(np.quantile(probas,q=k))


AGGREGATORS = {"max": arg_max_prob, "top2": partial(top_k_prob, k=2), "top3": partial(top_k_prob, k=3),
               "top5": partial(top_k_prob, k=5), "top7": partial(top_k_prob, k=7),
               "top10": partial(top_k_prob, k=10), "p90": partial(k_percentile_prob, k=0.9),
               "p75": partial(k_percentile_prob, k=0.75), "median": median_prob, "mean": mean_prob}
