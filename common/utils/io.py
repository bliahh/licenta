import os
import json


def save_labeled_samples(labeled_samples, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(labeled_samples, f, indent=2)

    return True


def save_metrics(metrics, dir, filename="metrics.json"):
    os.makedirs(dir, exist_ok=True)
    with open(os.path.join(dir, filename), "w") as f:
        json.dump(metrics, f, indent=2)


def save_in_file(string, filename):
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    with open(filename, "a") as f:
        f.write(string)
