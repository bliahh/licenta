import torch
from torch.utils.data import Sampler

from patch_sliding.config import POSITIVE, NEG_BIF, NEGATIVE


class PosNegSampler(Sampler):
    """
    Sampler that balances positive and negative patches during training.

    At every epoch it takes all positive patches and a new random subset of
    negatives (neg_ratio negatives per positive), then shuffles them together.
    Over several epochs all negatives are eventually seen.

    :param labels: list of patch labels (1 = positive, 0 = negative; other values are ignored)
    :param neg_ratio: number of negative patches sampled per positive patch
    """

    def __init__(self, labels, neg_ratio=3):
        # indices of positive and negative patches
        self.pos = [i for i, l in enumerate(labels) if l == 1]
        self.neg = [i for i, l in enumerate(labels) if l == 0]
        # number of negatives per epoch (capped at the available negatives)
        self.n_neg = min(len(self.neg), neg_ratio * len(self.pos))

    def __iter__(self):
        # new random subset of negatives for this epoch
        neg = torch.randperm(len(self.neg))[:self.n_neg].tolist()
        idx = self.pos + [self.neg[i] for i in neg]
        # shuffle positives and negatives together
        return iter([idx[i] for i in torch.randperm(len(idx)).tolist()])

    def __len__(self):
        # patches per epoch
        return len(self.pos) + self.n_neg


class HardPosNegSampler(Sampler):
    """
    Like PosNegSampler, but a fraction hard_ratio of the negatives drawn every epoch
    comes from the hard negatives (category NEG_BIF: negative patches with many bifurcations).

    :param labels: list of patch categories (1, -1, 2, 0)
    :param neg_ratio: number of negative patches (hard + easy) sampled per positive patch
    :param hard_ratio: fraction of the sampled negatives that are hard negatives
    """

    def __init__(self, labels, neg_ratio=3,hard_ratio=0.5):
        self.hard_ratio = hard_ratio

        self.hard_neg = [i for i, l in enumerate(labels) if l == NEG_BIF]
        self.neg = [i for i, l in enumerate(labels) if l == NEGATIVE]
        self.pos = [i for i, l in enumerate(labels) if l == POSITIVE]

        n_total_neg = min(len(self.neg) + len(self.hard_neg), int(neg_ratio * len(self.pos)))

        self.n_h_neg = min(len(self.hard_neg), int(round(n_total_neg * hard_ratio)))
        self.n_neg = min(len(self.neg), n_total_neg - self.n_h_neg)

    def __iter__(self):
        h_neg = torch.randperm(len(self.hard_neg))[:self.n_h_neg].tolist()
        neg = torch.randperm(len(self.neg))[:self.n_neg].tolist()
        idx = self.pos + [self.hard_neg[i] for i in h_neg] + [self.neg[i] for i in neg]

        return iter([idx[i] for i in torch.randperm(len(idx)).tolist()])

    def __len__(self):
        return len(self.pos) + self.n_h_neg + self.n_neg


def build_sampler(labels, name, neg_ratio=3, hard_ratio=0.5):
    """Sampler selected by name: "hard", "posneg" or None (no sampler)."""
    if name is None:
        return None
    if name == "hard":
        return HardPosNegSampler(labels=labels, neg_ratio=neg_ratio, hard_ratio=hard_ratio)
    if name == "posneg":
        return PosNegSampler(labels=labels, neg_ratio=neg_ratio)
    raise ValueError(f"Unknown sampler '{name}'. Use 'hard', 'posneg' or None.")
