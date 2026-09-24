import torch


class FocalLoss(torch.nn.Module):

    def __init__(self, alpha=0.5, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.float()
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_factor = (1.0 - p_t).pow(self.gamma)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * focal_factor * bce

        if self.reduction == "mean":
            return loss.mean()

        if self.reduction == "sum":
            return loss.sum()

        return loss



def get_loss_params(loss_name, params):
    if loss_name == "bce":
        return params.get("pos_weight", 1.0), 0.5, 2.0
    return None, params.get("focal_alpha", 0.5), params.get("focal_gamma", 2.0)


def build_loss(loss_name, pos_weight=None, focal_alpha=0.5, focal_gamma=2.0, device=None):

    loss_name = loss_name.lower()

    if loss_name == "bce":
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
        else:
            pos_weight_tensor = None

        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    if loss_name == "focal":
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma).to(device)

    raise ValueError(f"Unknown loss '{loss_name}'.")

