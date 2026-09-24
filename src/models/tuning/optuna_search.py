import optuna
from src.config import DEVICE
from src.models.trainer import train_model


def objective(trial,name,train_loader,val_loader,loss_name):

    lr = trial.suggest_float("lr",1e-5,1e-3,log=True,)

    weight_decay = trial.suggest_float("weight_decay",1e-7,1e-3,log=True)

    dropout = trial.suggest_float("dropout",0.2,0.5,)

    if loss_name == "bce":
        pos_weight = trial.suggest_float("pos_weight",0.3,3.0,)
        focal_alpha = 0.5
        focal_gamma = 2.0
        print(f"\n[TRIAL {trial.number}] "f"{name} | BCE")
        print(f"lr={lr:.6g}")
        print(f"weight_decay={weight_decay:.6g}")
        print(f"dropout={dropout:.3f}")
        print(f"pos_weight={pos_weight:.3f}")

    elif loss_name == "focal":

        pos_weight = None
        focal_alpha = trial.suggest_float("focal_alpha",0.5,0.9)
        focal_gamma = trial.suggest_float("focal_gamma",1.0,5.0,)
        print(
            f"\n[TRIAL {trial.number}] {name} | FOCAL | "
            f"lr={lr:.6g} | "
            f"weight_decay={weight_decay:.6g} | "
            f"dropout={dropout:.3f} | "
            f"alpha={focal_alpha:.3f} | "
            f"gamma={focal_gamma:.3f}"
        )

    else:
        raise ValueError(f"Unknown loss: {loss_name}")
    try:
        model, best_val_auc = train_model(
            name=name,
            train_loader=train_loader,
            val_loader=val_loader,
            n_pos=194,
            n_neg=71,
            loss_name=loss_name,
            pos_weight=pos_weight,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            lr=lr,
            weight_decay=weight_decay,
            epochs=20,
            dropout=dropout,
            device=DEVICE,
            save_path=(f"trial_"f"{loss_name}_"f"{name}_"f"{trial.number}.pth"),
            trial=trial,
            patience=None,)

    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:

        print(
            f"[TRIAL {trial.number}] "
            f"{name}/{loss_name} FAILED: "
            f"{type(e).__name__}: {e}"
        )

        raise optuna.exceptions.TrialPruned() from e

    return float(best_val_auc)
