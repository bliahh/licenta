import sklearn
from sklearn.model_selection import  RandomizedSearchCV
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, balanced_accuracy_score
from xgboost import XGBClassifier

ID_COLUMNS = ["file", "patient", "scan", "label"]

center_labels = {1:0, 2:1, 4:2, 5:3}


def map_center_labels(table):
    y_data = table["center"]
    y_final = y_data.map(center_labels)
    return y_final

def meta_build_train_test_split(dataset, include_center=False, seed=42):
    y_data = dataset["label"]
    x_data = dataset.drop(columns=[c for c in ID_COLUMNS if c in dataset])
    if not include_center and "center" in x_data:
        x_data = x_data.drop(columns=["center"])
    elif "center" in x_data:
        x_data["center"] = x_data["center"].astype("category")
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_data, y_data, test_size=0.2, random_state=seed, stratify=y_data)

    return x_train, y_train, x_test, y_test





def train_xgboost(x_train, y_train, multiclass=False):
    print("\nTRAIN\n")
    model = XGBClassifier(n_jobs=-1, random_state=42, tree_method="hist", enable_categorical=True, eval_metric="mlogloss" if multiclass else "logloss")
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    }
    random_search = RandomizedSearchCV(model, param_grid, n_iter=30, cv=5, random_state=42, n_jobs=-1, scoring="roc_auc_ovr" if multiclass else "roc_auc")
    random_search.fit(x_train, y_train)
    print("Best Params:", random_search.best_params_)
    print("CV AUC:", round(random_search.best_score_, 3))
    return random_search.best_estimator_, random_search.best_params_


def test_classifier(model, x_test, y_test):
    print("\nTEST\n")
    p = model.predict_proba(x_test)[:, 1]
    y_pred = (p >= 0.5).astype(int)
    print(f"AUC test: {roc_auc_score(y_test, p):.3f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    return p





def test_classifier_multiclass(model, x_test, y_test, class_names=None):
    print("\nTEST\n")
    p = model.predict_proba(x_test)
    y_pred = p.argmax(axis=1)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"AUC test (ovr): {roc_auc_score(y_test, p, multi_class='ovr'):.3f}")
    print(f"Balanced accuracy: {balanced_acc:.3f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=class_names))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    return p, balanced_acc


