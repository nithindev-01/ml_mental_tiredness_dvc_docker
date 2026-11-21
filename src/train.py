"""
Stage 2 - TRAIN
Reads data/processed/train.csv + test.csv, tunes 10 regression algorithms with Optuna
(n_trials from params.yaml), logs each algorithm to its own MLflow experiment, and saves
the best model (lowest CV RMSE) to models/model.pkl and the leaderboard to models/leaderboard.csv.

Run from the project root:
    python src/train.py
"""

import os
import json
import time
import yaml
import warnings

os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # or 1
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # no pop-up windows, figures go to MLflow
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
import optuna

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    explained_variance_score,
    max_error,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# --------------------------------------------------------------------------- #
# Config (from params.yaml)
# --------------------------------------------------------------------------- #
with open("params.yaml") as f:
    params = yaml.safe_load(f)["train"]

RANDOM_STATE = params["random_state"]
N_TRIALS = params["n_trials"]          # same for every algorithm
CV_FOLDS = params["cv_folds"]
TARGET = "mental_tiredness_score"
EXPERIMENT_PREFIX = ""   # e.g. "Regression_" -> experiments named Regression_Ridge, ...

# --------------------------------------------------------------------------- #
# Load processed data (created by src/prepare.py)
# --------------------------------------------------------------------------- #
train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

X_train, y_train = train.drop(TARGET, axis=1), train[TARGET]
X_test, y_test = test.drop(TARGET, axis=1), test[TARGET]

num_cols = X_train.select_dtypes(include="number").columns.tolist()
cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()
print(f"Train: {X_train.shape}, Test: {X_test.shape}")


# --------------------------------------------------------------------------- #
# Pipeline builders
# --------------------------------------------------------------------------- #
def build_preprocessor(scaler_type):
    scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler),
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols),
    ])


def build_model(name, p):
    """Create a model from a parameter dict (used by Optuna trials AND the final fit)."""
    if name == "Ridge":
        return Ridge(alpha=p["alpha"], solver=p["solver"])
    if name == "Lasso":
        return Lasso(alpha=p["alpha"], max_iter=20000, random_state=RANDOM_STATE)
    if name == "ElasticNet":
        return ElasticNet(alpha=p["alpha"], l1_ratio=p["l1_ratio"],
                          max_iter=20000, random_state=RANDOM_STATE)
    if name == "KNN":
        return KNeighborsRegressor(n_neighbors=p["n_neighbors"], weights=p["weights"], p=p["p"])
    if name == "DecisionTree":
        return DecisionTreeRegressor(
            criterion=p["criterion"], max_depth=p["max_depth"],
            min_samples_split=p["min_samples_split"], min_samples_leaf=p["min_samples_leaf"],
            max_features=p["max_features"], random_state=RANDOM_STATE)
    if name == "SVR":
        return SVR(kernel=p["kernel"], C=p["C"], epsilon=p["epsilon"],
                   gamma=p.get("gamma", "scale"), cache_size=1000)
    if name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"],
            min_samples_split=p["min_samples_split"], min_samples_leaf=p["min_samples_leaf"],
            max_features=p["max_features"], bootstrap=p["bootstrap"],
            random_state=RANDOM_STATE, n_jobs=1)
    if name == "ExtraTrees":
        return ExtraTreesRegressor(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"],
            min_samples_split=p["min_samples_split"], min_samples_leaf=p["min_samples_leaf"],
            max_features=p["max_features"], bootstrap=p["bootstrap"],
            random_state=RANDOM_STATE, n_jobs=1)
    if name == "GradientBoosting":
        return GradientBoostingRegressor(
            n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
            max_depth=p["max_depth"], min_samples_split=p["min_samples_split"],
            min_samples_leaf=p["min_samples_leaf"], max_features=p["max_features"],
            subsample=p["subsample"], random_state=RANDOM_STATE)
    if name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(
            learning_rate=p["learning_rate"], max_iter=p["max_iter"],
            max_depth=p["max_depth"], max_leaf_nodes=p["max_leaf_nodes"],
            min_samples_leaf=p["min_samples_leaf"], l2_regularization=p["l2_regularization"],
            random_state=RANDOM_STATE)
    raise ValueError(name)


def build_pipeline(name, p):
    return Pipeline([
        ("Preprocessor", build_preprocessor(p["scaler_type"])),
        # degree=1 is an identity step; linear models can tune degree 1 or 2
        ("Poly", PolynomialFeatures(degree=p.get("poly_degree", 1), include_bias=False)),
        ("Model", build_model(name, p)),
    ])


# --------------------------------------------------------------------------- #
# Search spaces (one function per algorithm, all return a param dict)
# --------------------------------------------------------------------------- #
def space_ridge(t):
    return {
        "alpha": t.suggest_float("alpha", 1e-3, 1e3, log=True),
        "solver": t.suggest_categorical("solver", ["auto", "svd", "cholesky", "lsqr"]),
        "poly_degree": t.suggest_int("poly_degree", 1, 2),
    }


def space_lasso(t):
    return {
        "alpha": t.suggest_float("alpha", 1e-4, 10.0, log=True),
        "poly_degree": t.suggest_int("poly_degree", 1, 2),
    }


def space_elasticnet(t):
    return {
        "alpha": t.suggest_float("alpha", 1e-4, 10.0, log=True),
        "l1_ratio": t.suggest_float("l1_ratio", 0.05, 0.95),
        "poly_degree": t.suggest_int("poly_degree", 1, 2),
    }


def space_knn(t):
    return {
        "n_neighbors": t.suggest_int("n_neighbors", 3, 51, step=2),
        "weights": t.suggest_categorical("weights", ["uniform", "distance"]),
        "p": t.suggest_int("p", 1, 3),
    }


def space_dt(t):
    return {
        "criterion": t.suggest_categorical("criterion", ["squared_error", "friedman_mse", "absolute_error"]),
        "max_depth": t.suggest_int("max_depth", 2, 30),
        "min_samples_split": t.suggest_int("min_samples_split", 2, 30),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 30),
        "max_features": t.suggest_categorical("max_features", [None, "sqrt", "log2"]),
    }


def space_svr(t):
    p = {
        "kernel": t.suggest_categorical("kernel", ["linear", "rbf"]),
        "C": t.suggest_float("C", 1e-2, 1e2, log=True),
        "epsilon": t.suggest_float("epsilon", 1e-2, 1.0, log=True),
    }
    if p["kernel"] == "rbf":
        p["gamma"] = t.suggest_float("gamma", 1e-4, 1e-1, log=True)
    return p


def space_rf(t):
    return {
        "n_estimators": t.suggest_int("n_estimators", 100, 400, step=50),
        "max_depth": t.suggest_int("max_depth", 5, 30),
        "min_samples_split": t.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 20),
        "max_features": t.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "bootstrap": t.suggest_categorical("bootstrap", [True, False]),
    }


def space_gb(t):
    return {
        "n_estimators": t.suggest_int("n_estimators", 100, 400, step=50),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": t.suggest_int("max_depth", 2, 8),
        "min_samples_split": t.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 20),
        "max_features": t.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
    }


def space_hgb(t):
    return {
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_iter": t.suggest_int("max_iter", 100, 500, step=50),
        "max_depth": t.suggest_int("max_depth", 3, 12),
        "max_leaf_nodes": t.suggest_int("max_leaf_nodes", 15, 63),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 5, 50),
        "l2_regularization": t.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
    }


# name -> (search space function, number of CV folds)
SEARCH = {
    "Ridge":               (space_ridge, CV_FOLDS),
    "Lasso":               (space_lasso, CV_FOLDS),
    "ElasticNet":          (space_elasticnet, CV_FOLDS),
    "KNN":                 (space_knn, CV_FOLDS),
    "DecisionTree":        (space_dt, CV_FOLDS),
    "SVR":                 (space_svr, 3),        # SVR is slow, so 3 folds
    "RandomForest":        (space_rf, CV_FOLDS),
    "ExtraTrees":          (space_rf, CV_FOLDS),  # same space as RandomForest
    "GradientBoosting":    (space_gb, CV_FOLDS),
    "HistGradientBoosting": (space_hgb, CV_FOLDS),
}


def make_objective(name):
    space_fn, folds = SEARCH[name]

    def objective(trial):
        p = {"scaler_type": trial.suggest_categorical("scaler_type", ["standard", "minmax"])}
        p.update(space_fn(trial))
        pipe = build_pipeline(name, p)
        kf = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(
            pipe, X_train, y_train,
            scoring="neg_root_mean_squared_error", cv=kf, n_jobs=-1,
        )
        return -scores.mean()  # CV RMSE (minimise)

    return objective


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
def evaluate(y_true, y_pred, n_features):
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1) if n > n_features + 1 else np.nan
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2,
        "adj_r2": adj_r2,
        "median_ae": median_absolute_error(y_true, y_pred),
        "explained_var": explained_variance_score(y_true, y_pred),
        "max_error": max_error(y_true, y_pred),
    }


def make_plots(y_true, y_pred, title):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].scatter(y_true, y_pred, s=8, alpha=0.5)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax[0].plot(lims, lims, "r--")
    ax[0].set_xlabel("Actual")
    ax[0].set_ylabel("Predicted")
    ax[0].set_title(f"{title}: Actual vs Predicted")
    resid = y_true - y_pred
    ax[1].scatter(y_pred, resid, s=8, alpha=0.5)
    ax[1].axhline(0, color="r", linestyle="--")
    ax[1].set_xlabel("Predicted")
    ax[1].set_ylabel("Residual")
    ax[1].set_title(f"{title}: Residuals")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
import logging
logging.getLogger("mlflow.sklearn").setLevel(logging.ERROR)   # hides the harmless pickle warning

from mlflow.tracking import MlflowClient
_client = MlflowClient()


def set_experiment_safe(name):
    """Select (or create) an experiment; restore it first if it was deleted in the UI."""
    exp = _client.get_experiment_by_name(name)
    if exp is not None and exp.lifecycle_stage == "deleted":
        _client.restore_experiment(exp.experiment_id)
    mlflow.set_experiment(name)


results = {}
fitted_pipelines = {}
model_dict = {name: i for i, name in enumerate(SEARCH.keys())}
scaler_dict = {"standard": 0, "minmax": 1}

for model_name in SEARCH:
    print(f"\n--- Optimizing {model_name} ({N_TRIALS} trials) ---")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )

    # ---- Hyperparameter tuning (plain Optuna, no MLflow callback -> no child runs) ----
    t0 = time.time()
    study.optimize(make_objective(model_name), n_trials=N_TRIALS)
    hpt_time = time.time() - t0

    best_params = study.best_params
    print(f"Best CV RMSE for {model_name}: {study.best_value:.4f}")

    # ---- One experiment per algorithm, one run inside it ----
    set_experiment_safe(f"{EXPERIMENT_PREFIX}{model_name}")
    with mlflow.start_run(run_name=model_name):
        # ---- Final model with best params ----
        pipe = build_pipeline(model_name, best_params)

        t0 = time.time()
        pipe.fit(X_train, y_train)
        fit_time = time.time() - t0

        t0 = time.time()
        y_pred = pipe.predict(X_test)
        predict_time = time.time() - t0
        y_train_pred = pipe.predict(X_train)

        n_feat = pipe[:-1].transform(X_test.head(5)).shape[1]
        train_m = evaluate(y_train, y_train_pred, n_feat)
        test_m = evaluate(y_test, y_pred, n_feat)

        print(f"{model_name}  Train RMSE={train_m['rmse']:.4f} | Test RMSE={test_m['rmse']:.4f} | "
              f"Test MAE={test_m['mae']:.4f} | Test R2={test_m['r2']:.4f}")

        # ---- Model size ----
        tmp_path = f"{model_name}_tmp.pkl"
        joblib.dump(pipe, tmp_path)
        model_size = os.path.getsize(tmp_path)
        os.remove(tmp_path)

        # ---- MLflow logging ----
        mlflow.set_tag("algorithm", model_name)

        # Optuna trial history lives inside this single run (no child runs)
        for tr in study.trials:
            mlflow.log_metric("trial_cv_rmse", tr.value, step=tr.number)
        study.trials_dataframe().to_csv("trials.csv", index=False)
        mlflow.log_artifact("trials.csv", artifact_path="optuna")
        os.remove("trials.csv")
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("model_id", model_dict[model_name])
        mlflow.log_metric("Scaler_id", scaler_dict[best_params["scaler_type"]])
        mlflow.log_metric("best_cv_rmse", study.best_value)
        for k, v in train_m.items():
            mlflow.log_metric(f"train_{k}", v)
        for k, v in test_m.items():
            mlflow.log_metric(f"test_{k}", v)
        mlflow.log_metric("rmse_gap", test_m["rmse"] - train_m["rmse"])  # overfitting indicator
        mlflow.log_metric("hpt_time", hpt_time)
        mlflow.log_metric("train_time", fit_time)
        mlflow.log_metric("test_time", predict_time)
        mlflow.log_metric("model_size", model_size)

        fig = make_plots(y_test.values, np.asarray(y_pred), model_name)
        mlflow.log_figure(fig, f"{model_name}_diagnostics.png")
        plt.close(fig)

        mlflow.sklearn.log_model(
            pipe,
            name=f"{model_name}_mental_tiredness_model",
            serialization_format="cloudpickle",
        )

    fitted_pipelines[model_name] = pipe
    results[model_name] = {
        "Model": model_name,
        "CV_RMSE": study.best_value,
        "Train_RMSE": train_m["rmse"],
        "Test_RMSE": test_m["rmse"],
        "Test_MAE": test_m["mae"],
        "Test_MSE": test_m["mse"],
        "Test_R2": test_m["r2"],
        "Test_AdjR2": test_m["adj_r2"],
        "Test_MedAE": test_m["median_ae"],
        "Test_ExplVar": test_m["explained_var"],
        "Test_MaxErr": test_m["max_error"],
        "RMSE_Gap": test_m["rmse"] - train_m["rmse"],
        "HPT_Time_s": hpt_time,
        "Fit_Time_s": fit_time,
        "Predict_Time_s": predict_time,
        "Model_Size_KB": model_size / 1024,
        "Best_Params": best_params,
    }

# --------------------------------------------------------------------------- #
# Leaderboard
# --------------------------------------------------------------------------- #
leaderboard = pd.DataFrame(results.values()).sort_values("CV_RMSE").reset_index(drop=True)
leaderboard.index += 1

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
print("\n================ LEADERBOARD (sorted by CV RMSE) ================")
print(leaderboard.drop(columns=["Best_Params"]).round(4).to_string())

# Best model is chosen on cross-validation (not on the test set) to avoid test leakage
best_name = leaderboard.loc[1, "Model"]
best_row = leaderboard.loc[1]
print(f"\nBest model (lowest CV RMSE): {best_name}")
print(f"  Test RMSE = {best_row['Test_RMSE']:.4f}")
print(f"  Test MAE  = {best_row['Test_MAE']:.4f}")
print(f"  Test R2   = {best_row['Test_R2']:.4f}")
print(f"  Best params: {best_row['Best_Params']}")

os.makedirs("models", exist_ok=True)
leaderboard.to_csv("models/leaderboard.csv", index_label="Rank")
joblib.dump(fitted_pipelines[best_name], "models/model.pkl")
with open("models/best_model_info.json", "w") as f:
    json.dump({"best_model": best_name, "best_params": best_row["Best_Params"]}, f, indent=2, default=str)
print("\nSaved: models/model.pkl, models/leaderboard.csv, models/best_model_info.json")