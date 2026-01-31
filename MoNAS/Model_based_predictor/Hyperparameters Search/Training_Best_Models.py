
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    RandomForestRegressor,
)
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neighbors import KNeighborsRegressor

import optuna
import joblib
from xgboost import XGBRegressor

# ============================
# 1. General Configuration
# ============================

import os
import random

RANDOM_STATE = 0


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_global_seed(RANDOM_STATE)

N_SPLITS_OUTER = 10
N_SPLITS_INNER = 10
N_TRIALS_PER_MODEL = 100  

MODEL_NAMES = [
    "xgb",
    "gbrt",
    "extratrees",
    "svr",
    "gpr",
    "knn",
    "rf",
]

# Families
FAMILIES = {
    "boosting": ["xgb", "gbrt"],         
    "bagging_trees": ["rf", "extratrees"],
    "kernel": ["svr", "gpr"],
    "instance": ["knn"],
}

# ============================
# 2. Data load
# ============================

file_path = "1193_P_Trained_models_psnr_result.csv"
data = pd.read_csv(file_path, header=None)


X = np.array([eval(row[0]) for row in data.values], dtype=float)
y = data.iloc[:, -1].values.astype(float)

print(f"Loaded dataset -> X: {X.shape}, y: {y.shape}")

# ============================
# 3. C-index
# ============================


def c_index(y_true, y_pred):
    
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n = len(y_true)
    assert n == len(y_pred)

    num = 0.0
    den = 0.0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if y_true[i] == y_true[j]:
                continue
            den += 1.0
            diff_true = y_true[i] - y_true[j]
            diff_pred = y_pred[i] - y_pred[j]
            if diff_true * diff_pred > 0:
                num += 1.0
            elif diff_pred == 0:
                num += 0.5
    if den == 0:
        return 0.0
    return num / den


# ============================
# 4. Models + Search Space
# ============================


def create_model_from_trial(model_name, trial):
    

    # ------------------------------------------------------------------
    # XGBoost (XGB)
    # ------------------------------------------------------------------
    if model_name == "xgb":
        max_depth = trial.suggest_int("xgb_max_depth", 9, 13)
        n_estimators = trial.suggest_int("xgb_n_estimators", 250, 750)
        learning_rate = trial.suggest_float(
            "xgb_learning_rate", 0.01, 0.15, log=True
        )
        subsample = trial.suggest_float("xgb_subsample", 0.5, 1.0)
        colsample_bytree = trial.suggest_float("xgb_colsample_bytree", 0.55, 0.85)
        reg_lambda = trial.suggest_float("xgb_reg_lambda", 20.0, 30.0, log=True)
        # low > 0 cuando log=True
        reg_alpha = trial.suggest_float("xgb_reg_alpha", 1e-3, 1.5, log=True)

        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            tree_method="hist",
        )
        return model

    # ------------------------------------------------------------------
    # GradientBoostingRegressor (GBRT)
    # ------------------------------------------------------------------
    elif model_name == "gbrt":
        max_depth = trial.suggest_int("gbrt_max_depth", 4, 10)
        min_samples_leaf = trial.suggest_int("gbrt_min_samples_leaf", 15, 40)
        n_estimators = trial.suggest_int("gbrt_n_estimators", 300, 650)
        learning_rate = trial.suggest_float(
            "gbrt_learning_rate", 0.02, 0.15, log=True
        )
        subsample = trial.suggest_float("gbrt_subsample", 0.7, 1.0)

        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            learning_rate=learning_rate,
            subsample=subsample,
            random_state=RANDOM_STATE,
        )
        return model

    # ------------------------------------------------------------------
    # ExtraTreesRegressor
    # ------------------------------------------------------------------
    elif model_name == "extratrees":
        max_depth = trial.suggest_int("extratrees_max_depth", 15, 25)
        min_samples_split = trial.suggest_int("extratrees_min_samples_split", 2, 8)
        min_samples_leaf = trial.suggest_int("extratrees_min_samples_leaf", 1, 4)
        n_estimators = trial.suggest_int("extratrees_n_estimators", 250, 550)
        max_features = trial.suggest_categorical(
            "extratrees_max_features", ["sqrt", "log2"]
        )

        model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        return model

    # ------------------------------------------------------------------
    # SVR (RBF)
    # ------------------------------------------------------------------
    elif model_name == "svr":
        C_svr = trial.suggest_float("svr_C", 1.0, 100.0, log=True)
        epsilon = trial.suggest_float("svr_epsilon", 0.02, 0.2, log=True)
        gamma = trial.suggest_float("svr_gamma", 0.01, 0.08, log=True)

        svr = SVR(C=C_svr, epsilon=epsilon, kernel="rbf", gamma=gamma)
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svr", svr),
            ]
        )
        return model

    # ------------------------------------------------------------------
    # GaussianProcessRegressor (GPR)
    # ------------------------------------------------------------------
    elif model_name == "gpr":
        alpha = trial.suggest_float("gpr_alpha", 1e-6, 1e-2, log=True)
        length_scale = trial.suggest_float("gpr_length_scale", 0.5, 2.5, log=True)
        constant_value = trial.suggest_float("gpr_constant", 1.0, 10.0, log=True)

        kernel = C(constant_value) * RBF(length_scale=length_scale)
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            normalize_y=True,
            random_state=RANDOM_STATE,
        )
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("gpr", gpr),
            ]
        )
        return model

    # ------------------------------------------------------------------
    # KNN
    # ------------------------------------------------------------------
    elif model_name == "knn":
        n_neighbors = trial.suggest_int("knn_n_neighbors", 5, 80)
        weights = trial.suggest_categorical("knn_weights", ["distance"])
        p = trial.suggest_int("knn_p", 1, 2)

        knn = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=weights,
            p=p,
        )
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("knn", knn),
            ]
        )
        return model

    # ------------------------------------------------------------------
    # RandomForestRegressor (RF)
    # ------------------------------------------------------------------
    elif model_name == "rf":
        max_depth = trial.suggest_int("rf_max_depth", 10, 20)
        min_samples_split = trial.suggest_int("rf_min_samples_split", 2, 4)
        min_samples_leaf = trial.suggest_int("rf_min_samples_leaf", 1, 3)
        n_estimators = trial.suggest_int("rf_n_estimators", 350, 700)
        max_features = trial.suggest_categorical(
            "rf_max_features", ["sqrt", "log2"]
        )

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        return model

    else:
        raise ValueError(f"No Model {model_name}")


def create_model_from_params(model_name, params):
    

    class DummyTrial:
        def __init__(self, params):
            self.params = params

        def suggest_int(self, name, low, high):
            return int(self.params[name])

        def suggest_float(self, name, low, high, log=False):
            return float(self.params[name])

        def suggest_categorical(self, name, choices):
            return self.params[name]

    dummy_trial = DummyTrial(params)
    return create_model_from_trial(model_name, dummy_trial)


# ============================
# 5. Objective (inner CV) – min MSE
# ============================

def make_objective(model_name, X_inner, y_inner):
    def objective(trial):
        model = create_model_from_trial(model_name, trial)
        cv_inner = KFold(
            n_splits=N_SPLITS_INNER,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        mse_values = []

        for tr_idx, val_idx in cv_inner.split(X_inner):
            X_tr, X_val = X_inner[tr_idx], X_inner[val_idx]
            y_tr, y_val = y_inner[tr_idx], y_inner[val_idx]

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            mse_values.append(mse)

        return float(np.mean(mse_values))  

    return objective


# ============================
# 6. Nested CV + BO
# ============================

def run_nested_cv_with_bo(X, y):
    n_samples = X.shape[0]
    outer_cv = KFold(
        n_splits=N_SPLITS_OUTER,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results_rows = []
    best_params_by_model_and_fold = {m: {} for m in MODEL_NAMES}
    oof_predictions = {m: np.full(n_samples, np.nan, dtype=float) for m in MODEL_NAMES}

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
        print(f"\n=== Outer fold {fold_idx}/{N_SPLITS_OUTER} ===")
        X_train_outer, X_test_outer = X[train_idx], X[test_idx]
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]

        for model_name in MODEL_NAMES:
            print(f"  -> Optimizando modelo: {model_name}")
            study = optuna.create_study(direction="minimize")
            objective = make_objective(model_name, X_train_outer, y_train_outer)
            study.optimize(objective, n_trials=N_TRIALS_PER_MODEL, show_progress_bar=False)

            best_params = study.best_params
            best_value = study.best_value  # MSE medio en inner CV

            print(f"     Mejor MSE inner: {best_value:.4f}")

            best_model = create_model_from_params(model_name, best_params)
            best_model.fit(X_train_outer, y_train_outer)

            y_pred_test = best_model.predict(X_test_outer)
            c_idx_outer = c_index(y_test_outer, y_pred_test)

            mse = mean_squared_error(y_test_outer, y_pred_test)
            rmse = float(np.sqrt(mse))
            mae = mean_absolute_error(y_test_outer, y_pred_test)
            r2 = r2_score(y_test_outer, y_pred_test)
            if np.std(y_test_outer) > 0 and np.std(y_pred_test) > 0:
                pearson = float(np.corrcoef(y_test_outer, y_pred_test)[0, 1])
            else:
                pearson = 0.0

            print(
                f"     Outer fold {fold_idx}: C={c_idx_outer:.4f}, "
                f"MSE={mse:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}, Pearson={pearson:.4f}"
            )

            oof_predictions[model_name][test_idx] = y_pred_test

            best_params_by_model_and_fold[model_name][fold_idx] = {
                **best_params,
                "inner_mean_mse": best_value,
                "outer_c_index": c_idx_outer,
                "outer_mse": mse,
                "outer_rmse": rmse,
                "outer_mae": mae,
                "outer_r2": r2,
                "outer_pearson": pearson,
            }

            row = {
                "fold": fold_idx,
                "model": model_name,
                "inner_mean_mse": best_value,
                "outer_c_index": c_idx_outer,
                "outer_mse": mse,
                "outer_rmse": rmse,
                "outer_mae": mae,
                "outer_r2": r2,
                "outer_pearson": pearson,
                "n_train_outer": len(train_idx),
                "n_test_outer": len(test_idx),
                "best_params_json": json.dumps(best_params),
            }
            results_rows.append(row)

    return results_rows, best_params_by_model_and_fold, oof_predictions


# ============================
# 7. Summary
# ============================

def summarize_by_model(results_rows, best_params_by_model_and_fold):
    summary_rows = []

    for model_name in MODEL_NAMES:
        model_rows = [r for r in results_rows if r["model"] == model_name]

        outer_c = np.array([r["outer_c_index"] for r in model_rows], dtype=float)
        outer_mse = np.array([r["outer_mse"] for r in model_rows], dtype=float)
        outer_rmse = np.array([r["outer_rmse"] for r in model_rows], dtype=float)
        outer_mae = np.array([r["outer_mae"] for r in model_rows], dtype=float)
        outer_r2 = np.array([r["outer_r2"] for r in model_rows], dtype=float)
        outer_pearson = np.array([r["outer_pearson"] for r in model_rows], dtype=float)

        mean_c = float(outer_c.mean())
        mean_mse = float(outer_mse.mean())
        mean_rmse = float(outer_rmse.mean())
        mean_mae = float(outer_mae.mean())
        mean_r2 = float(outer_r2.mean())
        mean_pearson = float(outer_pearson.mean())

        std_c = float(outer_c.std())
        std_mse = float(outer_mse.std())
        std_rmse = float(outer_rmse.std())
        std_mae = float(outer_mae.std())
        std_r2 = float(outer_r2.std())
        std_pearson = float(outer_pearson.std())

        # Ahora el "mejor fold" se elige por MSE (mínimo)
        best_idx = int(np.argmin(outer_mse))
        best_fold_row = model_rows[best_idx]
        best_fold = best_fold_row["fold"]
        best_outer_mse = best_fold_row["outer_mse"]

        best_params = best_params_by_model_and_fold[model_name][best_fold].copy()
        best_params_pure = {
            k: v
            for k, v in best_params.items()
            if k not in [
                "inner_mean_mse",
                "outer_c_index",
                "outer_mse",
                "outer_rmse",
                "outer_mae",
                "outer_r2",
                "outer_pearson",
            ]
        }

        summary_rows.append(
            {
                "model": model_name,
                "mean_outer_c_index": mean_c,
                "std_outer_c_index": std_c,
                "mean_outer_mse": mean_mse,
                "std_outer_mse": std_mse,
                "mean_outer_rmse": mean_rmse,
                "std_outer_rmse": std_rmse,
                "mean_outer_mae": mean_mae,
                "std_outer_mae": std_mae,
                "mean_outer_r2": mean_r2,
                "std_outer_r2": std_r2,
                "mean_outer_pearson": mean_pearson,
                "std_outer_pearson": std_pearson,
                "best_fold": best_fold,
                "best_fold_outer_mse": best_outer_mse,
                "best_params_json": json.dumps(best_params_pure),
            }
        )

    return summary_rows


# ============================
# 8. Ensemble 
# ============================

def run_ensemble_optimization(y, oof_predictions, ensemble_model_names):
    """
    Usa predicciones out-of-fold de ensemble_model_names para optimizar pesos
    vía Optuna, minimizando MSE.
    Devuelve info del ensamble, métricas globales y predicciones OOF del ensamble.
    """
    preds_matrix = []
    for m in ensemble_model_names:
        preds_matrix.append(oof_predictions[m])
    preds_matrix = np.vstack(preds_matrix).T  # (n_samples, n_models)

    assert not np.isnan(preds_matrix).any(), "Hay NaNs en predicciones OOF"

    def objective(trial):
        raw_weights = []
        for m in ensemble_model_names:
            w = trial.suggest_float(f"w_{m}", 0.0, 1.0)
            raw_weights.append(w)
        raw_weights = np.array(raw_weights, dtype=float)
        if raw_weights.sum() == 0:
            weights = np.ones_like(raw_weights) / len(raw_weights)
        else:
            weights = raw_weights / raw_weights.sum()

        y_pred_ens = preds_matrix @ weights
        return mean_squared_error(y, y_pred_ens)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50, show_progress_bar=False)

    best_weights_raw = np.array(
        [study.best_params[f"w_{m}"] for m in ensemble_model_names], dtype=float
    )
    best_weights = best_weights_raw / best_weights_raw.sum()

    y_pred_ensemble = preds_matrix @ best_weights

    mse = mean_squared_error(y, y_pred_ensemble)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y, y_pred_ensemble)
    r2 = r2_score(y, y_pred_ensemble)
    if np.std(y) > 0 and np.std(y_pred_ensemble) > 0:
        pearson = float(np.corrcoef(y, y_pred_ensemble)[0, 1])
    else:
        pearson = 0.0
    c_idx = c_index(y, y_pred_ensemble)

    print("\n=== ENSAMBLE FINAL (OOF) ===")
    print("Modelos en el ensamble:", ensemble_model_names)
    for m, w in zip(ensemble_model_names, best_weights):
        print(f"  {m}: {w:.4f}")
    print(
        f"C-index (OOF)={c_idx:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, "
        f"MAE={mae:.4f}, R2={r2:.4f}, Pearson={pearson:.4f}"
    )

    ensemble_info = {
        "models": ensemble_model_names,
        "weights": best_weights.tolist(),
        "best_mse_oof": float(mse),
        "raw_params": study.best_params,
    }

    ensemble_metrics = {
        "c_index": float(c_idx),
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "pearson": float(pearson),
    }

    return ensemble_info, ensemble_metrics, y_pred_ensemble




class WeightedEnsembleRegressor:
    
    def __init__(self, models, weights):
        self.models = models
        self.weights = np.asarray(weights, dtype=float)
        if self.weights.sum() == 0:
            self.weights = np.ones_like(self.weights) / len(self.weights)
        else:
            self.weights = self.weights / self.weights.sum()

    def predict(self, X):
        preds = []
        for m in self.models:
            preds.append(m.predict(X))
        preds = np.vstack(preds).T  # (n_samples, n_models)
        return preds @ self.weights


# ============================
# 10. Main
# ============================

if __name__ == "__main__":
    # 10.1 Nested CV + BO
    results_rows, best_params_by_model_and_fold, oof_predictions = run_nested_cv_with_bo(
        X, y
    )

    df_by_fold = pd.DataFrame(results_rows)
    df_by_fold.to_csv("mse_nested_cv_by_fold.csv", index=False)
    print("\nSave: mse_nested_cv_by_fold.csv")

    # 10.2 summary
    summary_rows = summarize_by_model(results_rows, best_params_by_model_and_fold)

    # 10.3 Selecct best model per family MSE
    ensemble_model_names = []
    for fam, candidates in FAMILIES.items():
        rows_for_fam = [r for r in summary_rows if r["model"] in candidates]
        if not rows_for_fam:
            continue
        # Ahora minimizamos MSE promedio
        best_row = min(rows_for_fam, key=lambda r: r["mean_outer_mse"])
        best_model_name = best_row["model"]
        ensemble_model_names.append(best_model_name)
        print(
            f"Familye '{fam}': best model = {best_model_name} "
            f"(mean MSE={best_row['mean_outer_mse']:.4f})"
        )

    # 10.4 Ensaemble 
    ensemble_info, ensemble_metrics, ensemble_oof_pred = run_ensemble_optimization(
        y, oof_predictions, ensemble_model_names
    )

    with open("mse_ensemble_info.json", "w", encoding="utf-8") as f:
        json.dump(ensemble_info, f, indent=2)
    print("Guardado: mse_ ensemble_info.json")

    # 10.5 Add 'ensemble' to nested_cv_by_model
    ensemble_row = {
        "model": "ensemble",
        "mean_outer_c_index": ensemble_metrics["c_index"],
        "std_outer_c_index": 0.0,
        "mean_outer_mse": ensemble_metrics["mse"],
        "std_outer_mse": 0.0,
        "mean_outer_rmse": ensemble_metrics["rmse"],
        "std_outer_rmse": 0.0,
        "mean_outer_mae": ensemble_metrics["mae"],
        "std_outer_mae": 0.0,
        "mean_outer_r2": ensemble_metrics["r2"],
        "std_outer_r2": 0.0,
        "mean_outer_pearson": ensemble_metrics["pearson"],
        "std_outer_pearson": 0.0,
        "best_fold": -1,
        "best_fold_outer_mse": ensemble_metrics["mse"],
        "best_params_json": json.dumps(
            {
                "weights": ensemble_info["weights"],
                "models": ensemble_info["models"],
            }
        ),
    }
    summary_rows.append(ensemble_row)

    df_by_model = pd.DataFrame(summary_rows)
    df_by_model.to_csv("mse_nested_cv_by_model.csv", index=False)
    print("Save: mse_nested_cv_by_model.csv")

    # 10.6 Full Training
    print("\n=== Full Training ===")
    final_models = {}

    for row in summary_rows:
        model_name = row["model"]
        if model_name == "ensemble":
            continue  
        best_params = json.loads(row["best_params_json"])

        print(f"  -> final model: {model_name}")
        final_model = create_model_from_params(model_name, best_params)
        final_model.fit(X, y)

        filename = f"final_model_{model_name}.pkl"
        joblib.dump(final_model, filename)
        final_models[model_name] = final_model
        print(f"     Save: {filename}")

    
    print("\n=== Creating final ensemble and save as .pkl ===")
    ensemble_models = [final_models[m] for m in ensemble_info["models"]]
    ensemble_weights = np.array(ensemble_info["weights"], dtype=float)

    ensemble_regressor = WeightedEnsembleRegressor(ensemble_models, ensemble_weights)

    ensemble_filename = "final_model_ensemble.pkl"
    joblib.dump(ensemble_regressor, ensemble_filename)
    print(f" Final ensamble: {ensemble_filename}")
