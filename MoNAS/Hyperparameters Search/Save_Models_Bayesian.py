import numpy as np
import pandas as pd
import optuna
from optuna.exceptions import TrialPruned

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr, pearsonr

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    BayesianRidge
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,

)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor

import joblib
import os
import multiprocessing


# ===============================================================
# 1. LOAD DATA
# ===============================================================

file_path = "1193_P_Trained_models_psnr_result.csv"  # adjust if needed
data = pd.read_csv(file_path, header=None)

# First column: list-like string -> Python list -> np.array
X = np.array([eval(row[0]) for row in data.values], dtype=float)

# Last column is the target
y = data.iloc[:, -1].values.astype(float)

print(f"Loaded dataset -> X: {X.shape}, y: {y.shape}")


# ===============================================================
# 2. METRICS (C-index + error metrics + safe correlations)
# ===============================================================

def c_index(y_true, y_pred):
    """
    Concordance index (pairwise ranking metric).
    Complexity O(n^2), but with ~2000 samples it's acceptable.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n = len(y_true)
    num = 0.0
    den = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] == y_true[j]:
                continue
            den += 1
            diff_true = y_true[i] - y_true[j]
            diff_pred = y_pred[i] - y_pred[j]
            if diff_true * diff_pred > 0:
                num += 1
            elif diff_pred == 0:
                num += 0.5

    return num / den if den > 0 else np.nan


def mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Avoid correlation warnings when inputs are constant
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        spearman = np.nan
        pearson = np.nan
    else:
        try:
            spearman = spearmanr(y_true, y_pred).correlation
        except Exception:
            spearman = np.nan

        try:
            pearson = pearsonr(y_true, y_pred)[0]
        except Exception:
            pearson = np.nan

    cidx = c_index(y_true, y_pred)
    mape_val = mape(y_true, y_pred)

    return dict(
        mse=mse,
        rmse=rmse,
        mae=mae,
        r2=r2,
        spearman=spearman,
        pearson=pearson,
        cindex=cidx,
        mape=mape_val
    )


# ===============================================================
# 3. TRAIN/TEST SPLIT + STANDARD SCALING
# ===============================================================

RANDOM_STATE = 4
print("Using RANDOM_STATE =", RANDOM_STATE)
TEST_SIZE = 0.20

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")
print("Scaler saved to models/scaler.pkl")


# ===============================================================
# 4. SEARCH-SPACE UTILITIES (coarse → fine)
# ===============================================================

def log_range_around(center, base_low, base_high, factor_round):
    """
    Creates a log-scale range around 'center' within [base_low, base_high],
    using a multiplicative factor for widening/narrowing.
    """
    center = float(center)
    if center <= 0:
        return base_low, base_high

    low = max(base_low, center / factor_round)
    high = min(base_high, center * factor_round)
    if low >= high:
        low, high = base_low, base_high
    return low, high


def lin_range_around(center, base_low, base_high, frac_width):
    """
    Linear range around 'center' within [base_low, base_high],
    with width = frac_width * (base_high - base_low).
    """
    center = float(center)
    width = (base_high - base_low) * frac_width
    low = max(base_low, center - width / 2.0)
    high = min(base_high, center + width / 2.0)
    if low >= high:
        low, high = base_low, base_high
    return low, high


def int_range_around(center, base_low, base_high, frac_width, min_span=2):
    """
    Integer range around 'center' within [base_low, base_high].
    """
    center = int(center)
    span = max(min_span, int((base_high - base_low) * frac_width))
    low = max(base_low, center - span)
    high = min(base_high, center + span)
    if low >= high:
        low, high = base_low, base_high
    return int(low), int(high)


def get_frac_for_round(round_idx):
    """
    Controls how "narrow" the refined space is.
    round 1: 1.0 (full coarse range)
    round 2: 0.5
    round 3: 0.25
    """
    if round_idx == 2:
        return 0.5
    elif round_idx == 3:
        return 0.25
    else:
        return 1.0


# ===============================================================
# 5. MODELS AND SPACES (COARSE → REFINED)
# ===============================================================

def create_model_from_trial(trial, name, round_idx, best_params_prev=None):
    """
    Create model for a given round.
    round_idx = 1 -> coarse search (full wide space)
    round_idx = 2,3 -> refined around best_params_prev (if available).
    """

    refine_frac = get_frac_for_round(round_idx)

    # --- LINEAR MODELS ---
    if name == "Linear":
        return LinearRegression(n_jobs=-1)

    elif name == "Ridge":
        base_low, base_high = 1e-6, 1e4
        if round_idx == 1 or best_params_prev is None or "alpha" not in best_params_prev:
            low, high = base_low, base_high
        else:
            low, high = log_range_around(
                best_params_prev["alpha"], base_low, base_high,
                factor_round=10 if round_idx == 2 else 3
            )
        alpha = trial.suggest_float("alpha", low, high, log=True)
        return Ridge(alpha=alpha, random_state=RANDOM_STATE)

    elif name == "Lasso":
        base_low, base_high = 1e-6, 1e2
        if round_idx == 1 or best_params_prev is None or "alpha" not in best_params_prev:
            low, high = base_low, base_high
        else:
            low, high = log_range_around(
                best_params_prev["alpha"], base_low, base_high,
                factor_round=10 if round_idx == 2 else 3
            )
        alpha = trial.suggest_float("alpha", low, high, log=True)
        return Lasso(alpha=alpha, random_state=RANDOM_STATE, max_iter=8000)

    elif name == "ElasticNet":
        base_low, base_high = 1e-6, 1e2
        if round_idx == 1 or best_params_prev is None or "alpha" not in best_params_prev:
            low, high = base_low, base_high
        else:
            low, high = log_range_around(
                best_params_prev["alpha"], base_low, base_high,
                factor_round=10 if round_idx == 2 else 3
            )
        alpha = trial.suggest_float("alpha", low, high, log=True)

        if round_idx == 1 or best_params_prev is None or "l1_ratio" not in best_params_prev:
            l1_low, l1_high = 0.0, 1.0
        else:
            l1_low, l1_high = lin_range_around(
                best_params_prev["l1_ratio"], 0.0, 1.0, refine_frac
            )
        l1_ratio = trial.suggest_float("l1_ratio", l1_low, l1_high)

        return ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            random_state=RANDOM_STATE,
            max_iter=8000
        )

    # --- TREE MODELS ---
    elif name == "DecisionTree":
        base_md_low, base_md_high = 2, 20
        base_mss_low, base_mss_high = 2, 40
        base_msl_low, base_msl_high = 1, 40

        if round_idx == 1 or best_params_prev is None or "max_depth" not in best_params_prev:
            md_low, md_high = base_md_low, base_md_high
        else:
            md_low, md_high = int_range_around(
                best_params_prev["max_depth"], base_md_low, base_md_high, refine_frac
            )

        if round_idx == 1 or best_params_prev is None or "min_samples_split" not in best_params_prev:
            mss_low, mss_high = base_mss_low, base_mss_high
        else:
            mss_low, mss_high = int_range_around(
                best_params_prev["min_samples_split"], base_mss_low, base_mss_high, refine_frac
            )

        if round_idx == 1 or best_params_prev is None or "min_samples_leaf" not in best_params_prev:
            msl_low, msl_high = base_msl_low, base_msl_high
        else:
            msl_low, msl_high = int_range_around(
                best_params_prev["min_samples_leaf"], base_msl_low, base_msl_high, refine_frac
            )

        max_depth = trial.suggest_int("max_depth", md_low, md_high)
        min_split = trial.suggest_int("min_samples_split", mss_low, mss_high)
        min_leaf = trial.suggest_int("min_samples_leaf", msl_low, msl_high)

        return DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_split,
            min_samples_leaf=min_leaf,
            random_state=RANDOM_STATE
        )

    elif name == "RF":
        base_ne_low, base_ne_high = 50, 800
        base_md_low, base_md_high = 2, 20
        base_mss_low, base_mss_high = 2, 40
        base_msl_low, base_msl_high = 1, 40

        if round_idx == 1 or best_params_prev is None or "n_estimators" not in best_params_prev:
            ne_low, ne_high = base_ne_low, base_ne_high
        else:
            ne_low, ne_high = int_range_around(
                best_params_prev["n_estimators"], base_ne_low, base_ne_high, refine_frac
            )

        if round_idx == 1 or best_params_prev is None or "max_depth" not in best_params_prev:
            md_low, md_high = base_md_low, base_md_high
        else:
            md_low, md_high = int_range_around(
                best_params_prev["max_depth"], base_md_low, base_md_high, refine_frac
            )

        if round_idx == 1 or best_params_prev is None or "min_samples_split" not in best_params_prev:
            mss_low, mss_high = base_mss_low, base_mss_high
        else:
            mss_low, mss_high = int_range_around(
                best_params_prev["min_samples_split"], base_mss_low, base_mss_high, refine_frac
            )

        if round_idx == 1 or best_params_prev is None or "min_samples_leaf" not in best_params_prev:
            msl_low, msl_high = base_msl_low, base_msl_high
        else:
            msl_low, msl_high = int_range_around(
                best_params_prev["min_samples_leaf"], base_msl_low, base_msl_high, refine_frac
            )

        # NUEVO: max_features para RF
        max_features_choices = ["sqrt", "log2", None]
        if round_idx == 1 or best_params_prev is None or "max_features" not in best_params_prev:
            max_features = trial.suggest_categorical("max_features", max_features_choices)
        else:
            # usamos las mismas opciones categóricas; no hace falta refinar range
            max_features = trial.suggest_categorical("max_features", max_features_choices)

        n_estimators = trial.suggest_int("n_estimators", ne_low, ne_high)
        max_depth = trial.suggest_int("max_depth", md_low, md_high)
        min_split = trial.suggest_int("min_samples_split", mss_low, mss_high)
        min_leaf = trial.suggest_int("min_samples_leaf", msl_low, msl_high)

        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_split,
            min_samples_leaf=min_leaf,
            max_features=max_features,
            n_jobs=-1,
            random_state=RANDOM_STATE
        )

    elif name == "ExtraTrees":
        base_ne_low, base_ne_high = 50, 800
        base_md_low, base_md_high = 2, 20
        base_mss_low, base_mss_high = 2, 40
        base_msl_low, base_msl_high = 1, 40

        if round_idx == 1 or best_params_prev is None or "n_estimators" not in best_params_prev:
            ne_low, ne_high = base_ne_low, base_ne_high
        else:
            ne_low, ne_high = int_range_around(
                best_params_prev["n_estimators"], base_ne_low, base_ne_high, refine_frac
            )

        if round_idx == 1 or best_params_prev is None or "max_depth" not in best_params_prev:
            md_low, md_high = base_md_low, base_md_high
        else:
            md_low, md_high = int_range_around(
                best_params_prev["max_depth"], base_md_low, base_md_high, refine_frac
            )

        if round_idx == 1 or best_params_prev is None or "min_samples_split" not in best_params_prev:
            mss_low, mss_high = base_mss_low, base_mss_high
        else:
            mss_low, mss_high = int_range_around(
                best_params_prev["min_samples_split"], base_mss_low, base_mss_high, refine_frac
            )

        if round_idx == 1 or best_params_prev is None or "min_samples_leaf" not in best_params_prev:
            msl_low, msl_high = base_msl_low, base_msl_high
        else:
            msl_low, msl_high = int_range_around(
                best_params_prev["min_samples_leaf"], base_msl_low, base_msl_high, refine_frac
            )

        # NUEVO: max_features para ExtraTrees
        max_features_choices = ["sqrt", "log2", None]
        if round_idx == 1 or best_params_prev is None or "max_features" not in best_params_prev:
            max_features = trial.suggest_categorical("max_features", max_features_choices)
        else:
            max_features = trial.suggest_categorical("max_features", max_features_choices)

        n_estimators = trial.suggest_int("n_estimators", ne_low, ne_high)
        max_depth = trial.suggest_int("max_depth", md_low, md_high)
        min_split = trial.suggest_int("min_samples_split", mss_low, mss_high)
        min_leaf = trial.suggest_int("min_samples_leaf", msl_low, msl_high)

        return ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_split,
            min_samples_leaf=min_leaf,
            max_features=max_features,
            n_jobs=-1,
            random_state=RANDOM_STATE
        )

    elif name == "AdaBoost":
        base_ne_low, base_ne_high = 50, 600
        base_lr_low, base_lr_high = 1e-3, 2.0

        if round_idx == 1 or best_params_prev is None or "n_estimators" not in best_params_prev:
            ne_low, ne_high = base_ne_low, base_ne_high
        else:
            ne_low, ne_high = int_range_around(
                best_params_prev["n_estimators"], base_ne_low, base_ne_high, refine_frac
            )

        if round_idx == 1 or best_params_prev is None or "learning_rate" not in best_params_prev:
            lr_low, lr_high = base_lr_low, base_lr_high
        else:
            lr_low, lr_high = log_range_around(
                best_params_prev["learning_rate"], base_lr_low, base_lr_high,
                factor_round=10 if round_idx == 2 else 3
            )

        n_estimators = trial.suggest_int("n_estimators", ne_low, ne_high)
        lr = trial.suggest_float("learning_rate", lr_low, lr_high, log=True)

        return AdaBoostRegressor(
            n_estimators=n_estimators,
            learning_rate=lr,
            random_state=RANDOM_STATE
        )

    elif name == "GBRT":
        base_ne_low, base_ne_high = 50, 600
        base_lr_low, base_lr_high = 1e-3, 0.5
        base_md_low, base_md_high = 2, 10
        base_msl_low, base_msl_high = 1, 40
        base_subs_low, base_subs_high = 0.5, 1.0

        if round_idx == 1 or best_params_prev is None or "n_estimators" not in best_params_prev:
            ne_low, ne_high = base_ne_low, base_ne_high
        else:
            ne_low, ne_high = int_range_around(
                best_params_prev["n_estimators"], base_ne_low, base_ne_high, refine_frac
            )

        if round_idx == 1 or best_params_prev is None or "learning_rate" not in best_params_prev:
            lr_low, lr_high = base_lr_low, base_lr_high
        else:
            lr_low, lr_high = log_range_around(
                best_params_prev["learning_rate"], base_lr_low, base_lr_high,
                factor_round=10 if round_idx == 2 else 3
            )

        if round_idx == 1 or best_params_prev is None or "max_depth" not in best_params_prev:
            md_low, md_high = base_md_low, base_md_high
        else:
            md_low, md_high = int_range_around(
                best_params_prev["max_depth"], base_md_low, base_md_high, refine_frac
            )

        if round_idx == 1 or best_params_prev is None or "min_samples_leaf" not in best_params_prev:
            msl_low, msl_high = base_msl_low, base_msl_high
        else:
            msl_low, msl_high = int_range_around(
                best_params_prev["min_samples_leaf"], base_msl_low, base_msl_high, refine_frac
            )

        if round_idx == 1 or best_params_prev is None or "subsample" not in best_params_prev:
            subs_low, subs_high = base_subs_low, base_subs_high
        else:
            subs_low, subs_high = lin_range_around(
                best_params_prev["subsample"], base_subs_low, base_subs_high, refine_frac
            )

        n_estimators = trial.suggest_int("n_estimators", ne_low, ne_high)
        lr = trial.suggest_float("learning_rate", lr_low, lr_high, log=True)
        max_depth = trial.suggest_int("max_depth", md_low, md_high)
        min_leaf = trial.suggest_int("min_samples_leaf", msl_low, msl_high)
        subsample = trial.suggest_float("subsample", subs_low, subs_high)

        return GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=lr,
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            subsample=subsample,
            random_state=RANDOM_STATE
        )

    elif name == "HGBR":
        base_md_low, base_md_high = 2, 20
        base_lr_low, base_lr_high = 1e-3, 0.5
        base_mi_low, base_mi_high = 100, 800
        base_leaf_nodes_low, base_leaf_nodes_high = 15, 255
        base_l2_low, base_l2_high = 1e-5, 1.0

        if round_idx == 1 or best_params_prev is None or "max_depth" not in best_params_prev:
            md_low, md_high = base_md_low, base_md_high
        else:
            md_low, md_high = int_range_around(
                best_params_prev["max_depth"], base_md_low, base_md_high, refine_frac
            )

        if round_idx == 1 or best_params_prev is None or "lr" not in best_params_prev:
            lr_low, lr_high = base_lr_low, base_lr_high
        else:
            lr_low, lr_high = log_range_around(
                best_params_prev["lr"], base_lr_low, base_lr_high,
                factor_round=10 if round_idx == 2 else 3
            )

        if round_idx == 1 or best_params_prev is None or "max_iter" not in best_params_prev:
            mi_low, mi_high = base_mi_low, base_mi_high
        else:
            mi_low, mi_high = int_range_around(
                best_params_prev["max_iter"], base_mi_low, base_mi_high, refine_frac
            )

        if round_idx == 1 or best_params_prev is None or "max_leaf_nodes" not in best_params_prev:
            leaf_low, leaf_high = base_leaf_nodes_low, base_leaf_nodes_high
        else:
            leaf_low, leaf_high = int_range_around(
                best_params_prev["max_leaf_nodes"], base_leaf_nodes_low, base_leaf_nodes_high, refine_frac
            )

        if round_idx == 1 or best_params_prev is None or "l2_regularization" not in best_params_prev:
            l2_low, l2_high = base_l2_low, base_l2_high
        else:
            l2_low, l2_high = log_range_around(
                best_params_prev["l2_regularization"], base_l2_low, base_l2_high,
                factor_round=10 if round_idx == 2 else 3
            )

        max_depth = trial.suggest_int("max_depth", md_low, md_high)
        lr = trial.suggest_float("lr", lr_low, lr_high, log=True)
        max_iter = trial.suggest_int("max_iter", mi_low, mi_high)
        max_leaf_nodes = trial.suggest_int("max_leaf_nodes", leaf_low, leaf_high)
        l2_regularization = trial.suggest_float("l2_regularization", l2_low, l2_high, log=True)

        return HistGradientBoostingRegressor(
            max_depth=max_depth,
            learning_rate=lr,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            l2_regularization=l2_regularization,
            random_state=RANDOM_STATE
        )

    # --- SVR ---
    elif name == "SVR":
        base_C_low, base_C_high = 1e-3, 1e4
        base_eps_low, base_eps_high = 1e-4, 1.0
        base_gam_low, base_gam_high = 1e-5, 5.0

        if round_idx == 1 or best_params_prev is None or "C" not in best_params_prev:
            C_low, C_high = base_C_low, base_C_high
        else:
            C_low, C_high = log_range_around(
                best_params_prev["C"], base_C_low, base_C_high,
                factor_round=30 if round_idx == 2 else 10
            )

        if round_idx == 1 or best_params_prev is None or "epsilon" not in best_params_prev:
            eps_low, eps_high = base_eps_low, base_eps_high
        else:
            eps_low, eps_high = log_range_around(
                best_params_prev["epsilon"], base_eps_low, base_eps_high,
                factor_round=30 if round_idx == 2 else 10
            )

        if round_idx == 1 or best_params_prev is None or "gamma" not in best_params_prev:
            gam_low, gam_high = base_gam_low, base_gam_high
        else:
            gam_low, gam_high = log_range_around(
                best_params_prev["gamma"], base_gam_low, base_gam_high,
                factor_round=30 if round_idx == 2 else 10
            )

        C_val = trial.suggest_float("C", C_low, C_high, log=True)
        eps = trial.suggest_float("epsilon", eps_low, eps_high, log=True)
        gamma = trial.suggest_float("gamma", gam_low, gam_high, log=True)

        return SVR(C=C_val, epsilon=eps, gamma=gamma, kernel="rbf")

    # --- KNN ---
    elif name == "KNN":
        base_nn_low, base_nn_high = 1, 80

        if round_idx == 1 or best_params_prev is None or "n_neighbors" not in best_params_prev:
            nn_low, nn_high = base_nn_low, base_nn_high
        else:
            nn_low, nn_high = int_range_around(
                best_params_prev["n_neighbors"], base_nn_low, base_nn_high, refine_frac
            )

        n_neighbors = trial.suggest_int("n_neighbors", nn_low, nn_high)
        weights = trial.suggest_categorical("weights", ["uniform", "distance"])
        p = trial.suggest_int("p", 1, 2)
        return KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=weights,
            p=p,
            n_jobs=-1
        )

    # --- MLP ---
    elif name == "MLP":
        hidden_choices = [
            (16,), (32,), (64,), (128,),
            (32, 32), (64, 32), (128, 64)
        ]
        if round_idx == 1 or best_params_prev is None or "hidden" not in best_params_prev:
            hidden = trial.suggest_categorical("hidden", hidden_choices)
        else:
            # keep same global choices; refinement for categorical is not needed
            hidden = trial.suggest_categorical("hidden", hidden_choices)

        base_alpha_low, base_alpha_high = 1e-7, 1e-1
        base_lr_low, base_lr_high = 1e-5, 1e-2

        if round_idx == 1 or best_params_prev is None or "alpha" not in best_params_prev:
            alpha_low, alpha_high = base_alpha_low, base_alpha_high
        else:
            alpha_low, alpha_high = log_range_around(
                best_params_prev["alpha"],
                base_alpha_low, base_alpha_high,
                factor_round=30 if round_idx == 2 else 10
            )

        if round_idx == 1 or best_params_prev is None or "lr_init" not in best_params_prev:
            lr_low, lr_high = base_lr_low, base_lr_high
        else:
            lr_low, lr_high = log_range_around(
                best_params_prev["lr_init"],
                base_lr_low, base_lr_high,
                factor_round=30 if round_idx == 2 else 10
            )

        alpha = trial.suggest_float("alpha", alpha_low, alpha_high, log=True)
        lr_init = trial.suggest_float("lr_init", lr_low, lr_high, log=True)

        return MLPRegressor(
            hidden_layer_sizes=hidden,
            alpha=alpha,
            learning_rate_init=lr_init,
            max_iter=3000,
            random_state=RANDOM_STATE
        )

    # --- Bayesian Ridge ---
    elif name == "BayesianRidge":
        base_low, base_high = 1e-9, 1e-2

        if round_idx == 1 or best_params_prev is None:
            a1_low, a1_high = base_low, base_high
            a2_low, a2_high = base_low, base_high
            l1_low, l1_high = base_low, base_high
            l2_low, l2_high = base_low, base_high
        else:
            a1_low, a1_high = log_range_around(
                best_params_prev.get("alpha_1", 1e-6),
                base_low, base_high,
                factor_round=10 if round_idx == 2 else 3
            )
            a2_low, a2_high = log_range_around(
                best_params_prev.get("alpha_2", 1e-6),
                base_low, base_high,
                factor_round=10 if round_idx == 2 else 3
            )
            l1_low, l1_high = log_range_around(
                best_params_prev.get("lambda_1", 1e-6),
                base_low, base_high,
                factor_round=10 if round_idx == 2 else 3
            )
            l2_low, l2_high = log_range_around(
                best_params_prev.get("lambda_2", 1e-6),
                base_low, base_high,
                factor_round=10 if round_idx == 2 else 3
            )

        alpha_1 = trial.suggest_float("alpha_1", a1_low, a1_high, log=True)
        alpha_2 = trial.suggest_float("alpha_2", a2_low, a2_high, log=True)
        lambda_1 = trial.suggest_float("lambda_1", l1_low, l1_high, log=True)
        lambda_2 = trial.suggest_float("lambda_2", l2_low, l2_high, log=True)

        return BayesianRidge(
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            lambda_1=lambda_1,
            lambda_2=lambda_2
        )

    # --- GPR(with adjusted kernel bounds) ---
    elif name == "GPR":
        base_ls_low, base_ls_high = 0.1, 20.0
        base_c_low, base_c_high = 0.1, 20.0
        base_alpha_low, base_alpha_high = 1e-6, 1e-1

        if round_idx == 1 or best_params_prev is None or "length_scale" not in best_params_prev:
            ls_low, ls_high = base_ls_low, base_ls_high
        else:
            ls_low, ls_high = log_range_around(
                best_params_prev["length_scale"], base_ls_low, base_ls_high,
                factor_round=10 if round_idx == 2 else 3
            )

        if round_idx == 1 or best_params_prev is None or "constant" not in best_params_prev:
            c_low, c_high = base_c_low, base_c_high
        else:
            c_low, c_high = log_range_around(
                best_params_prev["constant"], base_c_low, base_c_high,
                factor_round=10 if round_idx == 2 else 3
            )

        if round_idx == 1 or best_params_prev is None or "alpha" not in best_params_prev:
            a_low, a_high = base_alpha_low, base_alpha_high
        else:
            a_low, a_high = log_range_around(
                best_params_prev["alpha"], base_alpha_low, base_alpha_high,
                factor_round=10 if round_idx == 2 else 3
            )

        length_scale = trial.suggest_float("length_scale", ls_low, ls_high, log=True)
        constant = trial.suggest_float("constant", c_low, c_high, log=True)
        alpha = trial.suggest_float("alpha", a_low, a_high, log=True)

        kernel = ConstantKernel(
            constant_value=constant,
            constant_value_bounds=(1e-4, 1e4)
        ) * RBF(
            length_scale=length_scale,
            length_scale_bounds=(1e-6, 1e3)
        )

        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            normalize_y=True,
            random_state=RANDOM_STATE
        )

    # --- XGBoost ---
    elif name == "XGB":
        base_ne_low, base_ne_high = 100, 900
        base_md_low, base_md_high = 2, 12
        base_lr_low, base_lr_high = 0.005, 0.5
        base_subs_low, base_subs_high = 0.5, 1.0
        base_cols_low, base_cols_high = 0.5, 1.0
        base_lam_low, base_lam_high = 1e-4, 50.0
        base_alp_low, base_alp_high = 1e-4, 50.0

        if round_idx == 1 or best_params_prev is None or "n_estimators" not in best_params_prev:
            ne_low, ne_high = base_ne_low, base_ne_high
        else:
            ne_low, ne_high = int_range_around(
                best_params_prev["n_estimators"], base_ne_low, base_ne_high, refine_frac
            )

        if round_idx == 1 or best_params_prev is None or "max_depth" not in best_params_prev:
            md_low, md_high = base_md_low, base_md_high
        else:
            md_low, md_high = int_range_around(
                best_params_prev["max_depth"], base_md_low, base_md_high, refine_frac
            )

        if round_idx == 1 or best_params_prev is None or "learning_rate" not in best_params_prev:
            lr_low, lr_high = base_lr_low, base_lr_high
        else:
            lr_low, lr_high = log_range_around(
                best_params_prev["learning_rate"], base_lr_low, base_lr_high,
                factor_round=10 if round_idx == 2 else 3
            )

        if round_idx == 1 or best_params_prev is None or "subsample" not in best_params_prev:
            subs_low, subs_high = base_subs_low, base_subs_high
        else:
            subs_low, subs_high = lin_range_around(
                best_params_prev["subsample"], base_subs_low, base_subs_high, refine_frac
            )

        if round_idx == 1 or best_params_prev is None or "colsample_bytree" not in best_params_prev:
            cols_low, cols_high = base_cols_low, base_cols_high
        else:
            cols_low, cols_high = lin_range_around(
                best_params_prev["colsample_bytree"], base_cols_low, base_cols_high, refine_frac
            )

        if round_idx == 1 or best_params_prev is None or "reg_lambda" not in best_params_prev:
            lam_low, lam_high = base_lam_low, base_lam_high
        else:
            lam_low, lam_high = log_range_around(
                best_params_prev["reg_lambda"], base_lam_low, base_lam_high,
                factor_round=10 if round_idx == 2 else 3
            )

        if round_idx == 1 or best_params_prev is None or "reg_alpha" not in best_params_prev:
            alp_low, alp_high = base_alp_low, base_alp_high
        else:
            alp_low, alp_high = log_range_around(
                best_params_prev["reg_alpha"], base_alp_low, base_alp_high,
                factor_round=10 if round_idx == 2 else 3
            )

        n_estimators = trial.suggest_int("n_estimators", ne_low, ne_high)
        max_depth = trial.suggest_int("max_depth", md_low, md_high)
        lr = trial.suggest_float("learning_rate", lr_low, lr_high, log=True)
        subsample = trial.suggest_float("subsample", subs_low, subs_high)
        colsample_bytree = trial.suggest_float("colsample_bytree", cols_low, cols_high)
        reg_lambda = trial.suggest_float("reg_lambda", lam_low, lam_high, log=True)
        reg_alpha = trial.suggest_float("reg_alpha", alp_low, alp_high, log=True)

        return XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=lr,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    else:
        raise ValueError(f"Unknown model: {name}")



def create_model_from_params(params: dict, name: str):
    def g(k, default):
        return params.get(k, default)

    if name == "Linear":
        return LinearRegression(n_jobs=-1)

    elif name == "Ridge":
        return Ridge(alpha=g("alpha", 1.0), random_state=RANDOM_STATE)

    elif name == "Lasso":
        return Lasso(
            alpha=g("alpha", 0.1),
            random_state=RANDOM_STATE,
            max_iter=8000
        )

    elif name == "ElasticNet":
        return ElasticNet(
            alpha=g("alpha", 0.1),
            l1_ratio=g("l1_ratio", 0.5),
            random_state=RANDOM_STATE,
            max_iter=8000
        )

    elif name == "DecisionTree":
        return DecisionTreeRegressor(
            max_depth=g("max_depth", 5),
            min_samples_split=g("min_samples_split", 2),
            min_samples_leaf=g("min_samples_leaf", 1),
            random_state=RANDOM_STATE
        )

    elif name == "RF":
        return RandomForestRegressor(
            n_estimators=g("n_estimators", 200),
            max_depth=g("max_depth", 10),
            min_samples_split=g("min_samples_split", 2),
            min_samples_leaf=g("min_samples_leaf", 1),
            max_features=g("max_features", "sqrt"),
            n_jobs=-1,
            random_state=RANDOM_STATE
        )

    elif name == "ExtraTrees":
        return ExtraTreesRegressor(
            n_estimators=g("n_estimators", 200),
            max_depth=g("max_depth", 10),
            min_samples_split=g("min_samples_split", 2),
            min_samples_leaf=g("min_samples_leaf", 1),
            max_features=g("max_features", "sqrt"),
            n_jobs=-1,
            random_state=RANDOM_STATE
        )

    elif name == "AdaBoost":
        return AdaBoostRegressor(
            n_estimators=g("n_estimators", 100),
            learning_rate=g("learning_rate", 0.1),
            random_state=RANDOM_STATE
        )

    elif name == "GBRT":
        return GradientBoostingRegressor(
            n_estimators=g("n_estimators", 100),
            learning_rate=g("learning_rate", 0.1),
            max_depth=g("max_depth", 3),
            min_samples_leaf=g("min_samples_leaf", 1),
            subsample=g("subsample", 1.0),
            random_state=RANDOM_STATE
        )

    elif name == "HGBR":
        return HistGradientBoostingRegressor(
            max_depth=g("max_depth", 10),
            learning_rate=g("lr", 0.1),
            max_iter=g("max_iter", 200),
            max_leaf_nodes=g("max_leaf_nodes", 31),
            l2_regularization=g("l2_regularization", 0.0),
            random_state=RANDOM_STATE
        )

    elif name == "SVR":
        return SVR(
            C=g("C", 1.0),
            epsilon=g("epsilon", 0.1),
            gamma=g("gamma", 0.1),
            kernel="rbf"
        )

    elif name == "KNN":
        return KNeighborsRegressor(
            n_neighbors=g("n_neighbors", 5),
            weights=g("weights", "uniform"),
            p=g("p", 2),
            n_jobs=-1
        )

    elif name == "MLP":
        hidden_map = {
            (16,): (16,),
            (32,): (32,),
            (64,): (64,),
            (128,): (128,),
            (32, 32): (32, 32),
            (64, 32): (64, 32),
            (128, 64): (128, 64),
        }
        hidden = g("hidden", (64,))
        hidden = hidden_map.get(tuple(hidden), tuple(hidden))

        return MLPRegressor(
            hidden_layer_sizes=hidden,
            alpha=g("alpha", 1e-3),
            learning_rate_init=g("lr_init", 1e-3),
            max_iter=3000,
            random_state=RANDOM_STATE
        )

    elif name == "BayesianRidge":
        return BayesianRidge(
            alpha_1=g("alpha_1", 1e-6),
            alpha_2=g("alpha_2", 1e-6),
            lambda_1=g("lambda_1", 1e-6),
            lambda_2=g("lambda_2", 1e-6)
        )

    elif name == "GPR":
        length = g("length_scale", 1.0)
        constant = g("constant", 1.0)
        kernel = ConstantKernel(
            constant_value=constant,
            constant_value_bounds=(1e-4, 1e4)
        ) * RBF(
            length_scale=length,
            length_scale_bounds=(1e-6, 1e3)
        )
        alpha = g("alpha", 1e-3)
        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            normalize_y=True,
            random_state=RANDOM_STATE
        )

    elif name == "XGB":
        return XGBRegressor(
            n_estimators=g("n_estimators", 200),
            max_depth=g("max_depth", 4),
            learning_rate=g("learning_rate", 0.1),
            subsample=g("subsample", 1.0),
            colsample_bytree=g("colsample_bytree", 1.0),
            reg_lambda=g("reg_lambda", 1.0),
            reg_alpha=g("reg_alpha", 0.0),
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    else:
        raise ValueError(f"Unknown model: {name}")




MODEL_LIST = [
    "MLP","Linear", "Ridge", "Lasso", "ElasticNet",
    "DecisionTree", "AdaBoost", "RF", "ExtraTrees",
    "GBRT", "XGB", "SVR", "KNN",
    "BayesianRidge", "GPR"
]
#

# ===============================================================
# 6. OBJECTIVE (MAXIMIZE C-INDEX) + CV METRICS
# ===============================================================

N_SPLITS = 10
N_TRIALS = 50
CPU_COUNT = multiprocessing.cpu_count()


def make_objective(model_name, X_train, y_train, round_idx, best_params_prev):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        model = create_model_from_trial(trial, model_name, round_idx, best_params_prev)

        cidx_values = []

        for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)

            metrics = compute_metrics(y_val, y_pred)
            cidx = metrics["cindex"]
            cidx_values.append(cidx)

            trial.report(cidx if not np.isnan(cidx) else -1.0, step=fold_idx)
            if trial.should_prune():
                raise TrialPruned()

        cidx_mean = float(np.nanmean(cidx_values))
        if np.isnan(cidx_mean):
            cidx_mean = -1.0
        return cidx_mean

    return objective


def crossval_metrics_from_params(model_name, params, X_train, y_train):
    """
    Recompute CV metrics (mean and std) for reporting.
    ALSO return per-fold metrics to build a fold-level CSV.
    """
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    metrics_per_fold = []  # solo dicts de métricas
    fold_details = []      # guarda (fold_idx, métricas) para CSV

    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_train), start=1):
        model = create_model_from_params(params, model_name)
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        m = compute_metrics(y_val, y_pred)

        metrics_per_fold.append(m)
        fold_details.append((fold_idx, m))

    keys = metrics_per_fold[0].keys()
    mean_metrics = {}
    std_metrics = {}

    for k in keys:
        vals = np.array([mm[k] for mm in metrics_per_fold], dtype=float)
        mean_metrics[k] = float(np.nanmean(vals))
        std_metrics[k] = float(np.nanstd(vals))

    # devolvemos también los detalles por fold
    return mean_metrics, std_metrics, fold_details



# ===============================================================
# 7. THREE ROUNDS (COARSE → FINE) + LOG CSV OF HYPERPARAMS
# ===============================================================

best_params_round = {
    1: {m: None for m in MODEL_LIST},
    2: {m: None for m in MODEL_LIST},
    3: {m: None for m in MODEL_LIST},
}

hyperparams_rows = []  # <-- here we will store hyperparams+performance per round

for round_idx in [1, 2, 3]:
    print("\n" + "=" * 60)
    print(f"STARTING ROUND {round_idx} ({'COARSE' if round_idx == 1 else 'REFINED'} SEARCH)")
    print("=" * 60)

    for model_name in MODEL_LIST:
        print("\n----------------------------------------------")
        print(f"Round {round_idx} - Optimizing model: {model_name}")
        print("----------------------------------------------")

        prev_params = None if round_idx == 1 else best_params_round[round_idx - 1][model_name]

        study = optuna.create_study(
            direction="maximize",
            study_name=f"{model_name}_round{round_idx}_study",
            pruner=optuna.pruners.MedianPruner()
        )

        objective = make_objective(model_name, X_train, y_train, round_idx, prev_params)

        study.optimize(
            objective,
            n_trials=N_TRIALS,
            n_jobs=CPU_COUNT
        )

        best_trial = study.best_trial
        best_params = best_trial.params
        best_params_round[round_idx][model_name] = best_params

        print(f"  Best C-index (CV) in round {round_idx} for {model_name}: {best_trial.value}")
        print(f"  Best params: {best_params}")

        # Log hyperparams and performance for DB CSV
        row = {
            "round": round_idx,
            "model": model_name,
            "best_cv_cindex": best_trial.value,
        }
        for p_name, p_val in best_params.items():
            row[f"param_{p_name}"] = p_val
        hyperparams_rows.append(row)

# Save hyperparams per round
df_hyper = pd.DataFrame(hyperparams_rows)
df_hyper.to_csv("hyperparams_per_round.csv", index=False)
print("\nSaved hyperparameters per round to hyperparams_per_round.csv")


# ===============================================================
# 8. FINAL EVALUATION (ROUND 3 PARAMS) + MODELS + METRICS CSV
# ===============================================================

results = []
final_models = {}
fold_rows = []   # ⬅ aquí guardaremos métricas por fold y modelo

for model_name in MODEL_LIST:
    print("\n==============================================")
    print(f"FINAL STAGE - Building model: {model_name}")
    print("==============================================")

    final_params = best_params_round[3][model_name]

    # CV metrics with final params (mean/std + per-fold)
    mean_cv, std_cv, fold_details = crossval_metrics_from_params(
        model_name, final_params, X_train, y_train
    )

    # guardar info por fold para CSV de diferencias críticas
    for fold_idx, m in fold_details:
        fold_row = {
            "model": model_name,
            "fold": fold_idx,
            "cv_mse": m["mse"],
            "cv_rmse": m["rmse"],
            "cv_mae": m["mae"],
            "cv_r2": m["r2"],
            "cv_spearman": m["spearman"],
            "cv_pearson": m["pearson"],
            "cv_cindex": m["cindex"],
            "cv_mape": m["mape"],
        }
        fold_rows.append(fold_row)

    # Train final model on all training data
    final_model = create_model_from_params(final_params, model_name)
    final_model.fit(X_train, y_train)

    # Evaluate on test
    y_pred_test = final_model.predict(X_test)
    test_metrics = compute_metrics(y_test, y_pred_test)

    print(f"Test metrics for {model_name}:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    # Save model
    model_path = f"models/best_round3_{model_name}.pkl"
    joblib.dump(final_model, model_path)

    row = {
        "model": model_name,

        # CV means/stds
        "cv_mse_mean": mean_cv["mse"],
        "cv_mse_std": std_cv["mse"],
        "cv_rmse_mean": mean_cv["rmse"],
        "cv_rmse_std": std_cv["rmse"],
        "cv_mae_mean": mean_cv["mae"],
        "cv_mae_std": std_cv["mae"],
        "cv_r2_mean": mean_cv["r2"],
        "cv_r2_std": std_cv["r2"],
        "cv_spearman_mean": mean_cv["spearman"],
        "cv_spearman_std": std_cv["spearman"],
        "cv_pearson_mean": mean_cv["pearson"],
        "cv_pearson_std": std_cv["pearson"],
        "cv_cindex_mean": mean_cv["cindex"],
        "cv_cindex_std": std_cv["cindex"],
        "cv_mape_mean": mean_cv["mape"],
        "cv_mape_std": std_cv["mape"],

        # Test metrics
        "test_mse": test_metrics["mse"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_r2": test_metrics["r2"],
        "test_spearman": test_metrics["spearman"],
        "test_pearson": test_metrics["pearson"],
        "test_cindex": test_metrics["cindex"],
        "test_mape": test_metrics["mape"],

        "model_path": model_path
    }

    results.append(row)
    final_models[model_name] = final_model

# CSV 2: resultados agregados (ya lo tenías)
df_results = pd.DataFrame(results)
df_results.to_csv("model_results_cindex_round3.csv", index=False)
print("\nSaved final results to model_results_cindex_round3.csv")

# CSV 3: métricas por fold para diferencias críticas
df_folds = pd.DataFrame(fold_rows)
df_folds.to_csv("fold_metrics_round3.csv", index=False)
print("Saved per-fold CV metrics to fold_metrics_round3.csv")


