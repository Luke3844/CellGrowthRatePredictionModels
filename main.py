# =====================================================================
# MAIN PROGRAM
# =====================================================================

#============
## Imports ##
#============
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

import src.functions as sf  # This allows you to access functions from functions.py using sf.function_name()
from src.classes import Batch  # This allows you to import individual Classes from classes.py
from src.constants import *  # This imports all constants from constants.py

ph_optimal = sf.calculate_average(PH_LIMITS)
temperature_optimal = sf.calculate_average(TEMPERATURE_LIMITS)

batch = Batch(batch_id=1, ph_opt=ph_optimal, temperature_opt=temperature_optimal)
batch.display_info()

df = pd.read_csv(os.path.join("datasets", "dataset_cell_growth_4factor.csv"))

feature_cols = list(df.columns[1:])
target_col   = df.columns[0]

x_all = df.loc[:, feature_cols].to_numpy()
y_all = df.loc[:, target_col].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(
    x_all, y_all, test_size=0.2, random_state=0)

# ==============
# Feature subset generation
# ==============
def make_subsets(feature_cols, max_degree=5):
    n = len(feature_cols)
    out = []
    for i in range(n):
        out.append(([i], [feature_cols[i]], max_degree))
    for i in range(n):
        for j in range(i + 1, n):
            out.append(([i, j], [feature_cols[i], feature_cols[j]], max_degree))
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                out.append(([i, j, k],
                            [feature_cols[i], feature_cols[j], feature_cols[k]],
                            max_degree))
    out.append((list(range(n)), feature_cols, max_degree))
    return out

# ===========================
# Equation formatter for table
# ============================
def format_equation(intercept, coefs, powers, subset_names):
    parts = [f"{intercept:+.4f}"]
    for i in range(powers.shape[0]):
        term_parts = []
        for j in range(powers.shape[1]):
            name = subset_names[j].split(" [")[0]
            p    = int(powers[i, j])
            if p == 0:
                continue
            elif p == 1:
                term_parts.append(f"[{name}]")
            else:
                term_parts.append(f"[{name}]^{p}")
        term = " * ".join(term_parts)
        parts.append(f"{coefs[i]:+.4f}*{term}")
    return " ".join(parts)

# =================
# Polynomial models
# =================
poly_rows = []

for subset_idx, subset_names, max_degree in make_subsets(feature_cols, max_degree=5):
    x_sub_tr = x_train[:, subset_idx]
    x_sub_te = x_test[:,  subset_idx]
    feat_label = " + ".join(f.split(" [")[0] for f in subset_names)

    for degree in range(1, max_degree + 1):
        model = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=degree, include_bias=False),
            LinearRegression(),
        )

        cv = cross_validate(model, x_train[:, subset_idx], y_train, cv=5, scoring=("neg_root_mean_squared_error", "r2"), return_train_score=True)
        rmse_tr_cv = -cv["train_neg_root_mean_squared_error"]
        rmse_te_cv = -cv["test_neg_root_mean_squared_error"]
        r2_tr_cv   =  cv["train_r2"]
        r2_te_cv   =  cv["test_r2"]

        model.fit(x_sub_tr, y_train)
        rmse_tr = root_mean_squared_error(y_train, model.predict(x_sub_tr))
        rmse_te = root_mean_squared_error(y_test,  model.predict(x_sub_te))

        n_expanded = model["polynomialfeatures"].n_output_features_

        # Inline the fitted equation as a string
        powers    = model["polynomialfeatures"].powers_
        coefs     = model["linearregression"].coef_
        intercept = model["linearregression"].intercept_
        equation  = format_equation(intercept, coefs, powers, subset_names)

        poly_rows.append(dict(
            model            = "poly",
            features         = feat_label,
            degree           = degree,
            n_expanded       = n_expanded,
            equation         = equation,
            rmse_train       = rmse_tr,
            rmse_test        = rmse_te,
            rmse_train_cv    = rmse_tr_cv.mean(),
            rmse_train_cv_sd = rmse_tr_cv.std(),
            rmse_test_cv     = rmse_te_cv.mean(),
            rmse_test_cv_sd  = rmse_te_cv.std(),
            r2_train_cv      = r2_tr_cv.mean(),
            r2_train_cv_sd   = r2_tr_cv.std(),
            r2_test_cv       = r2_te_cv.mean(),
            r2_test_cv_sd    = r2_te_cv.std(),
        ))

# ==========
# kNN models
# ==========
knn_rows = []

for k in range(1, 21):
    for weights in ["uniform", "distance"]:
        model = make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=k, weights=weights),
        )

        cv = cross_validate(model, x_train, y_train, cv=5,
                            scoring=("neg_root_mean_squared_error", "r2"),
                            return_train_score=True)
        rmse_tr_cv = -cv["train_neg_root_mean_squared_error"]
        rmse_te_cv = -cv["test_neg_root_mean_squared_error"]
        r2_tr_cv   =  cv["train_r2"]
        r2_te_cv   =  cv["test_r2"]

        model.fit(x_train, y_train)
        rmse_tr = root_mean_squared_error(y_train, model.predict(x_train))
        rmse_te = root_mean_squared_error(y_test,  model.predict(x_test))

        knn_rows.append(dict(
            model            = "knn",
            features         = "all",
            degree           = np.nan,
            n_expanded       = x_all.shape[1],
            equation         = "",   # kNN has no closed-form equation
            rmse_train       = rmse_tr,
            rmse_test        = rmse_te,
            rmse_train_cv    = rmse_tr_cv.mean(),
            rmse_train_cv_sd = rmse_tr_cv.std(),
            rmse_test_cv     = rmse_te_cv.mean(),
            rmse_test_cv_sd  = rmse_te_cv.std(),
            r2_train_cv      = r2_tr_cv.mean(),
            r2_train_cv_sd   = r2_tr_cv.std(),
            r2_test_cv       = r2_te_cv.mean(),
            r2_test_cv_sd    = r2_te_cv.std(),
            k                = k,
            weights          = weights,
        ))

# =====================-===
# Making Tables for Analysis
# =========================
os.makedirs("tables", exist_ok=True)

# 1. Full comparison — one row per configuration, with inline equation
results_df = pd.DataFrame(poly_rows + knn_rows)

column_order = ["model", "features", "degree", "k", "weights", "n_expanded",
                "equation",
                "rmse_train", "rmse_test",
                "rmse_train_cv", "rmse_train_cv_sd",
                "rmse_test_cv",  "rmse_test_cv_sd",
                "r2_train_cv",   "r2_train_cv_sd",
                "r2_test_cv",    "r2_test_cv_sd"]
results_df = results_df[column_order]
results_df[results_df.select_dtypes("number").columns] = (
    results_df.select_dtypes("number").round(5)
)
results_df.to_csv(os.path.join("tables", "model_comparison.csv"), index=False)

# 2. Top-10 summary by CV test R²
top10_df = results_df.nlargest(10, "r2_test_cv")
top10_df[top10_df.select_dtypes("number").columns] = (
    top10_df.select_dtypes("number").round(4)
)
top10_df.to_csv(os.path.join("tables", "top10_by_r2.csv"), index=False)

# 3. Degree-1 polynomial equations only
linear_df = results_df[
    (results_df["model"] == "poly") & (results_df["degree"] == 1)
][["features", "equation", "rmse_test_cv", "r2_test_cv"]].copy()
linear_df[["rmse_test_cv", "r2_test_cv"]] = (
    linear_df[["rmse_test_cv", "r2_test_cv"]].round(5)
)
linear_df.to_csv(os.path.join("tables", "poly_linear_equations.csv"), index=False)

# 4. Degree-2 with squares but no cross-terms
quad_df = results_df[
    results_df["model"] == "poly_quadratic_no_interactions"
][["features", "n_expanded", "equation", "rmse_test_cv", "r2_test_cv"]].copy()
quad_df[["rmse_test_cv", "r2_test_cv"]] = (
    quad_df[["rmse_test_cv", "r2_test_cv"]].round(5)
)
quad_df.to_csv(os.path.join("tables", "poly_quadratic_no_interactions.csv"), index=False)