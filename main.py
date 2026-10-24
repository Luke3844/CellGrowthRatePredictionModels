# =====================================================================
# MAIN PROGRAM
# =====================================================================

## Imports ##
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split
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

## Initial Setup ##

rng = np.random.default_rng(seed = 1)

features = df.columns[1:]
targets = df.columns[0]

x_all = df.loc[:, features].to_numpy()
y_all = df.loc[:, targets].to_numpy()

kwargs_gscv = dict(cv=4, scoring="neg_root_mean_squared_error", return_train_score=True)

param_grid_poly = dict(polynomialfeatures__degree=[i for i in range(1, 6)])

param_grid_knn = dict(kneighborsregressor__n_neighbors=[i for i in range(1, 10)],
                      kneighborsregressor__weights=["uniform", "distance"])

## Starting ##

lc_n_training_samples = []
lc_rmse_poly_train = []
lc_rmse_knn_train = []
lc_rmse_poly_test = []
lc_rmse_knn_test = []

for n_samples in range(38, 318, 20):
    idxs = rng.choice(len(x_all), size=n_samples, replace=False)
    x = x_all[idxs]
    y = y_all[idxs]

    #Splitting data for testing and training
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    #GSCV for our polynomial models
    model_poly = make_pipeline(StandardScaler(), PolynomialFeatures(), LinearRegression())
    model_gscv_poly = GridSearchCV(model_poly, param_grid=param_grid_poly, **kwargs_gscv)
    model_gscv_poly.fit(x_train, y_train)

    #GSCV for knn models
    model_knn = make_pipeline(StandardScaler(), KNeighborsRegressor())
    model_gscv_knn = GridSearchCV(model_knn, param_grid = param_grid_knn, **kwargs_gscv)
    model_gscv_knn.fit(x_train, y_train)

    #Training Computing
    rmse_poly_train = root_mean_squared_error(y_train, model_gscv_poly.predict(x_train))
    rmse_knn_train = root_mean_squared_error(y_train, model_gscv_knn.predict(x_train))
    rmse_poly_test = root_mean_squared_error(y_test, model_gscv_poly.predict(x_test))
    rmse_knn_test = root_mean_squared_error(y_test, model_gscv_knn.predict(x_test))

    #Pulling out "Best" Hyperparameters
    best_degree = model_gscv_poly.best_params_["polynomialfeatures__degree"]
    best_n_neighbors = model_gscv_knn.best_params_["kneighborsregressor__n_neighbors"]
    best_weight = model_gscv_knn.best_params_["kneighborsregressor__weights"]

    n_training_samples = x_train.shape[0]

    lc_n_training_samples.append(n_training_samples)
    lc_rmse_poly_train.append(rmse_poly_train)
    lc_rmse_knn_train.append(rmse_knn_train)
    lc_rmse_poly_test.append(rmse_poly_test)
    lc_rmse_knn_test.append(rmse_knn_test)