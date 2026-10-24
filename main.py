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
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split
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

features = df.columns[:-1]
targets = df.columns[-1]

x_all = df.loc[:, features].to_numpy()
y_all = df.loc[:, targets].to_numpy()

kwargs_gscv = dict(cv=4, scoring="neg_root_mean_squared_error", return_train_score=True)

param_grid_poly = dict(polynomialfeatures__degree=[i for i in range(1, 21)])

param_grid_knn = dict(kneighborsregressor__n_neighbors=[i for i in range(1, 21)],
                      kneighborsregressor__weights=["uniform", "distance"])

## Starting ##

lc_n_training_samples = []
lc_rmse_poly_train = []
lc_rmse_knn_train = []
lc_rmse_poly_test = []
lc_rmse_knn_test = []

