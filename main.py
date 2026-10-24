# =====================================================================
# MAIN PROGRAM
# =====================================================================

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

df = pd.read_csv(os.path.join("datasets", "dataset_cell_growth_4factor.csv"))

ph_optimal = sf.calculate_average(PH_LIMITS)
temperature_optimal = sf.calculate_average(TEMPERATURE_LIMITS)

batch = Batch(batch_id=1, ph_opt=ph_optimal, temperature_opt=temperature_optimal)
batch.display_info()

features = df.columns[:-1]
targets = df.columns[-1]

x = df.loc[:, features].to_numpy()
y = df.loc[:, targets].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

model = make_pipeline(StandardScaler(), PolynomialFeatures(include_bias = False), LinearRegression())

param_grid = dict(polynomialfeatures__degree=[i for i in range(1, 11)])

model_gscv = GridSearchCV(model, param_grid = param_grid, cv = 4, scoring="neg_root_mean_squared_error", return_train_score=True)

model_gscv.fit(x_train, y_train)

df = pd.DataFrame(model_gscv.cv_results_)