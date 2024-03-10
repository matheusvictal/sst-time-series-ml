import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from dtw import *
from .subseries_dtw import SubserieDTW
from .sst_helper import SSTHelper
import random
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import differential_evolution
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler



class DSPipeline:
    """
    Pipeline de DTW + SVR para predicao de séries temporais
    """

    def __init__(
            self, df, lat: int, lon: int, split_date: str, forecast_horizon: int = 1, regressor = None):
        self.df = df
        self.lat = lat
        self.lon = lon
        self.split_date = split_date
        self.forecast_horizon = forecast_horizon
        if regressor is None:
            self.regressor = SVR()
        else:
            self.regressor = regressor
            

    def make_pipeline(
            self, series_sample_ratio: float = 1.0,
            top_subseries_ratio: float = 0.5,
            window: int = 24):

        self.window = window
        self.subserie_dtw = SubserieDTW(
            self.df, self.lat, self.lon, self.split_date, 
            window=self.window, forecast_horizon=self.forecast_horizon)
        
        self.subserie_dtw.get_all_subseries()
        # self.subserie_dtw.get_nearest_subseries(series_sample_ratio)
        self.x_train, self.y_train = self.subserie_dtw.get_train(top_subseries_ratio)
        self.x_test, self.y_test = self.subserie_dtw.get_test()

        self.fit()

        return self.predict(self.x_train)


    def fit(self, x = None, y = None):
        if x is None:
            x = self.x_train
        if y is None:
            y = self.y_train 
        self.model = MultiOutputRegressor(self.regressor)
            # SGDRegressor(
            #     learning_rate='optimal', 
            #     early_stopping=True)
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def evaluate(self):
        self.y_test_pred = self.predict(self.x_test)[0]
        rmse = SSTHelper.rmse(self.y_test, self.y_test_pred)
        return rmse

    def optimize(self):
        boundaries = [
            (0.01, 1),
        ]
        
        func = lambda X: self.make_pipeline(top_subseries_ratio=X[0])
        self.opt_results = differential_evolution(func, boundaries, maxiter=3, popsize=1)

    def optimization_results(self):
        print('series_sample_ratio:', self.opt_results.x[0])
        # print('top_subseries_ratio:', self.opt_results.x[1])
        print("Best RMSE: ", self.opt_results.fun)
        return self.opt_results.x[0]


    def plot_predict(self, point_name: str, ml_model_name: str):
        SSTHelper.default_plot(self.y_test, self.y_test_pred, point_name, ml_model_name)
        
    def evaluate_mape(self):
        mape = SSTHelper.mape(self.y_test, self.y_test_pred)
        return mape


