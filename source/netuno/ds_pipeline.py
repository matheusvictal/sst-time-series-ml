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
from scipy.optimize import differential_evolution
from sklearn.model_selection import GridSearchCV


class DSPipeline:
    """
    Pipeline de DTW + SVR para predicao de séries temporais
    """

    def __init__(
            self, df, lat: int, lon: int, split_date: str, forecast_horizon: int = 1):
        self.df = df
        self.lat = lat
        self.lon = lon
        self.split_date = split_date
        self.forecast_horizon = forecast_horizon

    def make_pipeline(
            self, series_sample_ratio: float = 1.0,
            top_subseries_ratio: float = 0.5,
            window: int = 48):

        self.window = window
        self.subserie_dtw = SubserieDTW(
            self.df, self.lat, self.lon, self.split_date, 
            window=self.window, forecast_horizon=self.forecast_horizon)
        self.subserie_dtw.get_nearest_subseries(series_sample_ratio)
        self.x_train, self.y_train = self.subserie_dtw.get_train(top_subseries_ratio)
        self.x_test, self.y_test = self.subserie_dtw.get_test()

        self.fit()

        return self.evaluate()


    def fit(self): 
        self.model = MultiOutputRegressor(SGDRegressor())
        self.model.fit(self.x_train, self.y_train)

    def evaluate(self):
        self.y_test_pred = self.model.predict(self.x_test)[0]
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
        print('top_subseries_ratio:', self.opt_results.x[1])
        print("Best RMSE: ", self.opt_results.fun)
        return self.opt_results.x[0], self.opt_results.x[1]


    def plot_predict(self):
        plt.figure(figsize=(15,6))
        plt.plot(range(0, 48), self.x_test[0], color = 'blue', linewidth=2.0, alpha = 0.6)
        plt.plot(range(48, self.forecast_horizon+48), self.y_test, color = 'red', linewidth=2.0, alpha = 0.6)
        plt.plot(range(48, self.forecast_horizon+48), self.y_test_pred, color = 'green', linewidth=2.0, alpha = 0.6)

        plt.legend(['Real Serie', 'Expected', 'Predicted'])
        plt.xlabel('Timestamp')
        plt.title("Predicting future SST")
        plt.show()



