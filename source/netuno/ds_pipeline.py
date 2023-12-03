import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from dtw import *
from .subseries_dtw import SubserieDTW
import random
from sklearn.multioutput import MultiOutputRegressor

class DSPipeline:
    """
    Pipeline de DTW + SVR para predicao de séries temporais
    """


    def make_pipeline(
            self, lat: int, lon: int, split_date: str, window: int = 48, 
            forecast_horizon: int = 1, series_sample_ratio: float = 0.3,
            subseries_ratio: float = 0.5):

        self.subserie_dtw = SubserieDTW(df, -56, f_inv(-80), split_date)
        self.subserie_dtw.get_nearest_subseries(1)
        x_train, y_train = indi.get_train(1)
        x_test, y_test = indi.get_test()
        multi_model = MultiOutputRegressor(SGDRegressor())
        multi_model.fit(x_train, y_train)



