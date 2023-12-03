import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from dtw import *
from .sst_helper import SSTHelper

class SubserieDTW:
    """ Use this class to process information before using SVR """

    lat: int
    lon: int
    train_proportion: int
    window: float
    df: pd.DataFrame
    list_subseries: list

    def __init__(self, df, lat, lon, split_date: str, window: int = 48, forecast_horizon: int = 1):
        print("Initializing")
        print(f"Lat: {lat}, Lon: {lon}")
        self.lat = lat
        self.lon = lon
        self.window = window
        self.forecast_horizon = forecast_horizon
        self.df = df

        self.point_df = SSTHelper.get_sst_series(df, lat, lon)
        self.df_len = len(self.point_df)

        #          Split into train/test
        # |---------train---------main-|---test---|      
        self.train_df, self.test_df = SSTHelper.split_train_test(self.point_df, split_date)
        # self.train_df = self.point_df[:start-window].reset_index(drop=True)
        # self.test_df = self.point_df[start+window:].reset_index(drop=True)
        
        # Get main subserie
        self.main_subserie = SSTHelper.get_subseries_by_index(self.train_df, len(self.train_df) - window, window)
        self.train_df = self.train_df[:-window]
        print(f"Total length: {self.df_len}")
        print(len(self.train_df), len(self.test_df))
        print(f"Train/test proportion: {len(self.train_df)/self.df_len}/{len(self.test_df)/self.df_len}")

    
    def get_nearest_subseries(self):
        list_subseries = []
        np_main_subserie = np.array(self.main_subserie['sst'])
        for i in tqdm(range(0, len(self.train_df) - (2*self.window))):
            subserie_df = SSTHelper.get_subseries_by_index(self.train_df, i, self.window)
            np_subserie = np.array(subserie_df['sst'])
            np_next_subserie = np.array(SSTHelper.get_subseries_by_index(self.train_df, i+self.window, self.forecast_horizon)['sst'])
            try:
                if len(np_main_subserie) == len(np_subserie):
                    alignment = dtw(np_main_subserie, np_subserie, keep_internals=True)
                    subserie = {
                        'df': np_subserie,
                        'start': i,
                        'distance': alignment.distance,
                        'alignment': alignment,
                        'next': np_next_subserie
                    }

                    list_subseries.append(subserie)
            except ValueError as e:
                print(f"Got value error: {e}")
        self.list_subseries = sorted(list_subseries, key=lambda item: item['distance'], reverse=False)
        print(f"Obtained {len(self.list_subseries)} subseries")
        return self.list_subseries
    
    def print_nearest_subserie(self):
        nearest = self.list_subseries[0]
        print(f"DTW Distance: {nearest['distance']}")
        nearest['alignment'].plot(type="twoway",offset=-2)

    def get_train(self, top_n_series: int = 200):
        x_train = [i['df'] for i in self.list_subseries[:top_n_series]]
        y_train = [i['next'] for i in self.list_subseries[:top_n_series]]
        return np.array(x_train), np.array(y_train)

    def get_test(self):
        x_test = []
        y_test = []
        for i in (range(0, 60)):
            x_test.append(np.array(SSTHelper.get_subseries_by_index(self.test_df, i, self.window)['sst']))
            y_test.append(np.array(SSTHelper.get_subseries_by_index(self.test_df, i+self.window, self.forecast_horizon)['sst']))
        return np.array(x_test), np.array(y_test)

    def get_point_df(self):
        return self.point_df
    