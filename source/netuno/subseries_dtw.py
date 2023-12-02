import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from dtw import *

class SubseriesDTW:
    lat: int
    lon: int
    start: int
    window: float
    df: pd.DataFrame
    list_subseries: list

    def __init__(self, df, lat, lon, start: int, window: int = 48, predict_window: int = 1):
        print("Initializing")
        print(f"Lat: {lat}, Lon: {lon}")
        self.lat = lat
        self.lon = lon
        self.start = start
        self.window = window
        self.predict_window = predict_window
        self.df = df

        self.point_df = SSTUtil.get_sst_series(df, lat, lon)

        # Get main subseries
        self.main_subseries = SSTUtil.get_subseries(self.point_df, start, window)

        #          Split into train/test
        # |---------train--------|-main-|---test---|
        self.train_df = self.point_df[:start-window].reset_index(drop=True)
        self.test_df = self.point_df[start+window:].reset_index(drop=True)
        
        self.df_len = len(self.point_df)
        print(f"Total length: {self.df_len}")
        print(f"Train/test proportion: {len(self.train_df)/self.df_len}/{len(self.test_df)/self.df_len}")

    
    def get_nearest_subseries(self):
        list_subseries = []
        np_main_subseries = np.array(self.main_subseries['sst'])
        for i in tqdm(range(0, len(self.train_df) - (2*self.window))):
            subseries_df = SSTUtil.get_subseries(self.train_df, i, self.window)
            np_subseries = np.array(subseries_df['sst'])
            np_next_subseries = np.array(SSTUtil.get_subseries(self.train_df, i+self.window, self.predict_window)['sst'])
            try:
                if len(np_main_subseries) == len(np_subseries):
                    alignment = dtw(np_main_subseries, np_subseries, keep_internals=True)
                    subseries = {
                        'df': np_subseries,
                        'start': i,
                        'distance': alignment.distance,
                        'alignment': alignment,
                        'next': np_next_subseries
                    }

                    list_subseries.append(subseries)
            except ValueError as e:
                print(f"Got value error: {e}")
        self.list_subseries = sorted(list_subseries, key=lambda item: item['distance'], reverse=False)
        print(f"Obtained {len(self.list_subseries)} subseries")
        return self.list_subseries
    
    def print_nearest_subseries(self):
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
            x_test.append(np.array(SSTUtil.get_subseries(self.test_df, i, self.window)['sst']))
            y_test.append(np.array(SSTUtil.get_subseries(self.test_df, i+self.window, self.predict_window)['sst']))
        return np.array(x_test), np.array(y_test)

    def get_point_df(self):
        return self.point_df