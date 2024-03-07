import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from dtw import *
from .sst_helper import SSTHelper
from sklearn.preprocessing import MinMaxScaler
import random

RANDOM_SEED = 4
PERCENTAGE_DTW = 0.3

class SubserieDTW:
    """ Tem como utilidade realizar preprocessamento antes do SVR """

    lat: int
    lon: int
    train_proportion: int
    window: float
    df: pd.DataFrame
    list_subseries: list

    def __init__(self, df, lat, lon, split_date: str, window: int = 48, forecast_horizon: int = 1, verbose: bool = False):
        self.lat = lat
        self.lon = lon
        self.window = window
        self.forecast_horizon = forecast_horizon
        self.df = df
        self.verbose = verbose

        self.point_df = SSTHelper.get_sst_series(df, lat, lon)
        scaler_data = np.array(self.point_df['sst'])

        self.df_len = len(self.point_df)

        #          Split into train/test
        # |---------train---------main-|---test---|      
        self.train_df, self.test_df = SSTHelper.split_train_test(self.point_df, split_date)
        # self.train_df = self.point_df[:start-window].reset_index(drop=True)
        # self.test_df = self.point_df[start+window:].reset_index(drop=True)
        
        # Get main subserie
        self.main_subserie = SSTHelper.get_subseries_by_index(self.train_df, len(self.train_df) - window, window)
        self.np_main_subserie = np.array(self.main_subserie['sst'])
        self.train_df = self.train_df[:-window]

        if self.verbose:
            print(df)
            print("Initializing")
            print(f"Lat: {lat}, Lon: {lon}")
            print(scaler_data)
            print(f"Total length: {self.df_len}")
            print(len(self.train_df), len(self.test_df))
            print(f"Train/test proportion: {len(self.train_df)/self.df_len}/{len(self.test_df)/self.df_len}")

    def get_all_subseries(self):
        """
        Pega todas as subseries dada uma série
        """
        list_subseries = []

        start_index_list = list(range(0, len(self.train_df) - (2*self.window)))

        for i in start_index_list:
            subserie_df = SSTHelper.get_subseries_by_index(self.train_df, i, self.window)
            np_subserie = np.array(subserie_df['sst'])
            np_next_subserie = np.array(SSTHelper.get_subseries_by_index(self.train_df, i+self.window, self.forecast_horizon)['sst'])
            try:
                subserie = {
                    'df': np_subserie,
                    'start': i,
                    'next': np_next_subserie
                }

                list_subseries.append(subserie)
            except ValueError as e:
                print(f"Got value error: {e}")
        self.list_subseries = list_subseries
        if self.verbose:
            print(f"Obtained {len(self.list_subseries)} subseries")
        return self.list_subseries
    
    
    def get_nearest_subseries(self, series_sample_ratio: float = 0.3):
        """
        Calcula o DTW de uma porcentagem series_sample_ration de subseries aleatórias no treino 
        """
        self.get_all_subseries()
        
        sampled_nearest_subseries = random.sample(
            self.list_subseries, int(series_sample_ratio * len(self.list_subseries)))

        for i in range(len(sampled_nearest_subseries)):
            subserie = sampled_nearest_subseries[i]
            np_subserie = subserie['df']
            try:
                alignment = dtw(self.np_main_subserie, np_subserie, keep_internals=True)
                subserie['distance'] = alignment.distance
                subserie['alignment'] = alignment
                sampled_nearest_subseries[i] = subserie
            except ValueError as e:
                print(f"Got value error: {e}")
        sampled_nearest_subseries = sorted(
            sampled_nearest_subseries, key=lambda item: item['distance'], reverse=False)
        if self.verbose:
            print(f"Obtained {len(sampled_nearest_subseries)} subseries")
        self.list_subseries = sampled_nearest_subseries
        return sampled_nearest_subseries
    
    def print_nearest_subserie(self):
        nearest = self.list_subseries[0]
        if self.verbose:
            print(f"DTW Distance: {nearest['distance']}")
        nearest['alignment'].plot(type="twoway",offset=-2)

    def get_train(self, top_subseries_ratio: float = 0.5):
        """
        Retorna x, y de treino utilizando as top subseries_ratio% series encontradas em get_nearest_subseries()
        """
        top_n_series = int(len(self.list_subseries) * top_subseries_ratio) + 1
        x_train = [i['df'] for i in self.list_subseries[:top_n_series]]
        y_train = [i['next'] for i in self.list_subseries[:top_n_series]]
        return np.array(x_train), np.array(y_train)

    def get_test(self, y_size: int = None):
        if y_size is None:
            y_size = self.forecast_horizon
        x_test = np.array(self.main_subserie['sst']).reshape(1, -1) 
        y_test = np.array(self.test_df['sst'])[:y_size]
        if self.forecast_horizon > len(y_test):
            raise ValueError(
                f'forecast_horizon ({self.forecast_horizon}) ' 
                f'deve ser menor \n ou a proporção de teste '
                f'({len(self.test_df)/self.df_len}) deve aumentar!')
        return x_test, y_test

    def get_point_df(self):
        return self.point_df

    def get_main_subserie(self):
        return self.main_subserie


    