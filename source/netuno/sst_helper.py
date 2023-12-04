import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import xarray as xr
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import shiftgrid
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from math import sqrt
import seaborn as sns

class SSTHelper:
    """ SST Helper

    Use o seguinte formato de data: YYYY-MM-DD
    """

    f_inv = lambda x: x + 180
    f = lambda x: ((x+180) % 360) - 180
    
    def load_dataset(filename: str) -> xr.Dataset:
        ds = xr.open_dataset(filename)
        return ds


    def load_dataframe(ds: xr.Dataset) -> pd.DataFrame:
        df = ds.to_dataframe()
        df = df.reset_index()
        df = df[df['nbnds'] == 0]
        f = lambda x: ((x+180) % 360) - 180
        df['lon'] = df['lon'].copy().apply(f)
        return df

    def get_sst_series(df: pd.DataFrame, lat: int, lon: int) -> pd.DataFrame:
        sst_series = df[(df['lat'] == lat) & (df['lon'] == lon)].reset_index(drop=True)
        sst_series.set_index('time', inplace=True)
        sst_series.drop(['lat', 'lon'], axis=1, inplace=True)
        return sst_series

    def get_subseries_by_index(df: pd.DataFrame, start_index: int, window: int) -> pd.DataFrame:
        return df.iloc[start_index:start_index+window]

    def get_subseries_by_date(df: pd.DataFrame, start_date: str, window: int) -> pd.DataFrame:
        return df.loc[start_date:][:window]

    def plot_sst(df: pd.DataFrame, start_date: int, window: int):
        fig, ax = plt.subplots()
        subseries = SSTHelper.get_subseries_by_index(df, start_date, window)
        ax.plot(subseries.index, subseries['sst'], linewidth=2.0)
        plt.xticks(rotation=30)
        plt.show()

    def split_train_test(df: pd.DataFrame, split_date: str):
        train = df.loc[:split_date]
        test = df.loc[split_date:][1:]
        return train, test

    def rmse(y_true, y_pred) -> float:
        return sqrt(mean_squared_error(y_true, y_pred))

    def mape(y_true, y_pred) -> float:
        return mean_absolute_percentage_error(y_true, y_pred)

    def sst_basemap(ds, day):

        # Função para conversão da projeção de coordenadas padrão
        f = lambda x: ((x+180) % 360) - 180

        # definição de coordenads para o padrão Basemap
        lats = ds.variables['lat'][:]
        lons = ds.variables['lon'][:]
        sst = ds.variables['sst'][:]
        time = ds.variables['time'][:]
        lons = f(lons)
        ind = np.argsort(lons)
        lons = lons[ind]
        sst = sst[:, :, ind]
    
        # Projeção dos dados utilizando Basemap
        fig = plt.figure(num=None, figsize=(15, 15) ) 
        m = Basemap(projection='cyl', llcrnrlon=-180, llcrnrlat=-75, urcrnrlon=180, urcrnrlat=85, resolution='i')

        lon, lat = np.meshgrid(lons,lats)

        x, y = m(lon, lat)

        st = np.squeeze(sst[day,:,:])
        print_date = np.datetime_as_string(time[day].values, unit='M')
        cs = m.contourf(x, y , st, cmap='jet', levels = 300)
        m.drawcoastlines()
        m.drawmapboundary()
        m.drawcountries(linewidth=1, linestyle='solid', color='k' ) 
        plt.ylabel("Latitude", fontsize=15, labelpad=35)
        plt.xlabel("Longitude", fontsize=15, labelpad=20)
        cbar = m.colorbar(cs, location='right', pad="3%")
        cbar.set_label('SST (degC)', fontsize=13)
        plt.title(f'SST filled contour map for {print_date}', fontsize=15)
        plt.show()

    def get_sst_series_default(df: pd.DataFrame, lat: int, lon: int) -> pd.DataFrame:
        df = df[df['nbnds'] == 0]
        sst_series = df[(df['lat'] == lat) & (df['lon'] == lon)].reset_index(drop=True)
        sst_series.set_index('time', inplace=True)
        sst_series.drop(['nbnds', 'time_bnds'], axis=1, inplace=True)
        return sst_series

    def MinMaxScaler(X):
        X_std = (X - np.min(X)) / (np.max(X) - np.min(X))
        print(X_std)
        X_scaled = X_std * (2) - 1
        return X_scaled

    def default_plot(y_real, y_pred, point_name: str, ml_model_name: str):
        """
        point_name: [Indian Ocean, Atlantic Ocean, etc]
        ml_model_name: [SVR, SARIMA, LSTM]
        """
        # Imprime as previsoes para o conjunto de teste e os valores reais
        fig, ax = plt.subplots(figsize=(15, 2.5))
        sns.set(style="whitegrid")

        color_dict = {
            'SVR': 'deeppink',
            'SARIMA': 'olive',
            'LSTM': 'orangered'
        }

        try:
            color_to_use = color_dict[ml_model_name]
        except KeyError:
            color_to_use = 'gray'
        
        x_index = [
            '2022-01',
            '2022-02',
            '2022-03',
            '2022-04',
            '2022-05',
            '2022-06',
            '2022-07',
            '2022-08',
            '2022-09',
            '2022-10',
            '2022-11',
            '2022-12'
        ]

        df_lines = pd.DataFrame(
            {'Data': x_index, 
            'Actual SST': y_real,
            'Predicted SST': y_pred})

        df_lines.set_index('Data', inplace=True, drop=True)

        sns.lineplot(data=df_lines,
                    palette={'Actual SST': 'indigo', 'Predicted SST': color_to_use},
                    linewidth=1.5)

        plt.xticks(rotation=45)
        plt.title(f'SST prediction for {point_name}', fontsize=16)
        plt.xlabel('Time indicator (test set)', fontsize=12)
        plt.ylabel('SST', fontsize=12)
        plt.show()



    def sst_basemap_dot_ref(ds, day, lon_dot, lat_dot):

        # Função para conversão da projeção de coordenadas padrão
        f = lambda x: ((x+180) % 360) - 180

        # definição de coordenads para o padrão Basemap
        lats = ds.variables['lat'][:]
        lons = ds.variables['lon'][:]
        sst = ds.variables['sst'][:]
        time = ds.variables['time'][:]
        lons = f(lons)
        ind = np.argsort(lons)
        lons = lons[ind]
        sst = sst[:, :, ind]
    
        # Projeção dos dados utilizando Basemap
        fig = plt.figure(num=None, figsize=(15, 15) ) 
        m = Basemap(projection='cyl', llcrnrlon=-180, llcrnrlat=-75, urcrnrlon=180, urcrnrlat=85, resolution='i')

        lon, lat = np.meshgrid(lons,lats)

        x, y = m(lon, lat)
        
        st = np.squeeze(sst[day,:,:])
        print_date = np.datetime_as_string(time[day].values, unit='M')
        cs = m.contourf(x, y , st, cmap='jet', levels = 300)
 
        lonss, latt = m(lon_dot, lat_dot)
        m.plot(lonss, latt, 'go', markersize=18)

        m.drawcoastlines()
        m.drawmapboundary()
        m.drawcountries(linewidth=1, linestyle='solid', color='k' )


        plt.ylabel("Latitude", fontsize=15, labelpad=35)
        plt.xlabel("Longitude", fontsize=15, labelpad=20)
        cbar = m.colorbar(cs, location='right', pad="3%")
        cbar.set_label('SST (degC)', fontsize=13)
        plt.title(f'SST filled contour map for {print_date}', fontsize=15)


        plt.show()