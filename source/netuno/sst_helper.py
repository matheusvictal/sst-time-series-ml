import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import xarray as xr
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import shiftgrid

class SSTHelper:
    """ SST Helper functions 

    Use the following date format: YYYY-MM-DD
    """

    def load_dataset(filename: str) -> xr.Dataset:
        ds = xr.open_dataset(filename)
        return ds


    def load_dataframe(ds: xr.Dataset) -> pd.DataFrame:
        df = ds.to_dataframe()
        df = df.reset_index()
        df = df[df['nbnds'] == 0]
        return df

    def get_sst_series(df: pd.DataFrame, lat: int, lon: int) -> pd.DataFrame:
        sst_series = df[(df['lat'] == lat) & (df['lon'] == lon)].reset_index(drop=True)
        sst_series.set_index('time', inplace=True)
        sst_series.drop(['lat', 'lon'], axis=1, inplace=True)
        return sst_series

    def get_subseries_by_index(df: pd.DataFrame, start_index: int, window: int) -> pd.DataFrame:
        return df.iloc[start_index:start_index+window]

    def get_subseries_by_date(df: pd.DataFrame, start_date: str, window: int) -> pd.DataFrame:
        return df.loc[date:][:window]

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