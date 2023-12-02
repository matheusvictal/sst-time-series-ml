import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
import netCDF4 as nc

# A função abaixo refebe um "ds" do tipo netCDF4, um indexador para a data de interesse e
# gera um mapa de calor com as SSTs na data de referência considerando-se a resolução 
# de "ds" (pontos na malha)
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