from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import numpy as np
from matplotlib import pyplot as plt
import scipy.io
import mat73



data_D=np.load("datos_finales_indicios_7param.npz")
D_shelf=data_D["D_shelf"]

data_point_ages=scipy.io.loadmat('Point_ages_xyzKocsisScotese_400.mat')

shelf_lonlatAge=data_point_ages['shelf_lonlatAge']
Point_timeslices=data_point_ages['Point_timeslices']
Point_timeslices=Point_timeslices[0]

data_indices=np.load('indices_por_anos_3_filt.npz')

indices_car=data_indices["indices_car"]
indices_med=data_indices["indices_med"]
indices_pac=data_indices["indices_pac"]

data_iso=np.load("isolineas.npz")
time=data_iso["time"]
#time_point=[0,15,25,50,105,120,130,150,160,180,195]
min_lat=data_iso["min_lat"]
max_lat=data_iso["max_lat"]

data_mask=mat73.loadmat('landShelfOceanMask_ContMargMaskKocsisScotese.mat')
landShelfOcean_Lat=data_mask['landShelfOcean_Lat']
landShelfOcean_Lon=data_mask['landShelfOcean_Lon']

#for t in range(shelf_lonlatAge.shape[1]):


fig=plt.figure(figsize=(15, 20))

gs = GridSpec(4, 11, figure=fig, height_ratios=[2,4,4,4], width_ratios=[2]*11, hspace=0.4, wspace=0.8)  # 4 filas, 4 columnas de referencia

ax1 = fig.add_subplot(gs[0, 0:3])  # Ocupa toda la primera fila

A = np.zeros((indices_pac.shape[1], shelf_lonlatAge.shape[1]))
mean_values = np.zeros(shelf_lonlatAge.shape[1])
year= np.full(shelf_lonlatAge.shape[1], np.nan)

for c in range(6):
    for a in range(2,D_shelf.shape[0]):
        for t in range(50,shelf_lonlatAge.shape[1]):
            ind_pac=indices_pac[t,:]
            ind_pac=ind_pac[~np.isnan(ind_pac)]
            D_pac=D_shelf[:,ind_pac.astype(int),:,c]
            A = np.zeros((len(ind_pac)))
            for i in range(len(ind_pac)):
                A[i] = D_pac[i,a, t]
            mean_values[t] = np.nanmedian(A)

            if not np.isnan(mean_values[t]):
                year[t] = Point_timeslices[t]

        # ⚠️ Solo después de llenar 'year', creamos el mask
        mask = ~np.isnan(year)

        ax1.plot(-year[mask], mean_values[mask], color='darkblue', alpha=0.8, linewidth=0.5, label='Media')
        
years_x=np.arange(0, 201, 50)

ax1.set_xticks(-years_x)
ax1.set_xticklabels([f"{int(y)}" for y in years_x])

mask = ~np.isnan(year)


ax1.set_xlabel('Time (Ma)', fontsize=12)
ax1.set_ylabel('Diversity', fontsize=12)

ax1.set_title('Pacific', fontsize=15)

ax2 = fig.add_subplot(gs[0, 4:7])  # Ocupa toda la primera fila

mean_values = np.zeros(shelf_lonlatAge.shape[1])
year= np.full(shelf_lonlatAge.shape[1], np.nan)
max_A=0

for c in range(6):
    for a in range(2,D_shelf.shape[0]):
        for t in range(50,shelf_lonlatAge.shape[1]):
            ind_car=indices_car[t,:]
            ind_car=ind_car[~np.isnan(ind_car)]
            D_car=D_shelf[:,ind_car.astype(int),:,c]
            A = np.zeros((len(ind_car)))
            if len(ind_car)>max_A:
                max_A=len(ind_car)
            for i in range(len(ind_car)):
                A[i] = D_car[i,a, t]

            mean_values[t] = np.nanmedian(A)

            if not np.isnan(mean_values[t]):
                year[t] = Point_timeslices[t]

        # ⚠️ Solo después de llenar 'year', creamos el mask
        mask = ~np.isnan(year)

        ax2.plot(-year[mask], mean_values[mask], color='red', alpha=0.8, linewidth=0.5, label='Media')

ax2.set_xticks(-years_x)
ax2.set_xticklabels([f"{int(y)}" for y in years_x])

ax2.set_xlabel('Time (Ma)', fontsize=12)
ax2.set_ylabel('Diversity', fontsize=12)

ax2.set_title('Caribbean', fontsize=15)

ax3 = fig.add_subplot(gs[0, 8:11])  # Ocupa toda la primera fila

A = np.zeros((indices_car.shape[1]))
mean_values = np.zeros(shelf_lonlatAge.shape[1])
year= np.full(shelf_lonlatAge.shape[1], np.nan)

max_A=0

for c in range(6):
    for a in range(2,D_shelf.shape[0]):
        for t in range(50,shelf_lonlatAge.shape[1]):
            ind_med=indices_med[t,:]
            ind_med=ind_med[~np.isnan(ind_med)]
            D_med=D_shelf[:,ind_med.astype(int),:,c]
            A = np.zeros((len(ind_med)))
            if len(ind_med)>max_A:
                max_A=len(ind_med)
            for i in range(len(ind_med)):
                A[i] = D_med[i,a, t]
            mean_values[t] = np.nanmedian(A)

            if not np.isnan(mean_values[t]):
                year[t] = Point_timeslices[t]

        # ⚠️ Solo después de llenar 'year', creamos el mask
        mask = ~np.isnan(year)

        ax3.plot(-year[mask], mean_values[mask], color='green', alpha=0.8, linewidth=0.5, label='Media')

ax3.set_xticks(-years_x)
ax3.set_xticklabels([f"{int(y)}" for y in years_x])

ax3.set_xlabel('Time (Ma)', fontsize=12)
ax3.set_ylabel('Diversity', fontsize=12)

ax3.set_title('Mediterranean', fontsize=15)

#time_point=[0,25,50,105,130,150,160,180,195]# Bien: 541(1), 390(13), 150(56), 2(76) --- Mal: 250(30), 
time_point=[195, 180, 160, 150, 130, 105, 50, 25, 0]
#time_iso=[0,26,52,107,131,149,168,178,196]
time_iso=[196,178,168,149,131,107,52,26,0]
x_exes=[1,1,1,2,2,2,3,3,3]
y_exes=[slice(0,3),slice(4,7),slice(8,11),slice(0,3),slice(4,7),slice(8,11),slice(0,3),slice(4,7),slice(8,11)]


for t in range(len(time_point)):

    ind=np.where(Point_timeslices==time_point[t])[0]
    ind2=np.where(time==time_iso[t])[0]

    ta=ind[0]
    ta2=ind2[0]

    ax = fig.add_subplot(gs[x_exes[t], y_exes[t]], projection=ccrs.Mollweide(central_longitude=0))

    lon_shelf = shelf_lonlatAge[:, ind[0], 0]  # (49688,)
    lat_shelf = shelf_lonlatAge[:, ind[0], 1] 

    coords_med = np.array([[shelf_lonlatAge[int(i), ta, 0], shelf_lonlatAge[int(i), ta, 1]] for i in indices_med[ta] if not np.isnan(i)])
    coords_pac = np.array([[shelf_lonlatAge[int(i), ta, 0], shelf_lonlatAge[int(i), ta, 1]] for i in indices_pac[ta] if not np.isnan(i)])
    coords_car = np.array([[shelf_lonlatAge[int(i), ta, 0], shelf_lonlatAge[int(i), ta, 1]] for i in indices_car[ta] if not np.isnan(i)])

    #print(ye[t], coords_pac.shape, coords_car.shape)

    scatter = ax.scatter(lon_shelf, lat_shelf, s=1, alpha=0.05, color='black', transform=ccrs.PlateCarree())

    highlight_1 = ax.scatter(coords_med[:,0], coords_med[:,1], s=1, alpha=1.0, color='green',  label='Mediterranean points', transform = ccrs.PlateCarree())
    highlight_2 = ax.scatter(coords_pac[:,0], coords_pac[:,1], s=1, alpha=1.0, color='darkblue',  label='Pacific points', transform = ccrs.PlateCarree())
    highlight_3 = ax.scatter(coords_car[:,0], coords_car[:,1], s=1, alpha=1.0, color='red',  label='Caribbean points', transform = ccrs.PlateCarree())

    ax.plot(landShelfOcean_Lon, min_lat[ta2,:], color='orange', linestyle='--', label='Isoline 25ºC', transform=ccrs.PlateCarree())
    ax.plot(landShelfOcean_Lon, max_lat[ta2,:], color='orange', linestyle="--", transform=ccrs.PlateCarree())

    ax.set_global()

    gl=ax.gridlines(draw_labels=False, linestyle='--', color='gray', alpha=0.5)


    ax.set_title(f'{time_point[t]} Ma', fontsize=15)

ax.legend(loc='upper left', bbox_to_anchor=(-2.1, 2.1), fontsize=8)

plt.savefig("Diversity_time.svg")

plt.show()