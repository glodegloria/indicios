#Save the libraries
import numpy as np
from matplotlib import pyplot as plt
import scipy.io
import pandas as pd

#Save the Data
data=np.load("Datos_D_shelf.npz")
D_shelf=data["D_shelf"]

indices=np.load("indices_points_big.npz")

indices_pac=indices["indices_pac"]
indices_car=indices["indices_car"]
indices_med=indices["indices_med"]

data_point_ages=scipy.io.loadmat('Point_ages_xyzKocsisScotese_400')#
Point_timeslices=data_point_ages['Point_timeslices']
shelf_lonlatAge=data_point_ages['shelf_lonlatAge']

#Create an empty vector for each hotspot
indices_med_un=[[] for k in range(D_shelf.shape[1])]
indices_car_un=[[] for k in range(D_shelf.shape[1])]
indices_pac_un=[[] for k in range(D_shelf.shape[1])]

#We start with the points that are Nan
indices_med_b=indices_med
indices_car_b=indices_car
indices_pac_b=indices_pac

#We go through every step, in reverse
for i in range(D_shelf.shape[1]-1,0,-1):
    A_med=D_shelf[indices_med_b][:,i]
    A_car=D_shelf[indices_car_b][:,i]
    A_pac=D_shelf[indices_pac_b][:,i]
    #We calculate if the 30% of the points, or more, are NaN
    if sum(np.isnan(A_med))/len(A_med)>=0.3:
        #indices_not_nan=indices_un[~np.isnan(A)]
        
        #We select the points that are NaN
        ind_nan=indices_med_b[np.isnan(A_med)]

        #We select the latitude and longitude of the points that are NaN, in that specific time
        [lon,lat]=[shelf_lonlatAge[ind_nan,i,0],shelf_lonlatAge[ind_nan,i,1]]

        #We look for the neighbor points, the points that are in a distance of the nan point 2.5
        diff_lon = np.abs(shelf_lonlatAge[:, i, 0, None] - lon[None, :])
        diff_lat = np.abs(shelf_lonlatAge[:, i, 1, None] - lat[None, :])
        
        mask = (diff_lon < 2.5) & (diff_lat < 2.5)
        
        # indices que cumplen para cualquiera de los lon/lat
        indices_nan = np.where(mask.any(axis=1))[0]

        B=D_shelf[indices_nan][:,i]

        indices_nan=indices_nan[~np.isnan(B)]

        indices_med_b=np.union1d(indices_med_b,indices_nan)
    if sum(np.isnan(A_car))/len(A_car)>=0.3:
        #indices_not_nan=indices_un[~np.isnan(A)]
        #We select the points that are NaN
        ind_nan=indices_car_b[np.isnan(A_car)]

        #We select the latitude and longitude of the points that are NaN, in that specific time
        [lon,lat]=[shelf_lonlatAge[ind_nan,i,0],shelf_lonlatAge[ind_nan,i,1]]

        #We look for the neighbor points, the points that are in a distance of the nan point 2.5
        diff_lon = np.abs(shelf_lonlatAge[:, i, 0, None] - lon[None, :])
        diff_lat = np.abs(shelf_lonlatAge[:, i, 1, None] - lat[None, :])
        
        mask = (diff_lon < 2.5) & (diff_lat < 2.5)
        
        # indices que cumplen para cualquiera de los lon/lat
        indices_nan = np.where(mask.any(axis=1))[0]

        B=D_shelf[indices_nan][:,i]

        indices_nan=indices_nan[~np.isnan(B)]

        indices_car_b=np.union1d(indices_car_b,indices_nan)
    if sum(np.isnan(A_pac))/len(A_pac)>=0.3:
        #indices_not_nan=indices_un[~np.isnan(A)]
        #We select the points that are NaN
        ind_nan=indices_pac_b[np.isnan(A_pac)]

        #We select the latitude and longitude of the points that are NaN, in that specific time
        [lon,lat]=[shelf_lonlatAge[ind_nan,i,0],shelf_lonlatAge[ind_nan,i,1]]

        #We look for the neighbor points, the points that are in a distance of the nan point 2.5
        diff_lon = np.abs(shelf_lonlatAge[:, i, 0, None] - lon[None, :])
        diff_lat = np.abs(shelf_lonlatAge[:, i, 1, None] - lat[None, :])
        
        mask = (diff_lon < 2.5) & (diff_lat < 2.5)
        
        # indices que cumplen para cualquiera de los lon/lat
        indices_nan = np.where(mask.any(axis=1))[0]

        B=D_shelf[indices_nan][:,i]

        indices_nan=indices_nan[~np.isnan(B)]

        indices_pac_b=np.union1d(indices_pac_b,indices_nan)
        
    indices_med_un[i]=indices_med_b
    indices_car_un[i]=indices_car_b
    indices_pac_un[i]=indices_pac_b
    print(i)
    print("------------------")

max_len_med = max(len(v) for v in indices_med_un)
max_len_car = max(len(v) for v in indices_car_un)
max_len_pac = max(len(v) for v in indices_pac_un)

# crear un array rectangular relleno con np.nan
rectangular_array_med = np.full((len(indices_med_un), max_len_med), np.nan)
rectangular_array_car = np.full((len(indices_car_un), max_len_car), np.nan)
rectangular_array_pac = np.full((len(indices_pac_un), max_len_pac), np.nan)



# rellenar los datos
for i, v in enumerate(indices_med_un):
    rectangular_array_med[i, :len(v)] = v
for i, v in enumerate(indices_car_un):
    rectangular_array_car[i, :len(v)] = v
for i, v in enumerate(indices_pac_un):
    rectangular_array_pac[i, :len(v)] = v

indices_med_un=rectangular_array_med
indices_car_un=rectangular_array_car
indices_pac_un=rectangular_array_pac

np.savez('indices_por_anos.npz',indices_med=indices_med_un, indices_car=indices_car_un, indices_pac=indices_pac_un)
                
                