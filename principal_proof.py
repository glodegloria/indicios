import scipy.io
import mat73
import numpy as np
import pandas as pd
import time
from rhonet import rhonet_evo
from alphadiv import alphadiv 
from gridMean import inditek_gridMean_alphadiv
from inditek_model_proof import inditek_model_proof


start_time = time.time()


def principal(kfood, Kmin, food_shelf, temp_shelf, ext_pattern, Kmax_mean, spec_min_mean, spec_max_mean, Q10_mean, ext_intercept_shelf_mean,ext_slope_mean, shelf_lonlatAge, Point_timeslices, latWindow,lonWindow,LonDeg, landShelfOcean_Lat,landShelfOcean_Lon, landShelfOceanMask, mean_obis,std_obis,ids_obis, proof, indices_pac, indices_med, indices_car):
#
#############################################################################################
# JUST FOR TESTING PURPOSES
################################################################################################
##
##CHOOSE model parameters:
     #kfood = 0.5 #[POC mol * m-2 yr-1] #1
     #spec_min_mean = 0.002 #[MA-1]   #0.1
     #spec_max_mean = 0.035 #[MA-1]   #la sacas
     #Q10_mean = 1.75 #n.u.   #la sacas 
     ### Carrying capacity of #genera at maximum food availability #la sacas  
     #Kmin=19# Carrying capacity of #genera at minimum food availability #10
     #ext_slope_mean=0 #la sacas
     #ext_intercept_shelf_mean=0 #la sacas
     ##
     #latWindow=2.5 #2.5
     #lonWindow=2.5 #2.5
     ##
     #ext_pattern=4 #3
     ##
     #Kmax_mean=161
     ##
     #data_food_temp=scipy.io.loadmat('Point_foodtemp_paleoconfKocsisScotese_option2_GenieV4.mat')
     ##
     ###print(data_food_temp.keys())
     ###food_ocean=data['food_ocean']
     #food_shelf=data_food_temp['food_shelf']
     ###temp_ocean=data['temp_ocean']
     #temp_shelf=data_food_temp['temp_shelf']
     ##
     #data_point_ages=scipy.io.loadmat('Point_ages_xyzKocsisScotese_400')#
     ###print(data_point_ages.keys())
     ###
     #Point_timeslices=data_point_ages['Point_timeslices']
     ###Point_timeslices=Point_timeslices[0]
     ###Point_timeslices=Point_timeslices[0]
     #shelf_lonlatAge=data_point_ages['shelf_lonlatAge']
     ##
     #data_LonDeg=scipy.io.loadmat('LonDeg.mat')
     ###print(data_LonDeg.keys())
     ##
     #LonDeg=data_LonDeg['LonDeg']
     ##
     #data_Mask=mat73.loadmat('landShelfOceanMask_ContMargMaskKocsisScotese.mat')
     ###print(data_Mask.keys())
     ##
     #landShelfOcean_Lat=data_Mask['landShelfOcean_Lat']
     #landShelfOcean_Lon=data_Mask['landShelfOcean_Lon']
     #landShelfOceanMask=data_Mask['landShelfOceanMask']
     #landShelfOceanMask = np.flip(landShelfOceanMask, axis=2)
     ##
     #data_obis=np.load("datos_obis.npz")
     ##
     #mean_obis=data_obis["mean_obis"]
     #std_obis=data_obis["obis_error"]
     #ids_obis=data_obis["index"]
     ##
     #data_proof=np.load("datos_proof.npz")
     #proof=data_proof[ "proof"]
#
     #indices=np.load("indices_points.npz")
     #indices_pac=indices["indices_pac"]
     #indices_med=indices["indices_med"]
     #indices_car=indices["indices_car"]
     #############################################
     ##END OF LOADING DATA
     #############################################

     #Calls the rhonet_evo function to calculate the rho_shelf and K_shelf matrices.

     [rho_shelf,K_shelf, ext_index]=rhonet_evo(kfood,Kmin,food_shelf,temp_shelf,ext_pattern,Kmax_mean,spec_min_mean,spec_max_mean, Q10_mean,ext_intercept_shelf_mean,ext_slope_mean,shelf_lonlatAge,Point_timeslices[0])
                     
     #Calls the alphadiv function to calculate the D_shelf matrix.

     [rho_shelf_eff,D_shelf]=alphadiv(Point_timeslices,shelf_lonlatAge,rho_shelf,K_shelf,latWindow,lonWindow,LonDeg, ext_index)

     #Calls the inditek_gridMean_alphadiv function to calculate the grid that covers the earth surface and the mean of the diversity in each grid cell.
     [X, Y, D]=inditek_gridMean_alphadiv(D_shelf,shelf_lonlatAge,landShelfOcean_Lat,landShelfOcean_Lon, landShelfOceanMask)

     D_nan=D[~np.isnan(D)]

     #Calculates the rss (Residual Sum of Squares) comparing the model diversity with the observed diversity.
     #This function will be changed soon to include the new data from OBIS.

     rss=inditek_model_proof(D,proof)

                    

     

     return rss, D, D_shelf[indices_pac,:], D_shelf[indices_med,:], D_shelf[indices_car,:]
     #elapsed_time = time.time() - start_time


     #print(f"La función tardó {elapsed_time:.4f} segundos.")






