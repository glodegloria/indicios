import scipy.io
import mat73
import numpy as np
import pandas as pd
import time
from rhonet import rhonet_evo
from alphadiv_opt import alphadiv
from gridMean import inditek_gridMean_alphadiv
from model_obis import inditek_model_obisScaled

start_time = time.time()


#def principal(Kmax_mean, spec_max_mean, Q10_mean, ext_intercept_shelf_mean,ext_slope_mean):
def principal(kfood, Kmin, food_shelf, temp_shelf, ext_pattern, Kmax_mean, spec_min_mean, spec_max_mean, Q10_mean, ext_intercept_shelf_mean,ext_slope_mean, shelf_lonlatAge, Point_timeslices, latWindow,lonWindow,LonDeg, landShelfOcean_Lat,landShelfOcean_Lon, landShelfOceanMask, d_obis,se_obis,idx_obis):

#############################################################################################
# JUST FOR TESTING PURPOSES
###############################################################################################

#CHOOSE model parameters:
#kfood = 0.5 #[POC mol * m-2 yr-1] #1
#spec_min_mean = 0.001 #[MA-1]   #0.1
#spec_max_mean = 0.035 #[MA-1]   #la sacas
#Q10_mean = 1.75 #n.u.   #la sacas 
#Kmax_mean=16# Carrying capacity of #genera at maximum food availability #la sacas  
#Kmin=4 # Carrying capacity of #genera at minimum food availability #10
#ext_slope_mean=3 #la sacas
#ext_intercept_shelf_mean=5 #la sacas
#
#latWindow=2.5 #2.5
#lonWindow=2.5 #2.5
#
#ext_pattern=2 #3
#
#data_food_temp=scipy.io.loadmat('Point_foodtemp_v241023.mat')
#
##food_ocean=data['food_ocean']
#food_shelf=data_food_temp['food_shelf']
##temp_ocean=data['temp_ocean']
#temp_shelf=data_food_temp['temp_shelf']
#
#data_point_ages=scipy.io.loadmat('Point_ages_xyzKocsisScotese_400')
#
#Point_timeslices=data_point_ages['Point_timeslices']
#shelf_lonlatAge=data_point_ages['shelf_lonlatAge']
#
#data_LonDeg=scipy.io.loadmat('LonDeg.mat')
##print(data_LonDeg.keys())
#
#LonDeg=data_LonDeg['LonDeg']
#
#data_Mask=mat73.loadmat('landShelfOceanMask_ContMargMaskKocsisScotese.mat')
##print(data_Mask.keys())
#
#landShelfOcean_Lat=data_Mask['landShelfOcean_Lat']
#landShelfOcean_Lon=data_Mask['landShelfOcean_Lon']
#landShelfOceanMask=data_Mask['landShelfOceanMask']
#landShelfOceanMask = np.flip(landShelfOceanMask, axis=2)
#
#data_obis=scipy.io.loadmat("obis_data.mat")
#
#d_obis=data_obis["d_obis"]
#se_obis=data_obis["se_obis"]
#idx_obis=data_obis["idx_obis"]

##############################################
#END OF LOADING DATA
##############################################

#Calls the rhonet_evo function to calculate the rho_shelf and K_shelf matrices.
     [rho_shelf,K_shelf]=rhonet_evo(kfood,Kmin,food_shelf,temp_shelf,ext_pattern,Kmax_mean,spec_min_mean,spec_max_mean, Q10_mean,ext_intercept_shelf_mean,ext_slope_mean,shelf_lonlatAge,Point_timeslices)

#Calls the alphadiv function to calculate the D_shelf matrix.
     [rho_shelf_eff,D_shelf]=alphadiv(Point_timeslices,shelf_lonlatAge,rho_shelf,K_shelf,latWindow,lonWindow,LonDeg)

#Calls the inditek_gridMean_alphadiv function to calculate the grid that covers the earth surface and the mean of the diversity in each grid cell.
     [X,Y,D]=inditek_gridMean_alphadiv(D_shelf,shelf_lonlatAge,landShelfOcean_Lat,landShelfOcean_Lon, landShelfOceanMask)

     #Calculates the rss (Residual Sum of Squares) comparing the model diversity with the observed diversity.
     #This function will be changed soon to include the new data from OBIS.
     rss=inditek_model_obisScaled(D,X,Y,d_obis,se_obis,idx_obis) 

     return rss
     #elapsed_time = time.time() - start_time

     #print(f"La función tardó {elapsed_time:.4f} segundos.")

      




