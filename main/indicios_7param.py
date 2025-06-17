import numpy as np
from latin_hypercube import latin_hypercube_sampling_ichains
from joblib import Parallel, delayed
import time
from metropolis import inditek_metropolis
import mat73
import scipy.io


def run_chain(iChain):
    
    params_current=np.transpose(initial_theta[0,:])
    #output=inditek_metropolis(LonDeg, landShelfOcean_Lat,landShelfOcean_Lon, landShelfOceanMask, d_obis,se_obis, idx_obis, shelf_lonlatAge, Point_timeslices, food_shelf, temp_shelf, initial_theta[iChain,:], mu, sigma, sigma_prop, ran,  nsamples, nparams)
    #output=inditek_metropolis(       initial_theta[iChain,:],      )
    output=inditek_metropolis(params_current, food_shelf, temp_shelf, Point_timeslices, shelf_lonlatAge, nsamples, nparams, mean_obis,std_obis, ids_obis, landShelfOceanMask, landShelfOcean_Lat, landShelfOcean_Lon, LonDeg, mu, sigma, ran, sigma_prop)

    return (
        output["params_proposed_history"], output["params_accepted_history"],
        output["rss_proposed_history"], output["rss_accepted_history"],
        output["acceptance_history"], output["log_posterior_diff_history"]
    )

start=time.time()

#MH-MCMC SETUP:
num_chains=4
nsamples=1000
nparams=7

#Mean of parameters distributions (mu):
#
#Kmax_mu = ; % Maximum carrying capacity (maximum number of genera in a point with the greatest food available in the time series)  
#spec_max_mu = 0.15; %greatest speciation rate according to FoodlimxTemplim  
#Q10_mu = 2; % parameter defining the thermal limitation for speciation  
#ext_intercept_shelf_mu = 0.01; % background extinction in the tropics  
#ext_slope_mu = 0.0005; %slope of extinction rate according to absolute latitude from 20º

mu=np.array([200,10,0.15,0.005,2,0.01,0.0005])

#Standard deviation of parameters (sigma):  
#
#Kmax_std = 80  
#spec_max_std = 0.05  
#Q10_std = 0.2  
#ext_intercept_shelf_std = 0.005  
#ext_slope_std = 0.00025

sigma=np.array([80,4,0.05,0.002,0.2,0.005,0.00025])

#which parameters to consider with gaussian distribution instead of uniform distribution (no a priori, only bounds: range)

#gaus=np.array([5]) # for now we only believe that Q10 should fall around the value 2 



########################## Parameter distribution of the search-window to defign the proposal:  

#Starting point of the chains (theta at time 1) found with latinhypercube to efficiently cover the param.distributions:  
#
#Kmax_theta = [100,400];    
#spec_max_theta = [0.01,0.3];  
#Q10_theta = [1,3];  
#ext_intercept_shelf_theta = [0,0.2];  
#ext_slope_theta = [-0.001,0.001];

ran=np.array([[250,500],[5,20],[0.1,1.5],[0.001,0.02],[1,3],[0,0.2],[-0.003,0.003]]) #range for parameter volume sampling to start the chains from equidistant points in the hipervolume
initial_theta=latin_hypercube_sampling_ichains(ran[:,0], ran[:,1], num_chains)

c=0.35
sigma_prop=c*sigma

#Range of parameters ([min,max])= range of tolerance for the proposed parameter values in the M-H iterations (out of these bounds we reject the proposal)
#
#Kmax_range = [50,400];  
#spec_max_range = [0.001,Inf];  
#Q10_range = [1,3];  
#ext_intercept_shelf_range = [0,Inf];  
#ext_slope_range = [-Inf,+Inf];  

ran=np.array([[80,1000],[1,50],[0.1,np.inf],[0.0001,0.05],[1,4],[0,np.inf],[-np.inf,+np.inf]])



#Pre-allocate variables to store results

params_proposed_history = np.zeros([nsamples,nparams, num_chains])
params_accepted_history = np.zeros([nsamples+1,nparams, num_chains])
rss_proposed_history = np.zeros([nsamples,num_chains])
rss_accepted_history = np.zeros([nsamples,num_chains])
acceptance_history = np.zeros([nsamples,num_chains])
log_posterior_diff_history = np.zeros([nsamples,num_chains])

####################################################### Load input variables 

data_food_temp=scipy.io.loadmat('Point_foodtemp_v241023.mat')

#print(data_food_temp.keys())
#food_ocean=data['food_ocean']
food_shelf=data_food_temp['food_shelf']
#temp_ocean=data['temp_ocean']
temp_shelf=data_food_temp['temp_shelf']

data_point_ages=scipy.io.loadmat('Point_ages_xyzKocsisScotese_400')#
#print(data_point_ages.keys())
#
Point_timeslices=data_point_ages['Point_timeslices']
Point_timeslices=Point_timeslices[0]
shelf_lonlatAge=data_point_ages['shelf_lonlatAge']

data_LonDeg=scipy.io.loadmat('LonDeg.mat')
#print(data_LonDeg.keys())

LonDeg=data_LonDeg['LonDeg']

data_Mask=mat73.loadmat('landShelfOceanMask_ContMargMaskKocsisScotese.mat')
#print(data_Mask.keys())

landShelfOcean_Lat=data_Mask['landShelfOcean_Lat']
landShelfOcean_Lon=data_Mask['landShelfOcean_Lon']
landShelfOceanMask=data_Mask['landShelfOceanMask']
landShelfOceanMask = np.flip(landShelfOceanMask, axis=2)

data_obis=np.load("datos_obis.npz")

mean_obis=data_obis["mean_obis"]
std_obis=data_obis["obis_error"]
ids_obis=data_obis["index"]

data_ice=scipy.io.loadmat('Point_ice_v241023.mat')
ice_shelf=data_ice["ice_shelf"]

#########################################################Start the parallel computation



results= Parallel(n_jobs=num_chains)(delayed(run_chain)(i) for i in range(num_chains))





for iChain, result in enumerate(results):
    #print("imprimimos result[0], [2] y [3]")
    #print(result[0])
    #print(result[2])
    #print(result[3])

    params_proposed_history[:, :, iChain] = result[0]
    params_accepted_history[:, :, iChain] = result[1]
    rss_proposed_history[:, iChain] = result[2].flatten()
    rss_accepted_history[:, iChain] = result[3].flatten()
    acceptance_history[:, iChain] = result[4].flatten()
    log_posterior_diff_history[:, iChain] = result[5].flatten()

np.savez("datos_finales_indicios_7param.npz", params_proposed_history=params_proposed_history, params_accepted_history=params_accepted_history, rss_proposed_history=rss_proposed_history, rss_accepted_history=rss_accepted_history, acceptance_history=acceptance_history, log_posterior_diff_history=log_posterior_diff_history)


end=time.time()
print('{:.4f} s'.format(end-start)) 
