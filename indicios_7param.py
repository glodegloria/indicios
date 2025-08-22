import numpy as np
from latin_hypercube import latin_hypercube_sampling_ichains
from joblib import Parallel, delayed
import time
from metropolis_7param import inditek_metropolis
import mat73
import scipy.io


def run_chain(iChain):
    
    params_current=np.transpose(initial_theta[iChain,:])
    #output=inditek_metropolis(LonDeg, landShelfOcean_Lat,landShelfOcean_Lon, landShelfOceanMask, d_obis,se_obis, idx_obis, shelf_lonlatAge, Point_timeslices, food_shelf, temp_shelf, initial_theta[iChain,:], mu, sigma, sigma_prop, ran,  nsamples, nparams)
    #output=inditek_metropolis(       initial_theta[iChain,:],      )
    output=inditek_metropolis(params_current, food_shelf, temp_shelf, Point_timeslices, shelf_lonlatAge, nsamples, nparams, mean_obis,std_obis, ids_obis, landShelfOceanMask, landShelfOcean_Lat, landShelfOcean_Lon, LonDeg, mu, sigma, ran, sigma_prop, proof, indices_pac, indices_med, indices_car, n_D)

    return (
        output["params_proposed_history"], output["params_accepted_history"],
        output["rss_proposed_history"], output["rss_accepted_history"],
        output["acceptance_history"], 
        output["residuals"], 
        output["AR_parameter"], 
        output["new_parameter"], output["sigma_prop"], output["D"],
        output["D_pac"], output["D_med"], output["D_car"]
    )

start=time.time()

#MH-MCMC SETUP:
num_chains=1 
nsamples=5
nparams=5
n_D=2

#Mean of parameters distributions (mu):
#
#Kmax_mu = 161; % Maximum carrying capacity (maximum number of genera in a point with the greatest food available in the time series)  
#Kmin_mu = 19;  % Minimum carrying capacity (maximum number of genera in a point with the greatest food available in the time series)  
#spec_max_mu = 0.035; %greatest speciation rate according to FoodlimxTemplim  
#spec_min_mu = 0.002; %lowest speciation rate according to FoodlimxTemplim
#Q10_mu = 1.75; % parameter defining the thermal limitation for speciation  

mu=np.array([161,19,0.035,0.002,1.75])

#Standard deviation of parameters (sigma):  
#
#Kmax_std = 80  
#spec_max_std = 0.05  
#Q10_std = 0.2  
#ext_intercept_shelf_std = 0.005  
#ext_slope_std = 0.00025

sigma=np.array([1.61,0.19,0.00035,0.00002,0.0175])

#which parameters to consider with gaussian distribution instead of uniform distribution (no a priori, only bounds: range)

#gaus=np.array([5]) # for now we only believe that Q10 should fall around the value 2 



########################## Parameter distribution of the search-window to defign the proposal:  

#Starting point of the chains (theta at time 1) found with latinhypercube to efficiently cover the param.distributions:  
#
#Kmax_theta = [152.95,169.05];    
#Kmin_theta = [18.05,19.95]
#spec_max_theta = [0.03325,0.03675];  
#spec_min_theta = [0.0019,0.0021]
#Q10_theta = [1.6625,1.8375];  

ran=np.array([[152.95,169.05],[18.05,19.95],[0.03325,0.03675],[0.0019,0.0021],[1.6625,1.8375]]) #range for parameter volume sampling to start the chains from equidistant points in the hipervolume
initial_theta=latin_hypercube_sampling_ichains(ran[:,0], ran[:,1], num_chains)

c=1
sigma_prop=c*sigma

#Range of parameters ([min,max])= range of tolerance for the proposed parameter values in the M-H iterations (out of these bounds we reject the proposal)
#
#Kmax_range = [80,1000] 
# Kmin_range = [1,150] 
#spec_max_range = [0.001,np.inf]  
#spec_min_range = [0.0001,0.05]
#Q10_range = [1,4]  

ran=np.array([[80,1000],[1,150],[0.001,np.inf],[0.0001,0.05],[1,4]])




####################################################### 
# Load input variables 
#######################################################

data_food_temp=scipy.io.loadmat('Point_foodtemp_paleoconfKocsisScotese_option2_GenieV4.mat')

#print(data_food_temp.keys())
#food_ocean=data['food_ocean']
food_shelf=data_food_temp['food_shelf']
#temp_ocean=data['temp_ocean']
temp_shelf=data_food_temp['temp_shelf']

data_point_ages=scipy.io.loadmat('Point_ages_xyzKocsisScotese_400.mat')#
#print(data_point_ages.keys())
#
Point_timeslices=data_point_ages['Point_timeslices']
#Point_timeslices = Point_timeslices[0]
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

data_proof=np.load("datos_proof.npz")
proof=data_proof[ "proof"]

indices=np.load("indices_points.npz")
     
indices_pac=indices["indices_pac"]
indices_med=indices["indices_med"]
indices_car=indices["indices_car"]

#Pre-allocate variables to store results

params_proposed_history = np.zeros([nsamples,nparams, num_chains])
params_accepted_history = np.zeros([nsamples+1,nparams, num_chains])
rss_proposed_history = np.zeros([nsamples,num_chains])
rss_accepted_history = np.zeros([nsamples,num_chains])
acceptance_history = np.zeros([nsamples,num_chains])
log_posterior_diff_history = np.zeros([nsamples,num_chains])
residuals = np.zeros([2,2978,num_chains])
AR_parameter=np.zeros([5,num_chains])
new_parameter=np.zeros([5,num_chains])
sigma_new=np.zeros([nsamples,nparams, num_chains])
D = np.zeros([int(nsamples/n_D)+1, 2978, num_chains]) # to store the D values for every 2 iterations, as it is only printed every 2 iterations
D_pac = np.zeros([int(nsamples/n_D)+1, len(indices_pac), 82, num_chains])
D_med = np.zeros([int(nsamples/n_D)+1, len(indices_med), 82, num_chains])
D_car = np.zeros([int(nsamples/n_D)+1, len(indices_car), 82, num_chains])

#########################################################
# Start the parallel computation
#########################################################



results= Parallel(n_jobs=num_chains)(delayed(run_chain)(i) for i in range(num_chains))





for iChain, result in enumerate(results):

    params_proposed_history[:, :, iChain] = result[0]
    params_accepted_history[:, :, iChain] = result[1]
    rss_proposed_history[:, iChain] = result[2].flatten()
    rss_accepted_history[:, iChain] = result[3].flatten()
    acceptance_history[:, iChain] = result[4].flatten()
    residuals[:,:,iChain] = result[5]
    AR_parameter[:,iChain]=result[6]
    new_parameter[:,iChain]=result[7]
    sigma_new[:,:,iChain]=result[8]
    D[:, :, iChain] = result[9]
    D_pac[:,:,:,iChain]=result[10]
    D_med[:,:,:,iChain]=result[11]
    D_car[:,:,:,iChain]=result[12]

    

#Save the final results

np.savez("datos_finales_indicios_7param.npz", params_proposed_history=params_proposed_history, params_accepted_history=params_accepted_history, rss_proposed_history=rss_proposed_history, rss_accepted_history=rss_accepted_history, acceptance_history=acceptance_history,  AR_parameter=AR_parameter, new_parameter=new_parameter, sigma_new=sigma_new, D=D, D_pac=D_pac, D_car=D_car, D_med=D_med)

#Finally, to measure the time it costs for the simulation
end=time.time()
print('{:.4f} s'.format(end-start)) 
