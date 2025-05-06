import numpy as np
from principal import principal
import scipy.io
import time
import mat73

#Charge the data (just for tests)

##############################################################################################
# JUST FOR TESTING PURPOSES TO LOAD THE DATA
###############################################################################################

data_point_ages=scipy.io.loadmat('Point_ages_xyzKocsisScotese_400.mat')
shelf_lonlatAge=data_point_ages['shelf_lonlatAge']
Point_timeslices=data_point_ages['Point_timeslices']
#
data_mask=mat73.loadmat('landShelfOceanMask_ContMargMaskKocsisScotese.mat')
landShelfOcean_Lat=data_mask['landShelfOcean_Lat']
landShelfOcean_Lon=data_mask['landShelfOcean_Lon']
landShelfOceanMask=data_mask['landShelfOceanMask']
landShelfOceanMask = np.flip(landShelfOceanMask, axis=2)

data_food_temp=scipy.io.loadmat('Point_foodtemp_v241023.mat')


food_shelf=data_food_temp['food_shelf']
temp_shelf=data_food_temp['temp_shelf']

data_LonDeg=scipy.io.loadmat('LonDeg.mat')
#print(data_LonDeg.keys())

LonDeg=data_LonDeg['LonDeg']

data_obis=scipy.io.loadmat("obis_data.mat")

d_obis=data_obis["d_obis"]
se_obis=data_obis["se_obis"]
idx_obis=data_obis["idx_obis"]

num_chains=2
nsamples=70
nparams=7
#
#
#
data=np.load("indicios.npz")
params_current=data["params_current"]
mu=data["mu"]
sigma=data["sigma"]
sigma_prop=data["sigma_prop"]
ran=data["ran"]

########################################################
#END OF LOADING DATA
#########################################################

#def inditek_metropolis(params_current, food_shelf, temp_shelf, Point_timeslices, shelf_lonlatAge, nsamples, nparams, d_obis,se_obis, idx_obis, landShelfOceanMask, landShelfOcean_Lat, landShelfOcean_Lon, LonDeg, mu, sigma, ran, sigma_prop):


#Charge the fix parameters

gaus=np.array([5])
#index=5    
kfood = 1 #[POC mol * m-2 yr-1]
#Kmin=10   #Carrying capacity of #genera at minimum food availability
lonWindow=2.5 # distance in degrees to search for particles from which diversity is "migrated" into the new coastal particles (newly submerged or artificially created by the paleotectonic model). e.g. for a 2x2 deg. window give 1 (lonWindow) & 1 (latWindow) values.
latWindow=2.5
#spec_min_mean = 0.1 #0.01; %minimum speciation rate that happens when food and temp are the lowest (#genera Myr^-1)
ext_pattern=3 # Zaffos curve = 1, % Alroy curve = 2, % Sepkoski curve = 3, % average all curves = 4

    #Storage for Diagnostics
output={

    "params_accepted_history": np.zeros([nsamples+1,nparams]),
    "params_proposed_history": np.zeros([nsamples,nparams]),
    "acceptance_history": np.zeros([nsamples,1]),
    "rss_accepted_history": np.zeros([nsamples,1]),
    "rss_proposed_history": np.zeros([nsamples,1]),
    "log_posterior_diff_history": np.zeros([nsamples,1])
    }


#Save the initial parameters in the output dictionary
output["params_accepted_history"][0,:]=params_current
output["params_proposed_history"][0,:]=params_current#Calculated in inditek_indicios as params_current=initial_theta(iChain,:)

        #Initial RSS Calculation (before the loop)



#Calculates the initial RSS (residual sum of squares) for the current parameters

#rss_current=principal(params_current[0], params_current[1], params_current[2], params_current[3], params_current[4])
rss_current = np.random.uniform(100, 5000)
#rss_current=principal(kfood, params_current[1], food_shelf, temp_shelf, ext_pattern, params_current[0], params_current[2], params_current[3], params_current[4], params_current[5], params_current[6], shelf_lonlatAge, Point_timeslices, latWindow,lonWindow,LonDeg, landShelfOcean_Lat,landShelfOcean_Lon, landShelfOceanMask, d_obis,se_obis,idx_obis)
temp=np.zeros(nparams)#force those with uniform distribution to a probability of 1 along the range (log(1)=0;)
if gaus.size>0:
    temp[gaus]=mu[gaus]#((params_current[gaus] - mu[gaus]) / sigma[gaus])**2


    # calculate the log(prior), log(likelihood) and log(posterior) of current parameters to compare to the proposed ones in the loop

log_prior_current=-0.5*sum(temp)
log_likelihood_current=-(1/2)*rss_current
log_posterior_current =log_prior_current+log_likelihood_current


            #Metropolis-Hastings samples with RSS   

    #Create the array to store the change of parameters in each iteration
#change_params=[[] for i in range(7)]
change_params=np.full([nparams,50], np.nan)

#Initialize the scalar index2, it is used to check if the parameter has changed in each iteration
#It is initialized to NaN to avoid problems in the first iteration
index2=np.nan

for iter in range(1,nsamples):

        #To change the parameter modified in each iteration but all with the same probability

        index1= np.random.randint(0, 7)

        #If the parameter is the same as in the previous iteration, the change_params array adds 1 to the index of the parameter that has changed
        if index1!=index2:
            #change_params[index1]+=1
            idx = np.isnan(change_params[index1]).argmax()
            change_params[index1][idx] = iter

        
        print("index1")
        print(index1)
        print("change_params")
        print(change_params)


        #Initializes the proposed parameters with the current ones
        params_proposed=params_current

        #If it is the 3st time that the parameter has changed, it modifies the sigma_prop of that parameter
        print(change_params[index1][np.isnan(change_params[index1])==0]) 
        if change_params[index1][np.isnan(change_params[index1])==0].size%3==0 and index2!=index1:
            
            #Calculates the acceptance rate (AR) of the parameter that has changed as the sum of the acceptance history divided by the number of iterations
            #With this code, it calculates the AR for all the parameters.
            #AR=np.nansum(output["acceptance_history"])/iter#No se si el AR ha de tener en cuenta solo a ese parametro o a todos en general
            valid_indices = change_params[index1][np.isnan(change_params[index1])==0].astype(int)
            AR=np.nansum(output["acceptance_history"][valid_indices])/iter#No se si el AR ha de tener en cuenta solo a ese parametro o a todos en general
            #AR=np.nansum(output["acceptance_history"][change_params[index1][np.isnan(change_params[index1])==0]])/iter#No se si el AR ha de tener en cuenta solo a ese parametro o a todos en general
            change_params[index1]=np.full([50], np.nan)
            print("AR")
            print(AR)
            #It changes the sigma_prop of the parameter that has changed
            sigma_prop[index1]=sigma_prop[index1]*AR/(0.4)


        print("sigma_prop")
        print(sigma_prop)

        #Perturb the selected parameter (propose a value randomly)=perturbation of parameter values according to a search-window represented by a gaussian of std sigma_prop
        A=np.random.randn(nparams,1)
        params_proposed[index1]=params_current[index1]+A[index1,0]*sigma_prop[index1]

        print("params_proposed")
        print(params_proposed)

        #Ensure Kmax_mean is an integer (maximum number of species in a location (particle))
        params_proposed[0]=np.round(params_proposed[0])


        #I run the acceptance procedure if all my parameters are in bounds (between the range defined in inditek_indicios)
        if np.all(params_proposed <= ran[:,1]) and np.all(params_proposed >=ran[:,0]):
            #Run the model and Calculate the RSS (residual sum of squares) for the proposed parameters
            rss_proposed = np.random.uniform(100, 5000) #aleatorio, para no gastar tanto tiempo
            #rss_proposed=principal(kfood, params_proposed[1], food_shelf, temp_shelf, ext_pattern, params_proposed[0], params_proposed[2], params_proposed[3], params_proposed[4], params_proposed[5], params_proposed[6], shelf_lonlatAge, Point_timeslices, latWindow,lonWindow,LonDeg, landShelfOcean_Lat,landShelfOcean_Lon, landShelfOceanMask, d_obis,se_obis,idx_obis) #con 7 parametros
            #rss_proposed=principal(kfood, Kmin, food_shelf, temp_shelf, ext_pattern, params_proposed[0], spec_min_mean, params_proposed[1], params_proposed[2], params_proposed[3], params_proposed[4], shelf_lonlatAge, Point_timeslices, latWindow,lonWindow,LonDeg, landShelfOcean_Lat,landShelfOcean_Lon, landShelfOceanMask, d_obis,se_obis,idx_obis) #con 5 parametros

            #as before, calculate the log(prior), log(likelihood) and log(posterior) of proposed parameters to compare to the current ones in the loop
            temp=np.zeros([nparams,1])
            if gaus.size>0:
                temp[gaus]=((params_current[gaus] - mu[gaus]) / sigma[gaus])**2
            log_prior_proposed=-0.5*sum(temp)
            #print("log_prior_proposed")
            #print(log_prior_proposed)
            log_likelihood_proposed=-(1/2)*rss_proposed
            #print("rss_proposed")
            #print(rss_proposed)
            #print("log_likelihood_proposed")
            #print(log_likelihood_proposed)
            log_posterior_proposed =log_prior_proposed+log_likelihood_proposed
            #print("log_posterior_proposed")
            #print(log_posterior_current)
            #####################################################################################
            #Save the results of the current iteration in the output dictionary

            output["params_proposed_history"][iter, 0:nparams]=params_proposed
            output["params_accepted_history"][iter, 0:nparams]=params_current
            output["rss_proposed_history"][iter]=rss_proposed
            #print("rss_proposed")
            output["rss_accepted_history"][iter]=rss_current
            #print("rss_current")
            #print(output["rss_accepted_history"][iter])
            output["log_posterior_diff_history"][iter]=log_posterior_proposed-log_posterior_current


            #Calculate Acceptance Probability according to the ratio between the likelihood of proposed vs current 
            # (log_posterior_proposed-log_posterior_current) and compare to a random (0-1) number u:

            u=np.random.rand()
            #print("np.log(u)")
            #print(np.log(u))
            #print("criterio")
            #print(log_posterior_proposed-log_posterior_current)
            if np.log(u)<log_posterior_proposed-log_posterior_current:

                

                acceptance_tagmark=1 #Mark as accepted
                # UPDATE parameter values for NEXT ITERATION

                params_current=params_proposed
                rss_current=rss_proposed
                log_posterior_current=log_posterior_proposed
            else:
                acceptance_tagmark=0
                # DO NOT UPDATE FOR NEXT ITERATION

            output["acceptance_history"][iter]=acceptance_tagmark

        else:
            #Just SKIP THE PROCEDURE for having any parameter proposal out of bounds
            output["params_proposed_history"][iter, 0:nparams]=params_proposed
            output["params_accepted_history"][iter, 0:nparams]=params_current
            output["rss_proposed_history"][iter]=np.nan
            output["rss_accepted_history"][iter]=rss_current
            output["log_posterior_diff_history"][iter]=np.nan
            output["acceptance_history"][iter]=0

        index2=index1

        #print("#########################################################################")
        #print("HECHO UNA ITERACION")
        print("############################################################################")
        input("Pulsa enter para continuar")

output["params_accepted_history"][iter+1, 0:nparams]=params_current

#save the output dictionary to a .npz file if you are checking

np.savez("datos_finales_metropolis.npz", params_proposed_history=output["params_proposed_history"], params_accepted_history=output["params_accepted_history"],
        rss_proposed_history=output["rss_proposed_history"], rss_accepted_history=output["rss_accepted_history"],
        acceptance_history=output["acceptance_history"], log_posterior_diff_history=output["log_posterior_diff_history"])


    #Return the output dictionary with the results of the Metropolis-Hastings algorithm
    #return output







