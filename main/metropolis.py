import numpy as np
from principal import principal
import scipy.io
import time
import mat73

def inditek_metropolis(LonDeg, landShelfOcean_Lat,landShelfOcean_Lon, landShelfOceanMask, d_obis,se_obis,idx_obis, shelf_lonlatAge, Point_timeslices, food_shelf, temp_shelf, params_current, mu, sigma, sigma_prop, ran,  nsamples, nparams):


###############################################################################################
## JUST FOR TESTING PURPOSES TO LOAD THE DATA
################################################################################################
#data_point_ages=scipy.io.loadmat('Point_ages_xyzKocsisScotese_400.mat')
#shelf_lonlatAge=data_point_ages['shelf_lonlatAge']
#Point_timeslices=data_point_ages['Point_timeslices']
##
#data_mask=mat73.loadmat('landShelfOceanMask_ContMargMaskKocsisScotese.mat')
#landShelfOcean_Lat=data_mask['landShelfOcean_Lat']
#landShelfOcean_Lon=data_mask['landShelfOcean_Lon']
#landShelfOceanMask=data_mask['landShelfOceanMask']
#landShelfOceanMask = np.flip(landShelfOceanMask, axis=2)
#
#data_food_temp=scipy.io.loadmat('Point_foodtemp_v241023.mat')
#
#
#food_shelf=data_food_temp['food_shelf']
#temp_shelf=data_food_temp['temp_shelf']
#
#data_LonDeg=scipy.io.loadmat('LonDeg.mat')
##print(data_LonDeg.keys())
#
#LonDeg=data_LonDeg['LonDeg']
#
#data_obis=scipy.io.loadmat("obis_data.mat")
#
#d_obis=data_obis["d_obis"]
#se_obis=data_obis["se_obis"]
#idx_obis=data_obis["idx_obis"]   
#


#nsamples=3
#nparams=7
#
    gaus=np.array([3])
#
#data=np.load("indicios.npz")
#params_current=data["params_current"]
#mu=data["mu"]
#sigma=data["sigma"]
#sigma_prop=data["sigma_prop"]
#ran=data["ran"]

########################################################
#END OF LOADING DATA
#########################################################

    #Charge the fix parameters
    kfood = 1 #[POC mol * m-2 yr-1]
    Kmin=10   #Carrying capacity of #genera at minimum food availability
    lonWindow=2.5 # distance in degrees to search for particles from which diversity is "migrated" into the new coastal particles (newly submerged or artificially created by the paleotectonic model). e.g. for a 2x2 deg. window give 1 (lonWindow) & 1 (latWindow) values.
    latWindow=2.5
    spec_min_mean = 0.1 #0.01; %minimum speciation rate that happens when food and temp are the lowest (#genera Myr^-1)
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
    output["params_proposed_history"][0,:]=params_current#Calculated in inditek_indicios as params_current=initial_theta(iChain,:)
    output["params_accepted_history"][0,:]=params_current
        #Initial RSS Calculation (before the loop)


    #Calculates the initial RSS (residual sum of squares) for the current parameters
    #rss_current = np.random.uniform(100, 5000)
    rss_current=principal(kfood, Kmin, food_shelf, temp_shelf, ext_pattern, params_current[0], spec_min_mean, params_current[1], params_current[2], params_current[3], params_current[4], shelf_lonlatAge, Point_timeslices, latWindow,lonWindow,LonDeg, landShelfOcean_Lat,landShelfOcean_Lon, landShelfOceanMask, d_obis,se_obis,idx_obis)
    temp=np.zeros([nparams,1]) #force those with uniform distribution to a probability of 1 along the range (log(1)=0;)
    if gaus.size>0: 
        temp[gaus]=((params_current[gaus] - mu[gaus]) / sigma[gaus])**2

    # calculate the log(prior), log(likelihood) and log(posterior) of current parameters to compare to the proposed ones in the loop

    log_prior_current=-0.5*sum(temp)
    log_likelihood_current=-(1/2)*rss_current
    log_posterior_current =log_prior_current+log_likelihood_current

        #Metropolis-Hastings samples with RSS   

    ########### Perturb the parameters (propose a value randomly according to a search-window represented by a gaussian o std sigma_prop

    A=np.random.randn(nparams,1)
    params_proposed=params_current+A[:,0]*sigma_prop

    # Ensure Kmax_mean is an integer (maximum number of species in a location (particle)

    params_proposed[0]=np.round(params_proposed[0])



    for iter in range(1,nsamples):
            #iter

            #Perturb the parameters (propose a value randomly)=perturbation of parameter values according to a search-window represented by a gaussian of std sigma_prop

        A=np.random.randn(nparams,1)
        params_proposed=params_current+A[:,0]*sigma_prop

        #Ensure Kmax_mean is an integer (maximum number of species in a location (particle))
        params_proposed[0]=np.round(params_proposed[0])

        print("params_proposed")
        print(params_proposed)

        #I run the acceptance procedure if all my parameters are in bounds (between the range defined in inditek_indicios)

        if np.all(params_proposed <= ran[:,1]) and np.all(params_proposed >=ran[:,0]):

            #Run the model and Calculate the RSS (residual sum of squares) for the proposed parameters
            #rss_proposed = np.random.uniform(100, 5000)
            rss_proposed=principal(kfood, Kmin, food_shelf, temp_shelf, ext_pattern, params_proposed[0], spec_min_mean, params_proposed[1], params_proposed[2], params_proposed[3], params_proposed[4], shelf_lonlatAge, Point_timeslices, latWindow,lonWindow,LonDeg, landShelfOcean_Lat,landShelfOcean_Lon, landShelfOceanMask, d_obis,se_obis,idx_obis)


            #as before, calculate the log(prior), log(likelihood) and log(posterior) of proposed parameters to compare to the current ones in the loop
            temp=np.zeros([nparams,1])
            if gaus.size>0:
                temp[gaus]=((params_current[gaus] - mu[gaus]) / sigma[gaus])**2
            log_prior_proposed=-0.5*sum(temp)
            log_likelihood_proposed=-(1/2)*rss_proposed
            log_posterior_proposed =log_prior_proposed+log_likelihood_proposed

            #Just for debugging purposes, print the values of the parameters and the RSS

            #print("log_prior_proposed")
            #print(log_prior_proposed)
            #print("rss_proposed")
            #print(rss_proposed)
            #print("log_likelihood_proposed")
            #print(log_likelihood_proposed)
            #print("log_posterior_proposed")
            #print(log_posterior_proposed)

            #####################################################################################
            #Save the results of the current iteration in the output dictionary

            output["params_proposed_history"][iter, 0:nparams]=params_proposed
            output["rss_proposed_history"][iter]=rss_proposed
            output["rss_accepted_history"][iter]=rss_current
            output["log_posterior_diff_history"][iter]=log_posterior_proposed-log_posterior_current#Me ha dado cero en la segunda iteracion
            output["params_accepted_history"][iter, 0:nparams]=params_current

            #Calculate Acceptance Probability according to the ratio between the likelihood of proposed vs current
            #  (log_posterior_proposed-log_posterior_current) and compare to a random (0-1) number u:
            u=np.random.rand()
            #print("np.log(u)")
            #print(np.log(u))
            #print("log_posterior_proposed-log_posterior_current:")
            #print(log_posterior_proposed-log_posterior_current)

            if np.log(u)<log_posterior_proposed-log_posterior_current:

                acceptance_tagmark=1 #Mark as accepted
            # UPDATE parameter values for NEXT ITERATION

                params_current=params_proposed.copy()
                rss_current=rss_proposed.copy()
                log_posterior_current=log_posterior_proposed.copy()
            else:
                acceptance_tagmark=0
                # DO NOT UPDATE FOR NEXT ITERATION

            print("acceptance_tagmark")
            print(acceptance_tagmark)
            output["acceptance_history"][iter]=acceptance_tagmark

        else:
            #Just SKIP THE PROCEDURE for having any parameter proposal out of bounds
            output["params_proposed_history"][iter, 0:nparams]=params_proposed
            output["params_accepted_history"][iter, 0:nparams]=params_current
            output["rss_proposed_history"][iter]=np.nan
            output["rss_accepted_history"][iter]=rss_current
            output["log_posterior_diff_history"][iter]=np.nan
            output["acceptance_history"][iter]=0

        print("#########################################################################")
        print("HECHO UNA ITERACION")
        print("############################################################################")
        #input("Press enter to continue")

        #print(output["rss_proposed_history"])
        #print(output["rss_accepted_history"])

    output["params_accepted_history"][iter+1, 0:nparams]=params_current
    #save the output dictionary to a .npz file if you are checking

#np.savez("datos_finales_metropolis.npz", params_proposed_history=output["params_proposed_history"], params_accepted_history=output["params_accepted_history"],
#        rss_proposed_history=output["rss_proposed_history"], rss_accepted_history=output["rss_accepted_history"],
#        acceptance_history=output["acceptance_history"], log_posterior_diff_history=output["log_posterior_diff_history"])



    #Return the output dictionary with all the results of the metropolis algorithm
    return output







