import numpy as np
from principal_proof import principal
import scipy.io
import time
import mat73

#Charge the data (just for tests)

##############################################################################################
# JUST FOR TESTING PURPOSES TO LOAD THE DATA
###############################################################################################

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
#num_chains=2
#nsamples=3
#nparams=7
##
##
##
#data=np.load("indicios.npz")
#params_current=data["params_current"]
#mu=data["mu"]
#sigma=data["sigma"]
#sigma_prop=data["sigma_prop"]
#ran=data["ran"]
#
########################################################
#END OF LOADING DATA
#########################################################

def inditek_metropolis(params_current, food_shelf, temp_shelf, Point_timeslices, shelf_lonlatAge, nsamples, nparams, mean_obis,std_obis, ids_obis, landShelfOceanMask, landShelfOcean_Lat, landShelfOcean_Lon, LonDeg, mu, sigma, ran, sigma_prop, proof, indices_pac, indices_med, indices_car, n_D):


    #Charge the fix parameters
    ext_intercept=0
    ext_slope=0
    gaus=np.array([4])  
    kfood = 0.5 #[POC mol * m-2 yr-1]
    lonWindow=2.5 # distance in degrees to search for particles from which diversity is "migrated" into the new coastal particles (newly submerged or artificially created by the paleotectonic model). e.g. for a 2x2 deg. window give 1 (lonWindow) & 1 (latWindow) values.
    latWindow=2.5
    ext_pattern=4 # Zaffos curve = 1, % Alroy curve = 2, % Sepkoski curve = 3, % average all curves = 4

        #Storage for Diagnostics
    output={

        "params_accepted_history": np.zeros([nsamples+1,nparams]),
        "params_proposed_history": np.zeros([nsamples,nparams]),
        "acceptance_history": np.zeros([nsamples,1]),
        "rss_accepted_history": np.zeros([nsamples,1]),
        "rss_proposed_history": np.zeros([nsamples,1]),
        #"log_posterior_diff_history": np.zeros([nsamples,1]),
        "residuals": np.zeros([2,2978]),
        "AR_parameter": np.zeros(5),
        "new_parameter": np.zeros(5),
        "sigma_prop": np.zeros([nsamples,nparams]),
        "D": np.zeros([int(nsamples/n_D)+1,2978]),
        "D_pac": np.zeros([int(nsamples/n_D)+1, len(indices_pac), 82]),
        "D_med": np.zeros([int(nsamples/n_D)+1, len(indices_med), 82]),
        "D_car": np.zeros([int(nsamples/n_D)+1, len(indices_car), 82])
        }


    #Save the initial parameters in the output dictionary
    output["params_accepted_history"][0,:]=params_current
    output["params_proposed_history"][0,:]=params_current#Calculated in inditek_indicios as params_current=initial_theta(iChain,:)

            #Initial RSS Calculation (before the loop)



    #Calculates the initial RSS (residual sum of squares) for the current parameters, it also saves the current global diversity and the 
    #diversity along time in the mediterranean, pacific and caribbean

    [rss_current,D,  D_pac, D_med, D_car]=principal(kfood, params_current[1], food_shelf, temp_shelf, ext_pattern, params_current[0], params_current[3], params_current[2], params_current[4], ext_intercept, ext_slope, shelf_lonlatAge, Point_timeslices, latWindow,lonWindow,LonDeg, landShelfOcean_Lat,landShelfOcean_Lon, landShelfOceanMask, mean_obis,std_obis,ids_obis, proof, indices_pac, indices_med, indices_car)

    temp=np.zeros(nparams)#force those with uniform distribution to a probability of 1 along the range (log(1)=0;)
    if gaus.size>0:
        temp[gaus]=mu[gaus]#((params_current[gaus] - mu[gaus]) / sigma[gaus])**2


        # calculate the log(prior), log(likelihood) and log(posterior) of current parameters to compare to the proposed ones in the loop

    log_prior_current=-0.5*sum(temp)
    log_likelihood_current=-(1/2)*rss_current
    log_posterior_current =log_prior_current+log_likelihood_current


                #Metropolis-Hastings samples with RSS   

    #Create the array to store the change of parameters in each iteration

    n_AR=100
    change_params=np.full([nparams,n_AR], np.nan)

    #Initialize the scalar index2, it is used to check if the parameter has changed in each iteration
    #It is initialized to NaN to avoid problems in the first iteration
    index2=np.nan

    AR_parameter=np.zeros(5)
    new_parameter=np.zeros(5)
    

    for iter in range(1,nsamples):


        #To change the parameter modified in each iteration but all with the same probability

        index1= np.random.randint(0, 5)
        new_parameter[index1]+=1

        #If the parameter is the same as in the previous iteration, the change_params array adds 1 to the index of the parameter that has changed

        if index1!=index2:
            #change_params[index1]+=1
            idx = np.isnan(change_params[index1]).argmax()
            change_params[index1][idx] = iter


        #Initializes the proposed parameters with the current ones
        params_proposed=params_current.copy()

        #If it is the 3st time that the parameter has changed, it modifies the sigma_prop of that parameter         #Guardar numero de veces que cambias un parametro y las veces que lo aceptas
        if change_params[index1][np.isnan(change_params[index1])==0].size%n_AR==0 and index2!=index1:
            
            #Calculates the acceptance rate (AR) of the parameter that has changed as the sum of the acceptance history divided by the number of iterations
            valid_indices = change_params[index1][np.isnan(change_params[index1])==0].astype(int)
            AR=np.nanmean(output["acceptance_history"][valid_indices])
            change_params[index1]=np.full([n_AR], np.nan)
            #It changes the sigma_prop of the parameter that has changed
            sigma_prop[index1]=max(sigma_prop[index1]*AR/(0.4),1e-12)


        #Perturb the selected parameter (propose a value randomly)=perturbation of parameter values according to a search-window represented by a gaussian of std sigma_prop
        A=np.random.randn(nparams,1)
        params_proposed[index1]=params_current[index1]+A[index1,0]*sigma_prop[index1]


        #I run the acceptance procedure if all my parameters are in bounds (between the range defined in inditek_indicios)
        
        if np.all(params_proposed <= ran[:,1]) and np.all(params_proposed >=ran[:,0]):

            #Run the model and Calculate the RSS (residual sum of squares) for the proposed parameters

            [rss_proposed,D, D_pac, D_med, D_car]=principal(kfood, params_proposed[1], food_shelf, temp_shelf, ext_pattern, params_proposed[0], params_proposed[3], params_proposed[2], params_proposed[4], ext_intercept, ext_slope, shelf_lonlatAge, Point_timeslices, latWindow,lonWindow,LonDeg, landShelfOcean_Lat,landShelfOcean_Lon, landShelfOceanMask, mean_obis,std_obis,ids_obis, proof, indices_pac, indices_med, indices_car) #con 7 parametros

            #As it is done before, calculate the log(prior), log(likelihood) and log(posterior) of proposed parameters to compare to the current ones in the loop
            temp=np.zeros([nparams,1])
            if gaus.size>0:
                temp[gaus]=((params_current[gaus] - mu[gaus]) / sigma[gaus])**2
            log_prior_proposed=-0.5*sum(temp)
            
            log_likelihood_proposed=-(1/2)*rss_proposed
            
            log_posterior_proposed =log_prior_proposed+log_likelihood_proposed
           
            #Save the results of the current iteration in the output dictionary

            output["params_proposed_history"][iter, 0:nparams]=params_proposed
            output["params_accepted_history"][iter, 0:nparams]=params_current
            output["rss_proposed_history"][iter]=rss_proposed
            output["rss_accepted_history"][iter]=rss_current
            output["sigma_prop"][iter]=sigma_prop
            if iter % n_D == 0:
                output["D"][int(iter/n_D),:]=D
                output["D_pac"][int(iter/n_D),:,:]=D_pac
                output["D_med"][int(iter/n_D),:,:]=D_med
                output["D_car"][int(iter/n_D),:,:]=D_car


            #Calculate Acceptance Probability according to the ratio between the likelihood of proposed vs current 
            # (log_posterior_proposed-log_posterior_current) and compare to a random (0-1) number u:

            u=np.random.rand()
            if np.log(u)<log_posterior_proposed-log_posterior_current:

                

                acceptance_tagmark=1 #Mark as accepted
                AR_parameter[index1]+=1
                # UPDATE parameter values for NEXT ITERATION

                params_current=params_proposed.copy()
                rss_current=rss_proposed
                log_posterior_current=log_posterior_proposed.copy()
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
            output["sigma_prop"][iter]=sigma_prop
            output["acceptance_history"][iter]=0

            if iter % n_D == 0:
                output["D"][int(iter/n_D),:]=np.nan
                output["D_pac"][int(iter/n_D),:,:]=D_pac
                output["D_med"][int(iter/n_D),:,:]=D_med
                output["D_car"][int(iter/n_D),:,:]=D_car
        index2=index1

        #print("#########################################################################")
        #print("ONE ITERATION DONE")
        #print("############################################################################")

        #print(f"new_parameter: {new_parameter}")
        #print(f"acceptance rate for the parameter: {AR_parameter}")

    output["params_accepted_history"][iter+1, 0:nparams]=params_current

    output["AR_parameter"]=AR_parameter
    output["new_parameter"]=new_parameter


################################################################
#save the output dictionary to a .npz file if you are checkingç
################################################################

    #np.savez("datos_finales_metropolis.npz", params_proposed_history=output["params_proposed_history"], params_accepted_history=output["params_accepted_history"],
    #    rss_proposed_history=output["rss_proposed_history"], rss_accepted_history=output["rss_accepted_history"],
    #    acceptance_history=output["acceptance_history"], log_posterior_diff_history=output["log_posterior_diff_history"])


    #Return the output dictionary with the results of the Metropolis-Hastings algorithm
    return output







