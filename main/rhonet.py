import numpy as np
import pandas as pd 

def rhonet_evo(kfood,Kmin,food_shelf,temp_shelf,ext_pattern,Kmax_mean,spec_min_mean,spec_max_mean, Q10_mean,ext_intercept_shelf_mean,ext_slope_mean,shelf_lonlatAge,Point_timeslices):
        
    
    data=pd.read_excel('rhoExtOriginal_b.xlsx')
    

    rhoExt=data.iloc[:,ext_pattern]
    time=data.iloc[:,0]
    time_ext=time[rhoExt<0]

    rho_shelf = np.tile(rhoExt, (shelf_lonlatAge.shape[0], 1))  
    #print(rhoExtshelf.shape)

    #Calculate Carrying Capacity (K) according to the range of greatest and lowest food available in the whole time series, 
    # after discarding the 0.01 outliers

    a=food_shelf[np.isnan(food_shelf)==0] #Takes the values of food_shelf that are not NaN
    Mfood=np.quantile(a,0.99)
    mfood=np.quantile(a,0.01)

    #Effective carrying capacity: max N of genera that can be supported in a point according to food at that point and time 

    K_shelf=Kmax_mean-(Kmax_mean-Kmin)*((Mfood-food_shelf)/(Mfood-mfood))

    #bounded between Kmax & Kmin (to reset those outlier values within the range)

    K_shelf[K_shelf>Kmax_mean]=Kmax_mean
    K_shelf[K_shelf<Kmin]=Kmin

    # Calculate Speciation Rate
    speciation_shelf=np.empty(food_shelf.shape)#empty matrix, same size as food_shelf to asign speciation
    extinction_shelf =np.empty(food_shelf.shape) # same for extinction rate

    for i in range(food_shelf.shape[1]):
        #Selects the values of temperature that are within the range of the 0.01 and 0.99 quantiles, to avoid outliers
        a=temp_shelf[:,i]
        Mtemp=np.quantile(a[np.isnan(a)==0],0.99)
        mtemp=np.quantile(a[np.isnan(a)==0],0.01)
    

        #Food limitation according to Michaelis-Menten analogous effect on population growth rate according to food availability 
        Qfood_shelf=food_shelf[:,i]/(kfood+food_shelf[:,i])

        #Thermal limitation according to Eppley curve defining the effect of temperature on metabolic rates and therefore on 
        # population growth rate according to temperature
        EppleyCurve=Q10_mean**((temp_shelf[:,i]-mtemp)/10)
        EppleyCurve_max=Q10_mean**((Mtemp-mtemp)/10)
        EppleyCurve=EppleyCurve/EppleyCurve_max
        Qtemp_shelf=EppleyCurve

        #bound food and thermal limitation to 0-1 range

        Qtemp_shelf[Qtemp_shelf>1]=1
        Qtemp_shelf[Qtemp_shelf<0]=0
        Qfood_shelf[Qfood_shelf > 1] = 1
        Qfood_shelf[Qfood_shelf < 0] = 0


        Qlim_shelf=Qfood_shelf*Qtemp_shelf#colimitation of food and temperature combined (product of both)

        # Calculate speciation rates according to food and temperature colimitation
        a=spec_max_mean - (spec_max_mean - spec_min_mean) * (1.0 - Qlim_shelf) #speciation dependent on food and temp limitation (temp limitation bounded to current thermal range, ie, considering aclimation)


        speciation_shelf[:,i]=a

        #Calculate Background Extinction Rate (latitude-dependent, both hemispheres) for each time slice
        #Shelf_lonlatAge is a 3 vector where the 1st represents longitude, the 2nd represents latitude and the 3rd represents Age
        shelf_lat = shelf_lonlatAge[:,i,1] # Extract latitude for the current time slice

        lat_shelf_abs = abs(shelf_lat) #convert to absolute latitude

        #ext_slope = 0.0001;            # Adjust slope as needed
        #ext_intercept_shelf = 0.01;  # Adjust intercept for shelf

        # Piecewise Linear Function
        extinction_shelf[:, i] = ext_intercept_shelf_mean + ext_slope_mean * np.fmax(0, lat_shelf_abs - 20)# Constant until 20, 
        #then linear (to test if extinction increases with latitude we apply this effect with a 'constant background 
        # extinction in the tropical areas (ext_intercept_shelf_mean) and increasing towards the poles at a constant pace 
        # (ext_slope_mean)
        b=extinction_shelf[:, i]
        # Ensure extinction rates are within bounds (for each time slice)
        # extinction_shelf(:, i) = max(extinction_shelf(:, i), ext_intercept_shelf_mean); #### NOTE THAT COMMENTING THIS I ALLOW THE EXT RATE TO DECREASE WITH INCREASING LATITUDE

    #Calculate Net Diversification Rate (rho) - Unconstrained
    rho_shelf1 = speciation_shelf - extinction_shelf

    #############################################################################
    #Hasta aqui hay que hacerlo si o si, obligatorio
    #############################################################################

    #Incorporate Mass Extinctions and Fill Gaps
    all_timeslices = np.arange(Point_timeslices[0], -1, -1)#Begin in 0 (because the last one is -1), if not it shows a problem

    #Save the years that have suffer the mass extinctions
    mass_ext_years=abs(time_ext)
    totalyears=Point_timeslices[0]+1

    #Create a boolean vector of size totalyears to mark the timeslices that are in Point_timeslices
    point_mask = np.zeros(totalyears, dtype=bool)
    point_mask[Point_timeslices] = True

    #Create a boolean vector of size totalyears to mark the timeslices that have mass extinctions
    ext_mask = np.zeros(totalyears, dtype=bool)
    ext_mask[mass_ext_years] = True


    #Select in the vector with the mask extinctions, the years that are in the Point_timeslices
    ext_mask = np.array(ext_mask).flatten()
    valid_mask = ~ext_mask[Point_timeslices]

    #Select in the Point_timeslices, the ones that don't suffer mass extinctions
    valid_timeslices = Point_timeslices[valid_mask]

    #Create a new array with the index that are in the Point_timeslices and don't suffer mass extinctions
    point_indices = np.nonzero(np.isin(all_timeslices, valid_timeslices))[0]

    #Copy in rho_shelf the values of rho_shelf1 that are in the Point_timeslices and don't suffer mass extinctions
    rho_shelf[:,point_indices]=rho_shelf1[:, valid_mask]

    #empty_cols = np.where(rho_shelf[0, :] == 0.01)[0]
    sorted_pts = np.sort(point_indices)

    #For the points that don't have mass extinctions and are not in the Point_timeslice, fill with the rho_shelf info of the next point_timeslice
    for i in range(len(sorted_pts) - 1):
        start = sorted_pts[i]
        end = sorted_pts[i + 1]
        gap_cols = np.where((rho_shelf[0, :] == 0.01) & (np.arange(542) > start) & (np.arange(542) < end))[0]
        if len(gap_cols) > 0:
            #print(sorted_pts[i], sorted_pts[i + 1], gap_cols)
            rho_shelf[:, gap_cols] = rho_shelf[:, end][:, np.newaxis]

    #create an index of the timeslices that have mass extinctions, to be used later
    ext_index=np.nonzero(np.isin(all_timeslices, abs(time_ext)))[0]



    #postPT=np.full([len(Point_timeslices[0]),1], np.nan)
#
    ## save rho for the 82 time frames at their corresponding position (posPT) in the -541MA:-1MA:0MA frames 
    ## that alraeady have the big extinction (rho<0) timeframes incorporated
#
#
    #for i in range(len(Point_timeslices[0])):
#
    #    a=np.where(all_timeslices==Point_timeslices[0][i])[0][0]#Select in all the sequence the years, just the ones that are in the Point_timeslices
    #    
    #    f=np.where(rho_shelf[:,a]<0)#Select the years in point_timeslices (the ones that appear in slices) that suffers a mass extinction
#
    #    #print(f)
    #    if f[0].size == 0:#If it doesn't suffers a mass extinctions
    #        #print(rho_ocean1[:,i])
    #        rho_shelf[:,a]=rho_shelf1[:,i]# It can be in both a because in the first one it goes from 1 to 541 just with the extinction ones, and in the second one it goes from 1 to 82
    #        postPT[i]=a
    #        print("a", a)
    #        print("all_time_slices",all_timeslices[a])
#
    ##print(postPT)
#
    #postPT = postPT[~np.isnan(postPT)].astype(int)
#
    ##print(postPT)

    #Fill the gaps that don't have extinction and are not in the pointslices by copying values from next point_timeslice


    #f=np.where(rho_shelf[0,:]==0.01)[0]
    #for i in range(0,len(postPT)-1):
#
    #    #print(postPT[i])
    #    v=f[np.where((f > postPT[i].item()) & (f < postPT[i + 1].item()))[0]]
    #    data_to_tile=rho_shelf[:, postPT[i+1]][:, np.newaxis]
    #    rho_shelf[:,v]=np.tile(data_to_tile, (1, len(v)))
    #    #rho_shelf[:,v]=np.tile(rho_shelf[:,postPT[i+1]],(len(v),1))
    #    #print(rho_shelf)
#
    #np.savez("datos_finales_rhonet_cambiados.npz", K_shelf=K_shelf, rho_shelf=rho_shelf, rho_original=rho_shelf1)

    return rho_shelf,K_shelf, ext_index