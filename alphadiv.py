from scipy.io import loadmat
import numpy as np
from haversine_distance import haversine_distance




def alphadiv(Point_timeslices,shelf_lonlatAge,rho_shelf,K_shelf,latWindow,lonWindow,LonDeg, ext_index):

    pt=Point_timeslices# position of 82 time slices in the 542 Myr (starting from 0 Ma (million tears ago)+1=position 1) to retrieve only that info from the final data matrix
    #print(pt)
    pt=np.fliplr(pt).flatten()

    Point_timeslices=Point_timeslices[0]


    # 1. Calculate alpha diversity from points
    D0 = 1 # initialise diversity at time 541 MA with #1 genus area^(-1)
    D_shelf=np.full([shelf_lonlatAge.shape[0],542], np.nan)
    rho_shelf_eff=np.full([shelf_lonlatAge.shape[0],542], np.nan)


    count=-1 #time frame resolved (MA) (there are 82 timeframes out of 542MA defined by the Point_timeslices)
    step=0 # 82 time frames (steps in the loop)
    ts2=Point_timeslices[0]+1 #next timeframe after ts (to fill the gap between both at each loop)


    for ts in Point_timeslices:

        #print("ts: "+str(ts))

        count += (ts2-ts)#Update the count variable

        #Get ages and active point positions from shelf data (lonlatAge dimensions: pointsxtimeframesx[longitude,latitude,age])
        ageS = shelf_lonlatAge[:, step, 2]
        posS=np.where(np.logical_and(~np.isnan(ageS), ageS>0))[0]

        




        # Initialize diversity for the first timeframe (ts == Point_timeslices(1))
        if ts == Point_timeslices[0]:
            D_shelf[posS, count] = D0 #seed the coastal platform with 1 genus everywhere (to every active point) at time 541Ma
        else:

            deltaAgeS = ageS[posS] - shelf_lonlatAge[posS, step - 1, 2] #(age at time ts) - (age at time ts-1)

        ############## different kinds of points are treated a bit different to diversify:
        # #1# Handle newly inundated shelf points ##
        # Points that didn't exist or were not inundated in time t-1 and are now active


            pos2S = posS[np.logical_and(np.isnan(deltaAgeS), ageS[posS] <= ts2 - ts)]#Selects the points that didn't exist at time t-1
            pos2S=np.concatenate((pos2S,posS[np.logical_and(shelf_lonlatAge[posS,step-1,2]==0,ageS[posS]<=ts2-ts)]))#Select the points that are 0 years old

            


            if pos2S.size > 0:#If there are points of this type go through them 1 by 1 to find its nearest neighbour from which receive diversity mimicking inmigration

                for k in range(len(pos2S)): #Iterate over all points of this type

                    point_lonlat = shelf_lonlatAge[pos2S[k], step, [0,1]] # point location

                    # Find points within the spatial window to initialize diversity from

                    lon_diff = np.abs(np.abs(point_lonlat[ 1]) - LonDeg[:, 0])  
                    f = np.argmin(lon_diff) 

                    lon=lonWindow * LonDeg[f,1]

                    #logical conditions

                    lonMask = abs(shelf_lonlatAge[posS, step, 0] - point_lonlat[0]) <= lon
                    latMask = abs(shelf_lonlatAge[posS, step, 1] - point_lonlat[1]) <= latWindow



                    #Get the position for both conditions combined:

                    lim=np.where(lonMask & latMask)[0]
                    f=np.where(D_shelf[posS[lim],count2]>0)[0]
                    lim = lim[f]

                    if f.size==0:

                        d=D0 # No point with accumulated diversity found, initialize with D0
                        

                    else:#If there are points with diversity in the area

                        neighbor_lonlat = shelf_lonlatAge[posS[lim], step, 0:2]  

                        dist=haversine_distance(point_lonlat, neighbor_lonlat)#Calculate the distance to all the points in the area

                        lim=lim[dist!=0]
                        dist=dist[dist!=0]##Remove the points that are 0 distance (the point itself)

                        lim=lim[dist==min(dist)]#Select the point with the minimum distance to the point of interest

                        d=np.nanmean(D_shelf[posS[lim], count2])#The diversity of the point of interest is the average of the diversity of the points in the area (the ones that are not 0 distance)


                    #Boundaries between the carrying capacity (K_shelf) and D0 (1 genus area^(-1))

                    if d>K_shelf[pos2S[k],step]:
                        a3+=1
                        #print(d)
                        d=K_shelf[pos2S[k],step] #force local extinction if imported diversity is greater than K.
                        
                    elif d<D0:
                        a4+=1
                        d=D0 #force d to be at least D0, 1.


                    # diversification keeping d in between D0 and local K bounds
                    #if np.logical_and(len(np.unique(rho_shelf[:,count2+1]))==1, np.all(np.unique(rho_shelf[:,count2+1]))<0): # extinction period
                    if count2+1 in ext_index:

                        d=max(D0,d+rho_shelf[pos2S[k],count2+1]*d)#bounded by D0
                        D_shelf[pos2S[k],count2+1]=min(K_shelf[pos2S[k],step],d)#bounded by K_shelf (The carrying capacity)
                        a5+=1
                    else: # normal diversification period

                        d=min(K_shelf[pos2S[k],step],d+rho_shelf[pos2S[k],count2+1]*d*(max(0,1-(d/K_shelf[pos2S[k],step])))) 
                        D_shelf[pos2S[k],count2+1]=max(D0,d)
                        a6+=1

                    

            # #2# Special case of continental shelf points that did not exist in
            #time-1 and were artificially added in the Gplates model to fill gaps 
            # with age of nearest neighbour continental-shelf points (thus we start diversity with the diversity in the nearest continental shelf points) #
            pos2S=posS[np.logical_and(np.isnan(deltaAgeS),ageS[posS]>ts2-ts)] # point in time t-1 did not exist or was above land
            pos2S=np.concatenate((pos2S,posS[np.logical_and(shelf_lonlatAge[posS,step-1,2]==0,ageS[posS]>ts2-ts)]))

           
            #print("Del tipo 2 hay"+str(len(pos2S)))
            if pos2S.size > 0:

                #print("Esta en paso 2")
                for k in range(len(pos2S)):
                    point_lonlat = [shelf_lonlatAge[pos2S[k], step, 0],shelf_lonlatAge[pos2S[k], step,1]] # point location

                    #   Find points within the spatial window to initialize diversity from, the spatial window has a length of 5, 10, 15 and 30 degrees to find the nearest neighbour
                    lim=np.where(np.logical_and(np.logical_and(shelf_lonlatAge[posS,step,0]<=point_lonlat[0]+5, shelf_lonlatAge[posS,step,0]>=point_lonlat[0]-5), np.logical_and(shelf_lonlatAge[posS,step,1]<=point_lonlat[1]+5,shelf_lonlatAge[posS,step,1]>=point_lonlat[1]-5 )))[0]
                    f=np.where(D_shelf[posS[lim],count2]>0)[0]
                    if f.size==0:
                        lim=np.where(np.logical_and(shelf_lonlatAge[posS,step,0]<=point_lonlat[0]+10, shelf_lonlatAge[posS,step,0]>=point_lonlat[0]-10) & np.logical_and(shelf_lonlatAge[posS,step,1]<=point_lonlat[1]+10,shelf_lonlatAge[posS,step,1]>=point_lonlat[1]-10 ))[0]
                        f=np.where(D_shelf[posS[lim],count2]>0)[0]
                    if f.size==0:
                        lim=np.where(np.logical_and(shelf_lonlatAge[posS,step,0]<=point_lonlat[0]+15, shelf_lonlatAge[posS,step,0]>=point_lonlat[0]-15) & np.logical_and(shelf_lonlatAge[posS,step,1]<=point_lonlat[1]+15,shelf_lonlatAge[posS,step,1]>=point_lonlat[1]-15 ))[0]
                        f=np.where(D_shelf[posS[lim],count2]>0)[0]
                    if f.size==0:
                        lim=np.where(np.logical_and(shelf_lonlatAge[posS,step,0]<=point_lonlat[0]+30, shelf_lonlatAge[posS,step,0]>=point_lonlat[0]-30) & np.logical_and(shelf_lonlatAge[posS,step,1]<=point_lonlat[1]+30,shelf_lonlatAge[posS,step,1]>=point_lonlat[1]-30 ))[0]
                        f=np.where(D_shelf[posS[lim],count2]>0)[0]

                    #It calculates the distance to all the points in the area and selects the one with the minimum distance to the point of interest
                    lim=lim[f]
                    neighbor_lonlat = shelf_lonlatAge[posS[lim], step, 0:2]  # forma (N, 2)
                    dist=haversine_distance(point_lonlat, neighbor_lonlat)

                    lim=lim[dist!=0]
                    dist=dist[dist!=0]

                    lim = lim[np.isclose(dist, dist.min())]

                    #Then, it calculates the diversity of the point of interest as the average of the diversity of the points in the area (the ones that are not 0 distance), equal to the previous case
                    d=np.nanmean(D_shelf[posS[lim],count2])

                    if d>K_shelf[pos2S[k],step]:

                        d=K_shelf[pos2S[k],step] #force local extinction if imported diversity is greater than K.
                        
                    elif d<D0:

                        d=D0 #force d to be at least D0, 1.
                        

                        # diversification keeping d in between D0 and local K bounds
                        #equal to the previous case
                    if count2+1 in ext_index:# extinction period

                        d=max(D0,d+rho_shelf[pos2S[k],count2+1]*d)#bounded by D0
                        D_shelf[pos2S[k],count2+1]=min(K_shelf[pos2S[k],step],d)#bounded by K_shelf (The carrying capacity)
                
                    else: # normal diversification period

                        d=np.fmin(K_shelf[pos2S[k],step],d+rho_shelf[pos2S[k],count2+1]*d*(max(0,1-(d/K_shelf[pos2S[k],step])))) 
                        D_shelf[pos2S[k],count2+1]=max(D0,d) 

             #3# Normal points
            

            pos2S=posS[np.logical_and(np.logical_and(deltaAgeS>0,np.round(deltaAgeS)<=ts2-ts),shelf_lonlatAge[posS,step-1,2]!=0)] #exisiting points with normal behaviour continue to accumulate diversity

            


            #boundaries between the carrying capacity (K_shelf) and D0 (1 genus area^(-1))

            
            D_shelf[pos2S, count2] = np.maximum(D_shelf[pos2S, count2], 1)
            

            d=np.minimum(D_shelf[pos2S,count2], K_shelf[pos2S,step])

            d=np.maximum(D0,d)#bounded by D0

            if count2+1 in ext_index:#if suffers an extinction
                d=np.maximum(D0,d+rho_shelf[pos2S,count2+1]*d)#bounded by D0
                D_shelf[pos2S,count2+1]=np.minimum(K_shelf[pos2S,step],d)#bounded by K_shelf (The carrying capacity)
                

            else: # normal diversification period
                
                
                
                rho_shelf_eff[pos2S,count2+1] = rho_shelf[pos2S,count2+1]* np.maximum(0, (1 - (d / K_shelf[pos2S,step])))
                d=np.fmin(K_shelf[pos2S,step],d+d*rho_shelf[pos2S,count2+1]*(1-(d/K_shelf[pos2S,step])))
                
                D_shelf[pos2S,count2+1]=np.maximum(D0,d) 
                z=D_shelf[pos2S, count2+1]
                
                
                
            d=np.maximum(d,D0)

            #All active points (the 3 kinds defined by #N# above) accumulate diversity along the time gap

            d=D_shelf[posS,count2+1]
            
            Myr=len(range(count2+2,count+1))# %time gap to still diversify
            scaling=np.ones((d.size,1))
            scaling[np.isnan(deltaAgeS)==0]=(np.minimum(deltaAgeS[np.isnan(deltaAgeS)==0]-1,Myr)/Myr)[0] #for points that appeared in the middle of the period and 
            #therefore did not accumulate diversity for the whole period (only during their age gap)
            rho=rho_shelf[:,count2+2:count+1]

            
            
            if np.any(np.isin(np.arange(count2 + 2, count+1), ext_index)): #for a period with any Myr with a extinction we need to sum Myr step-wise diversity

                
                for gap in range(count2+2,count+1):
                    
                    d=D_shelf[posS,gap-1]

                    d=np.minimum(d,K_shelf[posS,step])

                    d=np.maximum(d,D0)

                    if gap in ext_index: #if suffers an extinction
                        
                        d=np.maximum(D0,d+d*rho_shelf[posS,gap]*scaling.flatten())#bounded by D0 
                        
                        D_shelf[posS,gap]=np.fmin(K_shelf[posS,step],d)
                        
                    else:
                        rho_shelf_eff[posS,gap] = rho_shelf[posS,gap]* np.maximum(0, (1 - (d / K_shelf[posS,step])))
                        d=np.fmin(K_shelf[posS,step],d+d*rho_shelf[posS,gap]*np.maximum(0, (1-(d/K_shelf[posS,step])))*scaling.flatten())
                        D_shelf[posS,gap]=np.maximum(D0,d)   
            

            
            
            else: #for a period without extinction we can apply the logistic growth for as an exponential approaching
                # saturation by the Myr gap to skip the sum loop
                
                

                d=np.fmin(K_shelf[posS,step],d)
                
                d=np.maximum(d,D0)
                d=K_shelf[posS,step]/ (1 + ((K_shelf[posS,step] / d) - 1) * np.exp(-rho_shelf[posS,count]*Myr*scaling.flatten()))
                
                d=np.minimum(K_shelf[posS,step],d)
                d=np.maximum(d,D0)

                
                D_shelf[posS,count]=d #included to avoid explosive values due to the explonential growth nature
                rho_shelf_eff[posS,count] = rho_shelf[posS,count] * np.maximum(0, (1 - (d / K_shelf[posS,step])))


            #Reset to D=D0 when the points is covered by ice
            #ice=ice_shelf[:,step]
            #f=np.where(ice>0)[0]
            #D_shelf[f,count]=D0
        
        #Set the ts (time slice), count and step variables for the next iteration
        ts2=ts
        count2=count
        step = step + 1

        
    # Flip to order from point time slice 1 (0MA) to 542 (541MA) and get the Point time slices
    #for which the model is resolved (pt)

    D_shelf=np.flip(D_shelf, axis=1)
    D_shelf=D_shelf[:,pt]

    #% Flip back once the point time slices for which the model is resolved are compiled

    D_shelf=np.flip(D_shelf, axis=1)

    rho_shelf_eff=np.flip(rho_shelf_eff, axis=1)
    rho_shelf_eff=rho_shelf_eff[:, pt]

    # Flip back once the point time slices for which the model is resolved are compiled
    rho_shelf_eff=np.flip(rho_shelf_eff, axis=1)


    #To save the data in a .npz file for tests
    np.savez("datos_comprobacion_alphadiv.npz", z=z, D_shelf=D_shelf, scaling=scaling)

    

    return rho_shelf_eff, D_shelf