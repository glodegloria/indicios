from scipy.io import loadmat
import numpy as np
from haversine_distance import haversine_distance



def alphadiv(Point_timeslices,shelf_lonlatAge,rho_shelf,K_shelf,latWindow,lonWindow,LonDeg):
    
    pt=Point_timeslices
    pt=np.fliplr(pt).flatten() # position of 82 time slices in the 542 Myr (starting from 0 Ma (million tears ago)+1=position 1) to retrieve only that info from the final data matrix
    


    # 1. Calculate alpha diversity from points
    D0 = 1 # initialise diversity at time 541 MA with #1 genus area^(-1)
    
    n_points=shelf_lonlatAge.shape[0]
    D_shelf=np.full([n_points,542], np.nan)
    rho_shelf_eff=np.full([n_points,542], np.nan)


    count=-1 #time frame resolved (MA) (there are 82 timeframes out of 542MA defined by the Point_timeslices)
    step=0 # 82 time frames (steps in the loop)
    ts2=Point_timeslices[0][0]+1 #next timeframe after ts (to fill the gap between both at each loop)

    for ts in Point_timeslices[0]:

        count += (ts2-ts)#Update the count variable

        #Get ages and active point positions from shelf data (lonlatAge dimensions: pointsxtimeframesx[longitude,latitude,age])
        ageS = shelf_lonlatAge[:, step, 2]
        posS=np.where(np.logical_and(~np.isnan(ageS), ageS>0))[0]




        # Initialize diversity for the first timeframe (ts == Point_timeslices(1))
        if ts == Point_timeslices[0][0]:
            D_shelf[posS, count] = D0 #seed the coastal platform with 1 genus everywhere (to every active point) at time 541Ma

        else:

            deltaAgeS = ageS[posS] - shelf_lonlatAge[posS, step - 1, 2] #age at time ts- age at time ts-1

            pos2S = posS[np.logical_and(np.isnan(deltaAgeS), ageS[posS] <= ts2 - ts)]
            pos2S=np.concatenate((pos2S,posS[np.logical_and(shelf_lonlatAge[posS,step-1,2]==0,ageS[posS]<=ts2-ts)]))


            if pos2S.size > 0:
                #print("Esta en paso 1")
                for k in range(len(pos2S)):

                    point_lonlat = shelf_lonlatAge[pos2S[k], step, [0,1]] # point location

                    # Find points within the spatial window to initialize diversity from

                    lon_diff = np.abs(np.abs(point_lonlat[ 1]) - LonDeg[:, 0])  # diferencia de longitud
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
                        d=D0
                    else:

                        neighbor_lonlat=np.array([shelf_lonlatAge[posS[lim], step, 0],shelf_lonlatAge[posS[lim], step, 1]])
                        dist=haversine_distance(point_lonlat, neighbor_lonlat)


                        lim=lim[dist!=0]
                        dist=dist[dist!=0]

                        lim=lim[dist==np.min(dist)]

                        d=np.nanmean(D_shelf[posS[lim], count2])



                    d=np.clip(d, D0, K_shelf[pos2S[k], step])


                    # diversification keeping d in between D0 and local K bounds
                    if np.logical_and(len(np.unique(rho_shelf[:,count2+1]))==1, np.all(np.unique(rho_shelf[:,count2+1]))<0): # extinction period

                        d=np.maximum(D0,d+rho_shelf[pos2S[k],count2+1]*d)#bounded by D0
                        D_shelf[pos2S[k],count2+1]=np.minimum(K_shelf[pos2S[k],step],d)#bounded by K_shelf (The carrying capacity)
                    else: # normal diversification period

                        d=np.minimum(K_shelf[pos2S[k],step],d+rho_shelf[pos2S[k],count2+1]*d*(max(0,1-(d/K_shelf[pos2S[k],step])))) 
                        D_shelf[pos2S[k],count2+1]=np.maximum(D0,d)

            # #2# Special case of continental shelf points that did not exist in
            #time-1 and were artificially added in the Gplates model to fill gaps 
            # with age of nearest neighbour continental-shelf points (thus we start diversity with the diversity in the nearest continental shelf points) #
            pos2S=posS[np.logical_and(np.isnan(deltaAgeS),ageS[posS]>ts2-ts)] # point in time t-1 did not exist or was above land
            pos2S=np.concatenate((pos2S,posS[np.logical_and(shelf_lonlatAge[posS,step-1,2]==0,ageS[posS]>ts2-ts)]))
            #print("Del tipo 2 hay"+str(len(pos2S)))
            if pos2S.size > 0:

                #print("Esta en paso 2")
                for k in range(len(pos2S)):
                    point_lonlat = shelf_lonlatAge[pos2S[k], step, [0,1]] # point location

                    lim=np.where(np.logical_and(np.abs(shelf_lonlatAge[posS,step,0]-point_lonlat[0])<=5,  np.abs(shelf_lonlatAge[posS,step,1]-point_lonlat[1])<=5))[0]
                    f=np.where(D_shelf[posS[lim],count2]>0)[0]
                    if f.size==0:
                        lim=np.where(np.logical_and(np.abs(shelf_lonlatAge[posS,step,0]-point_lonlat[0])<=10,  np.abs(shelf_lonlatAge[posS,step,1]-point_lonlat[1])<=10))[0]
                        f=np.where(D_shelf[posS[lim],count2]>0)[0]
                    if f.size==0:
                        lim=np.where(np.logical_and(np.abs(shelf_lonlatAge[posS,step,0]-point_lonlat[0])<=15,  np.abs(shelf_lonlatAge[posS,step,1]-point_lonlat[1])<=15))[0]
                        f=np.where(D_shelf[posS[lim],count2]>0)[0]
                    if f.size==0:
                        lim=np.where(np.logical_and(np.abs(shelf_lonlatAge[posS,step,0]-point_lonlat[0])<=30,  np.abs(shelf_lonlatAge[posS,step,1]-point_lonlat[1])<=30))[0]
                        f=np.where(D_shelf[posS[lim],count2]>0)[0]

                    lim=lim[f]
                    neighbor_lonlat=np.array([shelf_lonlatAge[posS[lim], step, 0],shelf_lonlatAge[posS[lim], step, 1]])
                    dist=haversine_distance(point_lonlat, neighbor_lonlat)

                    lim=lim[dist!=0]
                    dist=dist[dist!=0]

                    lim=lim[dist==min(dist)]
                    
                    d=np.nanmean(D_shelf[posS[lim],count2])

                    d=np.clip(d, D0, K_shelf[pos2S[k],step])

                        # diversification keeping d in between D0 and local K bounds
                    if np.logical_and(len(np.unique(rho_shelf[:,count2+1]))==1, np.all(np.unique(rho_shelf[:,count2+1]))<0): # extinction period
                        d=np.maximum(D0,d+rho_shelf[pos2S[k],count2+1]*d)#bounded by D0
                        D_shelf[pos2S[k],count2+1]=np.minimum(K_shelf[pos2S[k],step],d)#bounded by K_shelf (The carrying capacity)
                
                    else: # normal diversification period
                        d=np.minimum(K_shelf[pos2S[k],step],d+rho_shelf[pos2S[k],count2+1]*d*(max(0,1-(d/K_shelf[pos2S[k],step])))) 
                        D_shelf[pos2S[k],count2+1]=np.maximum(D0,d) 

             #3# Normal points

            pos2S=posS[np.logical_and(np.logical_and(deltaAgeS>0,deltaAgeS<=ts2-ts),shelf_lonlatAge[posS,step-1,2]!=0)] #exisiting points with 

            D_shelf[pos2S, count2] = np.maximum(D_shelf[pos2S, count2], 1)


            d=np.minimum(D_shelf[pos2S,count2], K_shelf[pos2S,step])

            d=np.maximum(D0,d)#bounded by D0

            if np.logical_and(len(np.unique(rho_shelf[:,count2+1]))==1, np.all(np.unique(rho_shelf[:,count2+1]))<0): # extinction period
                d=np.maximum(D0,d+rho_shelf[pos2S,count2+1]*d)#bounded by D0
                D_shelf[pos2S,count2+1]=np.minimum(K_shelf[pos2S,step],d)#bounded by K_shelf (The carrying capacity)

            else: # normal diversification period
                rho_shelf_eff[pos2S,count2+1] = rho_shelf[pos2S,count2+1]* np.maximum(0, (1 - (d / K_shelf[pos2S,step])))
                d=np.minimum(K_shelf[pos2S,step],d+d*rho_shelf[pos2S,count2+1]*(1-(d/K_shelf[pos2S,step])))
                D_shelf[pos2S,count2+1]=np.maximum(D0,d)

            #All active points (the 3 kinds defined by #N# above) accumulate diversity along the time gap

            d=np.maximum(D_shelf[posS,count2+1],D0)
    
            Myr=len(range(count2+2,count))# %time gap to still diversify
            scaling=np.ones((d.size,1))
            scaling[np.isnan(deltaAgeS)==0]=(np.minimum(deltaAgeS[np.isnan(deltaAgeS)==0]-1,Myr)/Myr)[0] #for points that appeared in the middle of the period and therefore did not accumulate diversity for the whole period (only during their age gap)

            rho=rho_shelf[count2+2:count,:]

            if sum(sum(rho<0))>0: #for a period with any Myr with a extinction we need to sum Myr step-wise diversity
                for gap in range(count2+2,count+1):
                    d=D_shelf[posS,gap-1]


                    d=np.minimum(d,K_shelf[posS,step])
                    d=np.maximum(d,D0)

                    if np.logical_and(len(np.unique(rho_shelf[:,gap]))==1, np.all(np.unique(rho_shelf[:,gap]))<0):
                        d=np.maximum(D0,d+d*rho_shelf[posS,gap]*scaling)#bounded by D0 
                        D_shelf[posS,gap]=min(K_shelf[posS,step],d)
                        
                    else:
                        rho_shelf_eff[posS,gap] = rho_shelf[posS,gap]* np.maximum(0, (1 - (d / K_shelf[posS,step])))
                        d=np.where(np.isnan(d), K_shelf[posS,step], np.minimum(K_shelf[posS,step],d+d*rho_shelf[posS,gap]*np.maximum(0, (1-(d/K_shelf[posS,step])))*scaling.flatten()))
                        D_shelf[posS,gap]=np.maximum(D0,d)     
                        
            else:
                d=np.minimum(K_shelf[posS,step],d)
                d=np.maximum(d,D0)
                #%D_shelf(posS,count)=d.*(1+rho_shelf(posS,count).*(1-(d./K_shelf(posS,step))).*Myr.*scaling);
                d=K_shelf[posS,step]/ (1 + ((K_shelf[posS,step] / d) - 1) * np.exp(-rho_shelf[posS,count]*Myr*scaling.flatten()))
                d=np.minimum(K_shelf[posS,step],d)
                d=np.maximum(d,D0)
                D_shelf[posS,count]=d #included to avoid explosive values due to the explonential growth nature
                #print("Puntos exponencial")
                #print(D_shelf[posS,count])
                #input("Presiona Enter para continuar...")
                rho_shelf_eff[posS,count] = rho_shelf[posS,count] * np.maximum(0, (1 - (d / K_shelf[posS,step])))

            #ice=ice_shelf[:,step]
            #f=np.where(ice>0)[0]
            #D_shelf[f,count]=D0

        print("step: "+str(step))
        #print("count: "+str(count))


        ts2=ts
        count2=count
        step += 1

        #print("ts2: "+str(ts2))
        #print("count2: "+str(count2))



    D_shelf=np.flip(D_shelf, axis=1)[:,pt]
    D_shelf=np.flip(D_shelf, axis=1)

    rho_shelf_eff=np.flip(rho_shelf_eff, axis=1)[:, pt]
    rho_shelf_eff=np.flip(rho_shelf_eff, axis=1)

    #np.savez("datos_finales.npz", D_shelf=D_shelf, rho_shelf_eff=rho_shelf_eff)

    return D_shelf,  rho_shelf_eff     
