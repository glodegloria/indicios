import numpy as np
import scipy.io

#data=np.load("datos_finales_gridMean_nuevos.npz")
#D = data["D"]
#X = data["X"]
#Y = data["Y"]
#
#data=scipy.io.loadmat("obis_data.mat")
#
#d_obis=data["d_obis"]
#se_obis=data["se_obis"]
#idx_obis=data["idx_obis"]
#

def inditek_model_obisScaled(D,X,Y,d_obis,se_obis,idx_obis):

    print("model_obis")

    #Scale D from polygons of 800000 km2 at the equator to grids of 0.5*0.5º at the equator (3083 km2) as resolved in the original model grids
    obis=d_obis*((3083/800000)**0.3) #following as rule of thumb the modal diversity-area universal relationship Connor & McCoy 1979 The American Naturalist 113
    
    # model average diversity estimate per polygone
    model_polygone=np.full(obis.size, np.nan)

    model=D.flatten('F')
    for i in range(len(obis)):  
        idx = idx_obis[i][0]-1  # Extrae los índices  
        a = model[idx]  # Obtiene los valores de model en esos índices  
        if a[np.isnan(a)==0].size>0:
            model_polygone[i] = np.nanmean(a)  # Calcula la media ignorando NaN 

    residuals=model_polygone-obis[:,0]#250 [across 298 polygones covering the global coasts to a different extent (a fixed location of 48 polygones does not have data from the model to compare)]

    obis_error = se_obis*((3083/800000)**0.3)#%scale the standard error as the observation 


    A=(residuals/obis_error[:,0])**2#Applies the formula for the residuals of the model and the observations (obis) to calculate the RSS (Residual Sum of Squares)
    #print(A.shape)

    rss = np.sum(A[np.isnan(A)==0])#Sums the residuals that are not NaN 

    print(['Residual Sum of Squares (RSS): ', str(rss)])

    return rss