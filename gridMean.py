import numpy as np
import scipy.io
import mat73

###########################################
#USE FOR TESTING PURPOSES
###########################################

#data=np.load("datos_finales.npz")
#D_shelf = data["D_shelf"]
##print(D_shelf.shape)
#
#data=scipy.io.loadmat('Point_ages_xyzKocsisScotese_400.mat')
#shelf_lonlatAge=data['shelf_lonlatAge']
#
#data=mat73.loadmat('landShelfOceanMask_ContMargMaskKocsisScotese.mat')
#landShelfOcean_Lat=data['landShelfOcean_Lat']
#landShelfOcean_Lon=data['landShelfOcean_Lon']
#landShelfOceanMask=data['landShelfOceanMask']
#landShelfOceanMask = np.flip(landShelfOceanMask, axis=2)

##############################################

def inditek_gridMean_alphadiv(D_shelf,shelf_lonlatAge,landShelfOcean_Lat,landShelfOcean_Lon, landShelfOceanMask):


    #Creates a 2D grid of latitude and longitude values using meshgrid. 

    [X,Y]=np.meshgrid(landShelfOcean_Lon,landShelfOcean_Lat)#


    #Edges of the latitude and longitude bins
    lat_edges=np.arange(-90,90.5,0.5)
    lon_edges=np.arange(-180,180.5,0.5)

    #Mask of the land shelf ocean (LSO) to ignore the land and ocean areas in the grid
    LSOmask=np.transpose(landShelfOceanMask[:,:,landShelfOceanMask.shape[2]-1])

    #Creates the lat and lon arrays for the last time slice of the shelf_lonlatAge array. 
    lat=shelf_lonlatAge[:,shelf_lonlatAge.shape[1]-1,1]#
    lon=shelf_lonlatAge[:,shelf_lonlatAge.shape[1]-1,0]#
    d=D_shelf[:,D_shelf.shape[1]-1]#

    
    


    #Selects the latitudes and longitudes that are not NaN (not active points without diversity at time 0Myr) and ignores the NaN values.
    lat=lat[np.isnan(d)==0]#
    lon=lon[np.isnan(d)==0]  #ingnore NaN (no active points without diversity at time 0Myr)#
    d=d[np.isnan(d)==0]#

    



    lat_idx=np.digitize(lat, lat_edges) - 1
    lon_idx=np.digitize(lon,lon_edges) - 1

    grid_idx= lat_idx * X.shape[1] + lon_idx

    #Creates a 2D array of zeros with the same shape as the grid, to store the diversity values.
    D=np.zeros(X.shape)
    count=np.zeros(X.shape)

    #It goes from all the elements of the grid and adds the diversity values to the corresponding grid cell.
    #The count array is used to count the number of values added to each grid cell. It is used to calculate the mean diversity value for each grid cell.
    for i in range(len(grid_idx)):
        if LSOmask.flat[grid_idx[i]]==1:
            D.flat[grid_idx[i]] += d[i]
            count.flat[grid_idx[i]]+=1


    #It diVides to calculate the mean diversity value for each grid cell.
    D=D/count




    #It just selects the points that are 1 in the LSOmask (the ones that are not land or ocean) and the points that the diversity is not 0 (the ones that have diversity).
    D[LSOmask!=1]=np.nan
    D[D==0]=np.nan

    return X, Y, D





