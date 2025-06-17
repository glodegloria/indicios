
import numpy as np

def inditek_gridMean_alphadiv(D_shelf,ids):

    #lat=shelf_lonlatAge[:,shelf_lonlatAge.shape[1]-1,1]#
    #lon=shelf_lonlatAge[:,shelf_lonlatAge.shape[1]-1,0]#
    d=D_shelf[:,D_shelf.shape[1]-1]#

    #lat=lat[np.isnan(d)==0]#
    #lon=lon[np.isnan(d)==0]  #ingnore NaN (no active points without diversity at time 0Myr)#
    d=d[np.isnan(d)==0]#

    areaid=np.unique(ids)
    areaid=areaid[np.isnan(areaid)==0]

    D=np.zeros(len(areaid))
    count=np.zeros(len(areaid))

    for i in range(len(d)):
        if np.isnan(ids[i])==0:#ids is generated in filtrado obis
            D[int(ids[i])] += d[i]
            count[int(ids[i])] += 1

    D=D/count

    return D
