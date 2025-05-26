import numpy as np

def inditek_model_obisScaled(D, mean_obis, std_obis):

    residuals=mean_obis-D

    residuals=(residuals/std_obis)**2

    rss=np.sum(residuals)

    print(f"The residuals sum of squares is {rss}")

    return rss


