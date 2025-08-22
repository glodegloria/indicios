import numpy as np

def inditek_model_proof (D,proof):
    model=D

    residuals = (model - proof)/0.05#no hay rudio, añadirlo cuando preparo proof

    residuals=residuals**2

    residuals=residuals[np.isnan(residuals)==False]

    N=len(residuals)

    rss=np.sum(residuals)

    print("Residual Sum of Squares (RSS):", rss)

    return rss