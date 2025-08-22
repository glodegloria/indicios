import numpy as np
from scipy.stats.qmc import LatinHypercube

def latin_hypercube_sampling_ichains(min_vals, max_vals, num_chains):
    sampler = LatinHypercube(d=len(min_vals))
    initial_params = sampler.random(n=num_chains)
    for j in range(len(min_vals)):
        initial_params[:,j]=min_vals[j]+(max_vals[j]-min_vals[j])*initial_params[:,j]

    return initial_params
