# 🌍 Inditek 2.0

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](#)  

This repository implements a **Metropolis-Hastings MCMC framework** to calibrate a **7-parameter ecological model** simulating biodiversity dynamics on continental shelves across geological time.  
The model integrates **food and temperature limitations**, **speciation/extinction processes**, and **spatially explicit diversity accumulation**.

---

## 📂 Project Structure

- **`run_chain.py`**  
  Entry point to run parallel MCMC chains using **joblib**.  
  Initializes parameters via Latin Hypercube Sampling and executes `inditek_metropolis`.

- **`metropolis_7param.py`**  
  Contains the `inditek_metropolis` function implementing the Metropolis-Hastings algorithm:  
  - Proposes new parameters  
  - Runs the ecological model (`principal`)  
  - Evaluates likelihood (Residual Sum of Squares)
  - - Accepts/rejects proposals  

- **`principal_proof.py`**  
  Defines `principal`, the core ecological model that:  
  - Computes speciation rates (`rhonet`)  
  - Simulates diversity (`alphadiv`)  
  - Aggregates results to spatial grids (`inditek_gridMean_alphadiv`)  
  - Calculates **Residual Sum of Squares (RSS)** vs observed data  

- **`rhonet.py`**  
  Calculates effective speciation rates (`rho_shelf`) and carrying capacity (`K_shelf`) considering food, temperature, and mass extinctions.

- **`alphadiv.py`**  
  Simulates alpha diversity accumulation at each shelf point over time with logistic growth, dispersal, and extinction.

- **`inditek_gridMean_alphadiv.py`**  
  Aggregates point-based diversity into global gridded diversity maps.
  
- **Required data files**  
  - `Point_foodtemp_paleoconfKocsisScotese_option2_GenieV4.mat`  
  - `Point_ages_xyzKocsisScotese_400.mat`  
  - `LonDeg.mat`  
  - `landShelfOceanMask_ContMargMaskKocsisScotese.mat`  
  - `datos_obis.npz`  
  - `datos_proof.npz`  
  - `indices_points.npz`  

---

## ⚙️ Installation

Clone the repository:



## 📊 Outputs

Each MCMC run generates `.npz` files containing:

### MCMC diagnostics
- `params_proposed_history` / `params_accepted_history`  
- `rss_proposed_history` / `rss_accepted_history`  
- `acceptance_history`  

### Diversity outputs
- `D` → Global diversity  
- `D_pac`, `D_med`, `D_car` → Regional diversities (Pacific, Mediterranean, Caribbean)  

### Adaptive proposal diagnostics
- `sigma_new`  
- `AR_parameter`  
- `new_parameter`  

---

## 📖 Notes

- **Adaptive proposals**: `sigma_prop` is dynamically updated to maintain target acceptance rates.  
- **Spatial regions**: Diversity is tracked globally and regionally.  
- **Latin Hypercube Sampling**: Used for initializing parameter space exploration.  
- **Mass extinctions**: Integrated into `rhonet` to simulate ecological shocks.  
- **Parallel execution**: Independent chains are run using `joblib.Parallel`.  
