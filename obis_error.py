import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

#Para las que ya llegan a 2000 extrapolar a partir de 300 muestras

def modelo_exp_neg(x, a, b):
    return a * (1 - np.exp(-b * x))

def michaelis_menten(x, a, b, c):
    return (a * x) / (b + x) + c

def power_law(x, a, b):
    return a * np.power(x, b)


modelos = {
    'exp_neg': (modelo_exp_neg, 2),       # 2 parámetros
    'michaelis_menten': (michaelis_menten, 3),  # 3 parámetros
    'power_law': (power_law, 2)           # 2 parámetros
}

resultados = []  



resultado=pd.read_csv('areas_obis_error.csv')


areas=resultado['OBJECTID'].unique()

buenas=[4,7,8,9,10,12,13,15,19,20,25,27,29,32,33,34,36,37,39,40,42,45,46,47,52,54,56,59,61,63,66]

medias=[1,2,5,6,14,16,17,21,22,26,28,31,35,38,43,44,50,53,54,60,62]

malas=[3,18,64,48, 66, 50, 52, 47, 26]

for k in [48,64]:

    area=int(k)

    print(k)

    if area in buenas:
        n_iter = 200
        print("ES buena")
    if area in medias:
        n_iter=500
        print("Es media")
    if area in malas:
        print("Es mala")
        n_iter = 1000


    df_area = resultado[resultado['OBJECTID'] == area]
    muestras_unicas = df_area['muestra'].unique()
    muestra_a_generos = defaultdict(set)

    
    for _, row in df_area.iterrows():
        muestra_a_generos[row['muestra']].add(row['genusid'])
    generos_2000 = np.zeros(n_iter)

    n_muestras_total = min(len(muestras_unicas), 2000)

    for i in range(n_iter):
        print(f'{i} de {n_iter}')
        genera_per_sample=np.zeros(len(range(1, n_muestras_total, 10)))
        number_samples=np.zeros(len(range(1, n_muestras_total, 10)))


        for idx, n_samples in enumerate(range(1, n_muestras_total, 10)):
            muestras_seleccionadas = np.random.choice(muestras_unicas, n_samples, replace=False)
            generos_unicos=set()
            for muestra in muestras_seleccionadas:
                    generos_unicos.update(muestra_a_generos[muestra])
            number_samples[idx] = n_samples
            genera_per_sample[idx] = len(generos_unicos)

        # Ajustar todos los modelos y calcular su RMSE
        best_rmse = np.inf
        best_model = None
        best_params = None

        for nombre, (modelo, n_param) in modelos.items():
            try:
                bounds = (0, [np.inf] * n_param)
                params, _ = curve_fit(modelo, number_samples, genera_per_sample, bounds=bounds)
                y_pred = modelo(number_samples, *params)
                rmse = np.sqrt(mean_squared_error(genera_per_sample, y_pred))

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = modelo
                    best_params = params

            except RuntimeError:
                continue  # Si no converge el ajuste, pasa al siguiente modelo

        # Extrapolación con el mejor modelo
        genera_extrap = best_model(2000, *best_params)
        generos_2000[i] = genera_extrap

        generos_2000[i] = genera_extrap

    mean_generos = np.mean(generos_2000)
    std_generos = np.std(generos_2000)

    resultados.append({
            'area': area,
            'mean_generos': mean_generos,
            'std_generos': std_generos
        })
    
resultados_df = pd.DataFrame(resultados)

resultados_df.to_csv(f'bootstrap_obis_sin_cortar.csv', index=False)









