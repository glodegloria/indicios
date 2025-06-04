import pandas
import numpy as np
from collections import defaultdict

#Load the dataset
resultado=pandas.read_csv('areas_bootstrap.csv')

resultado = resultado.sort_values(by='OBJECTID')

#n_iter=200


areas=resultado['OBJECTID'].unique()

buenas=[4,7,8,9,10,12,13,15,19,20,25,27,29,32,33,34,36,37,39,40,42,45,46,47,48,52,54,56,59,61,63,66]

medias=[1,2,5,6,14,16,17,21,22,26,28,31,35,38,43,44,50,53,54,60,62]

malas=[3,18,64]





for area in areas:

    area=int(area)
    if area in buenas:
        n_iter=200
    elif area in medias:
        n_iter=500
    elif area in malas:
        n_iter=1000

    resultados = []  
    

    df_area = resultado[resultado['OBJECTID'] == area]

    print(len(df_area))

    muestras_unicas = df_area['muestra'].unique()

    muestra_a_generos = defaultdict(set)
    for _, row in df_area.iterrows():
        muestra_a_generos[row['muestra']].add(row['genusid'])

    n_muestras_total=min(2000,len(muestras_unicas))

    for n_muestras in range(1, n_muestras_total, 10):
        
        print(f"n_muestras{n_muestras} de {len(muestras_unicas)}")

        generos_por_iteracion = np.zeros(n_iter)

        for i in range(n_iter):

            # Randomly select n_muestras unique samples
            muestras_seleccionadas = np.random.choice(muestras_unicas, n_muestras, replace=False)

            generos_unicos=set()

            for muestra in muestras_seleccionadas:
                generos_unicos.update(muestra_a_generos[muestra])

            generos_por_iteracion[i]=len(generos_unicos)

            #Calculate the mean and standard deviation of the number of unique genera
        mean_generos = np.mean(generos_por_iteracion)
        std_generos = np.std(generos_por_iteracion)

        resultados.append({
            'area': area,
            'n_muestras': n_muestras,
            'mean_generos': mean_generos,
            'std_generos': std_generos
        })

    #Convert the results to a DataFrame
    resultados_df = pandas.DataFrame(resultados)

    df_ordenado = resultados_df.sort_values(by='area')

    df_ordenado.to_csv(f'bootstrap_muestras_{area}.csv', index=False)

    resultados=[]