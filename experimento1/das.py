import scipy.io
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter


# Cargar datos
data_point_ages = scipy.io.loadmat('Point_ages_xyzKocsisScotese_400.mat')
shelf_lonlatAge = data_point_ages['shelf_lonlatAge']  # (49688, 82, 3)
Point_timeslices = data_point_ages['Point_timeslices']

data_indices=np.load('indices_por_anos_3_filt.npz')


# Puntos que quieres destacar
indices_med=data_indices["indices_med"]
indices_pac=data_indices["indices_pac"]
indices_car=data_indices["indices_car"]


#If you want to use diversity data
#data_D=np.load("datos_finales_indicios_7param.npz")
#D_shelf=data_D["D_shelf"]
#print(D_shelf.shape)
#input("stop")

#If you want to use food data or temperature data
data_food_temp=scipy.io.loadmat('Point_foodtemp_v241023.mat')
temper_shelf=data_food_temp['temp_shelf']
food_shelf=data_food_temp['food_shelf']
temp_shelf=data_food_temp['temp_shelf']

data_color_br = scipy.io.loadmat("mycolormap_br.mat")
colors_br=data_color_br["mycolormap_br"]
br = ListedColormap(colors_br, name="mi_cmap_br")

# Crear figura y ejes una sola vez
fig, ax = plt.subplots(figsize=(8, 6))

title = ax.set_title(f'Time 0 Mya ago')

# Fijar límites de la vista según los datos válidos
valid_longs = shelf_lonlatAge[:, :, 0][~np.isnan(shelf_lonlatAge[:, :, 0])]
valid_lats = shelf_lonlatAge[:, :, 1][~np.isnan(shelf_lonlatAge[:, :, 1])]

ax.set_xlim(valid_longs.min(), valid_longs.max())
ax.set_ylim(valid_lats.min(), valid_lats.max())

# Crear scatter de todos los puntos (azul, pequeño)
scatter = ax.scatter([], [], s=1, alpha=0.5, color='lightgray')

# Crear scatter de múltiples puntos destacados (rojo, grande)
highlight_1 = ax.scatter([], [], c=[], s=10, alpha=1.0, cmap=br, norm=mcolors.LogNorm(vmin=0.01, vmax=10), #norm=LogNorm(vmin=1, vmax=50,base=2), 
                         edgecolor='black', linewidths=0.1, label='Mediterranean points')
highlight_2 = ax.scatter([], [], c=[], s=10, alpha=1.0, cmap=br, norm=mcolors.LogNorm(vmin=0.01, vmax=10),#norm=LogNorm(vmin=1, vmax=50, base=2), 
                         edgecolor='black', linewidths=0.1, label='Pacific points')
highlight_3 = ax.scatter([], [], c=[], s=10, alpha=1.0, cmap=br, norm=mcolors.LogNorm(vmin=0.01, vmax=10),#norm=LogNorm(vmin=1, vmax=50, base=2), 
                         edgecolor='black', linewidths=0.1, label='Caribean points')


#ax.legend(loc="upper right")



# Loop de tiempo hacia atrás
def update(frame):

    t = shelf_lonlatAge.shape[1] - 1 - frame

    
    if t <=0:
        t = 1
    longitudes = shelf_lonlatAge[:, t, 0]
    latitudes = shelf_lonlatAge[:, t, 1]

    # Actualizar todos los puntos
    scatter.set_offsets(np.c_[longitudes, latitudes])
    #scatter = ax.scatter(longitudes, latitudes, s=1, alpha=0.5, color='lightgray')

    

    # Coordenadas de los puntos destacados
    coords_med = np.array([[shelf_lonlatAge[int(i), t, 0], shelf_lonlatAge[int(i), t, 1]] for i in indices_med[t] if not np.isnan(i)])
    valid_idx = np.array([int(i) for i in indices_med[t] if not np.isnan(i)], dtype=int)  
    #D_sub = D_shelf[1:, valid_idx, t, :]   # shape → (41, N, 6)
    #D_med = np.nanmean(D_sub, axis=(0, 2))  # shape → (N,)  
    food_med = food_shelf[valid_idx, t]  # shape → (N,)
    #temp_med = temper_shelf[valid_idx, t]  # shape → (N,)
    
    highlight_1.set_offsets(coords_med)
    highlight_1.set_array(food_med)
    #highlight_1 = ax.scatter(coords_med[0], coords_med[1], s=10, alpha=1.0, color='red', edgecolor='black', linewidths=0.3, label='Mediterranean points')

    coords_pac = np.array([[shelf_lonlatAge[int(i), t, 0], shelf_lonlatAge[int(i), t, 1]] for i in indices_pac[t] if not np.isnan(i)])
    highlight_2.set_offsets(coords_pac)
    valid_idx = np.array([int(i) for i in indices_pac[t] if not np.isnan(i)], dtype=int)  
    #D_sub = D_shelf[1:, valid_idx, t, :]   # shape → (41, N, 6)
    #D_pac = np.nanmean(D_sub, axis=(0, 2))  # shape → (N,)  
    food_pac = food_shelf[valid_idx, t]  # shape → (N,)
    #temp_pac = temp_shelf[valid_idx, t]  # shape → (N,)
    highlight_2.set_array(food_pac)

    #highlight_2 = ax.scatter(coords_pac[0], coords_pac[1], s=10, alpha=1.0, color='red', edgecolor='black', linewidths=0.3, label='Caribbean points')


    ind_car=indices_car[t]
    if len(ind_car[~np.isnan(ind_car)])!=0:
        coords_car = np.array([[shelf_lonlatAge[int(i), t, 0], shelf_lonlatAge[int(i), t, 1]] for i in  indices_car[t] if not np.isnan(i)])      
    else:
        coords_car = np.empty((0, 2))
    highlight_3.set_offsets(coords_car)
    valid_idx = np.array([int(i) for i in indices_car[t] if not np.isnan(i)], dtype=int)  
    #D_sub = D_shelf[1:, valid_idx, t, :]   # shape → (41, N, 6)
    #D_car = np.nanmean(D_sub, axis=(0, 2))  # shape → (N,)  
    food_car = food_shelf[valid_idx, t]  # shape → (N,)
    #temp_car = temp_shelf[valid_idx, t]  # shape → (N,)
    highlight_3.set_array(food_car)
    #highlight_3 = ax.scatter(coords_car[0], coords_car[1], s=10, alpha=1.0, color='red', edgecolor='black', linewidths=0.3, label='Pacific points')



    # Actualizar título
    title.set_text(f'Time {Point_timeslices[0][t]} Mya ago')
    return scatter, highlight_1, highlight_2, highlight_3, title

    #if t % 5 == 0:
    #    fig.savefig(f"map_index/frame_t{t}.png", dpi=150)
# Crear animación

cbar = plt.colorbar(highlight_2, ax=ax)
cbar.set_label("Food (POC)", rotation=270, labelpad=15)
ani = FuncAnimation(fig, update, frames=33, interval=500, blit=False)

ani.save("food_ind.gif", writer=PillowWriter(fps=2))

plt.show()
