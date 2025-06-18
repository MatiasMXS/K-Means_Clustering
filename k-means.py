import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Leer el dataset con codificación compatible
data = pd.read_csv("Iris.csv")  # o "cp1252"

# Inicializar 2 centroides (SepalWidthCm, PetalLengthCm)
Centroides=np.array([
    [5.9,1.9],
    [4.2,1.6],
    [6.7,2.2],
    ])

X = data[["PetalLengthCm", "PetalWidthCm"]].values
labels = np.zeros(X.shape[0])
iteracion_final = 0
for iter in range(100):  # máximo de 100 iteraciones por seguridad
    centroides_previos = Centroides.copy()

    # Asignar cada punto al centroide más cercano
    for i in range(len(X)):
        distancias = [np.sqrt((Centroides[j][0] - X[i][0])**2 + (Centroides[j][1] - X[i][1])**2) for j in range(3)]
        labels[i] = np.argmin(distancias)

    # Actualizar centroides
    for j in range(3):
        puntos_del_cluster = X[labels == j]
        if len(puntos_del_cluster) > 0:
            Centroides[j] = np.mean(puntos_del_cluster, axis=0)

    # Cortar si los centroides no cambian
    if np.allclose(Centroides, centroides_previos):
        print(f"Convergió en la iteración {iter+1}")
        break

for j in range(2):
    print(f"Cordenadas del centroide {j+1} x={Centroides[j][0]} Y={Centroides[j][1]}")
# Colores para cada cluster
colores = ['blue', 'green', 'red']

# Graficar cada cluster con su color
for i in range(3):
    puntos = X[labels == i]
    plt.scatter(puntos[:, 0], puntos[:, 1], c=colores[i], label=f'Cluster {i+1}')

# Graficar centroides
plt.scatter(Centroides[:, 0], Centroides[:, 1], c='black', marker='X', s=200, label='Centroides')

plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("K-Means Clustering - Iris Dataset")
plt.legend()
plt.grid(True)
plt.show()