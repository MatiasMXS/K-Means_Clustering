import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("iris.csv")  # sin header=None

Centroides=np.array([
    [5.9,1.9],
    [4.3,1.6],
    [1.7,0.4],
    ])

X = data[["PetalLengthCm", "PetalWidthCm"]].values
labels = np.zeros(X.shape[0])
for iter in range(10):  # o más si querés
    # Asignar cada punto al centroide más cercano
    for i in range(len(X)):
        distancias = [np.sqrt((Centroides[j][0] - X[i][0])**2 + (Centroides[j][1] - X[i][1])**2) for j in range(3)]
        labels[i] = np.argmin(distancias)

    # Actualizar centroides con los promedios de sus puntos asignados
    for j in range(3):
        puntos_del_cluster = X[labels == j]
        if len(puntos_del_cluster) > 0:
            Centroides[j] = np.mean(puntos_del_cluster, axis=0)

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

