import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("log/snn_mountain_car_pareto/1020.out", delimiter=",")
fitness, fitmax, nconn, _ = data.T

plt.scatter(nconn, fitness, alpha=0.5, s=10)
plt.xlabel("nConn")
plt.ylabel("Mean Fitness")
plt.title("Población gen 100 — eje X: conectividad, Y: fitness medio")
plt.show()