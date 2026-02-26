import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def R(z, A, b):
    s = len(b)
    I = np.eye(s)
    e = np.ones(s)
    R = np.zeros_like(z, dtype=complex)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            zi = z[i,j]
            R[i,j] = 1 + zi * b @ np.linalg.solve(I - zi*A, e)
    return R


gamma = 1 - (1/np.sqrt(2))
print(gamma)

#DIRK
A_fe = np.array([[gamma, 0], [1-2*gamma, gamma]])
b_fe = np.array([1/2, 1/2])




x = np.linspace(-10,10,300)
y = np.linspace(-10,10,300)
X,Y = np.meshgrid(x,y)
Z = X + 1j*Y


dirk = np.abs(R(Z,A_fe,b_fe))

custom_lines = [
    Line2D([0], [0], color = "lightblue", lw =2),
]

plt.figure()
plt.contourf(X,Y,dirk,[0,1], colors = "lightblue", alpha = 0.3)
plt.contour(X,Y,dirk,[1], colors = "lightblue")


labels = ["DIRK"]

plt.axhline(0, alpha = 0.2, color = "black")
plt.axvline(0, alpha = 0.2, color = "black")
plt.title("Regions of Absolute Stability")
plt.xlabel("Re(z)")
plt.ylabel("Im(z)")
plt.grid(alpha = 0.3)
plt.legend(custom_lines, labels)
plt.savefig("prob1bc_hw3.svg")
plt.show()