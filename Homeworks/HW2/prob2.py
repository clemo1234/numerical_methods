import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def R(z, A, b):
    s = len(b) #size of b vector
    I = np.eye(s)
    e = np.ones(s)
    R = np.zeros_like(z, dtype=complex)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            zi = z[i,j]
            R[i,j] = 1 + zi * b @ np.linalg.solve(I - zi*A, e) #implementing eq. (11) on PDF
    return R


#From the tables

# FE

A_fe = np.array([[0.0]])
b_fe = np.array([1.0])

# Explicit midpoint rule
A_mid = np.array([[0,0],[1/2,0]])
b_mid = np.array([0,1])

# Kuttas 3rd order
A_k3 = np.array([[0,0,0],[1/2,0,0],[-1,2,0]])
b_k3 = np.array([1/6,2/3,1/6])

# RK4
A_rk4 = np.array([[0,0,0,0],
                  [1/2,0,0,0],
                  [0,1/2,0,0],
                  [0,0,1,0]])
b_rk4 = np.array([1/6,1/3,1/3,1/6])

# DOPRI5
A_dp = np.array([
[0,0,0,0,0,0,0],
[1/5,0,0,0,0,0,0],
[3/40,9/40,0,0,0,0,0],
[44/45,-56/15,32/9,0,0,0,0],
[19372/6561,-25360/2187,64448/6561,-212/729,0,0,0],
[9017/3168,-355/33,46732/5247,49/176,-5103/18656,0,0],
[35/384,0,500/1113,125/192,-2187/6784,11/84,0]
])

b_dp = np.array([35/384,0,500/1113,125/192,-2187/6784,11/84,0])


x = np.linspace(-4,2,300)
y = np.linspace(-4,4,300)
X,Y = np.meshgrid(x,y)
Z = X + 1j*Y


R_foward_euler = np.abs(R(Z,A_fe,b_fe))
R_mid = np.abs(R(Z,A_mid,b_mid))
R_k3 = np.abs(R(Z,A_k3,b_k3))
R_rk4 = np.abs(R(Z,A_rk4,b_rk4))
R_dp = np.abs(R(Z,A_dp,b_dp))

custom_lines = [
    Line2D([0], [0], color = "red", lw =2),
    Line2D([0], [0], color = "gold", lw =2),
    Line2D([0], [0], color = "cyan", lw =2),
    Line2D([0], [0], color = "green", lw =2),
    Line2D([0], [0], color = "orange", lw =2)
]

plt.figure()
plt.contourf(X,Y,R_foward_euler,[0,1], colors = "red", alpha = 0.3)
plt.contourf(X,Y,R_mid,[0,1], colors = "gold", alpha = 0.3)
plt.contourf(X,Y,R_k3,[0,1], colors = "cyan", alpha = 0.3)
plt.contourf(X,Y,R_rk4,[0,1], colors = "green", alpha = 0.3)
plt.contourf(X,Y,R_dp,[0,1], colors = "orange", alpha = 0.3)

plt.contour(X,Y,R_foward_euler,[1], colors = "red")
plt.contour(X,Y,R_mid,[1], colors = "gold")
plt.contour(X,Y,R_k3,[1], colors = "cyan")
plt.contour(X,Y,R_rk4,[1], colors = "green")
plt.contour(X,Y,R_dp,[1], colors = "orange")

labels = ["Foward Euler", "Midpoint Rule with Euler Predictor", "RK3", "RK4", "DOPRI5(4)"]

plt.axhline(0, alpha = 0.2, color = "black")
plt.axvline(0, alpha = 0.2, color = "black")
plt.title("Regions of Absolute Stability")
plt.xlabel("Re(z)")
plt.ylabel("Im(z)")
plt.grid(alpha = 0.3)
plt.legend(custom_lines, labels)
plt.savefig("prob2.svg")
plt.show()