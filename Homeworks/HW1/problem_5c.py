import numpy as np
import matplotlib.pyplot as plt

#Parameters
a0, a1, b0, b1 = (-4, 5, 4, 2)

def exact_soln(t):
    return np.array([np.cos(t), np.sin(t), -np.sin(t), np.cos(t)])

def f(Z):
    x, y, u, v = Z

    r_2 = x**2 + y**2

    return np.array([u, v, -x/r_2, -y/r_2])

init_state = np.array([1, 0, 0, 1])
colors = ["red", "purple", "lightgreen"]

N_list = [20, 40, 80]

norms = []

fig, axs = plt.subplots(1, 3, figsize=(20, 5), layout = "constrained")

for i, N in enumerate(N_list):

    h = 2*np.pi/N

    T = 4*np.pi

    T_Steps = int(T/h)

    Y = np.zeros((T_Steps+1, len(init_state)))

    Y[0] = init_state
    Y[1] = exact_soln(h)

    for n in range(1, T_Steps):
        Y[n+1] = a0*Y[n] + a1*Y[n-1] + h*(b0*f(Y[n]) + b1*f(Y[n-1]))

    norms.append(np.sqrt(Y[-1,0]**2 + Y[-1,1]**2))

    axs[i].plot(Y[:,0],Y[:,1], color = colors[i], label = f"Norm of {norms[i]}")
    axs[i].set_title(f"Solution X,Y for N = {N}")
    axs[i].set_xlabel("X")
    axs[i].set_ylabel("Y")
    axs[i].grid(True)
    axs[i].legend(loc = "lower center")

plt.savefig("prob5c.svg")
plt.show()

