import numpy as np
import matplotlib.pyplot as plt



def f(Z):
    x, y, u, v = Z

    r_2 = x**2 + y**2

    return np.array([u, v, -x/r_2, -y/r_2])

init_state = np.array([1, 0, 0, 1])

N_list = [20, 40, 80]

norms = []

fig, axs = plt.subplots(1, 3, figsize=(20, 5), layout = "constrained")
colors = ["red", "purple", "lightgreen"]

for i, N in enumerate(N_list):

    h = 2*np.pi/N

    T = 8*np.pi

    T_Steps = int(T/h)

    Y = np.zeros((T_Steps+1, len(init_state)))

    Y[0] = init_state

    for n in range(T_Steps):
        y_p = Y[n] + h*f(Y[n])
        
        mid_pt = (Y[n] + y_p)/2

        Y[n+1] = Y[n] + h*f(mid_pt)

    norms.append(np.sqrt(Y[-1,0]**2 + Y[-1,1]**2))

    axs[i].plot(Y[:,0],Y[:,1], color = colors[i], label = f"Norm of {norms[i]}")
    axs[i].set_title(f"Solution X,Y for N = {N}")
    axs[i].set_xlabel("X")
    axs[i].set_ylabel("Y")
    axs[i].grid(True)
    axs[i].legend(loc = "lower center")


plt.savefig("prob5d.svg")
plt.show()

