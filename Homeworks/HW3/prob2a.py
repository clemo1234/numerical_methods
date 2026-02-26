import numpy as np
import matplotlib.pyplot as plt

L = 1e4

gamma2 = 1 - 1/np.sqrt(2)
gamma3 = 1/2 + np.sqrt(3)/6

def phi(t):
    return np.sin(t + np.pi/4)

def dphi(t):
    return np.cos(t + np.pi/4)

def f(t,y):
    return -L*(y-phi(t)) + dphi(t)

def y_exact(t, y0):
    return np.exp(-L*t)*(y0 - phi(0)) + phi(t)

def dirk2(t,y,h):

    t1 = t + gamma2*h
    k1 = (-L*(y - phi(t1)) + dphi(t1))/ (1 + L*h*gamma2)

    t2 = t + h

    k2 = (-L*((y + h*k1*(1-gamma2)) - phi(t2)) + dphi(t2))/(1 + L*h*gamma2)

    y_step = y + h*(k1*(1-gamma2) + gamma2*k2)

    return y_step


def dirk3(t,y,h):
    t1 = t + gamma3*h
    k1 = (-L*(y - phi(t1)) + dphi(t1))/ (1 + L*h*gamma3)

    t2 = t + h*(1-gamma3)

    k2 = (-L*((y + h*k1*(1-2*gamma3)) - phi(t2)) + dphi(t2))/(1 + L*h*gamma3)

    y_step = y + h*(k1*(1/2) + (1/2)*k2)

    return y_step

def solver(method, y0, h, Tmax):
    t = 0
    y = y0
    N_steps = int(Tmax/h)

    times = []
    solns = []

    for n in range(N_steps):
        times.append(t)
        solns.append(y)

        y = method(t,y,h)

        t = t + h

    return np.array(times), np.array(solns)

d = 5/24

p_array = np.arange(1,6+d, d)

h_values = 10**(-p_array)

y0 = np.sin(np.pi/4) + 10

errors2 = []
errors3 = []

t_max = 10

for h in h_values:

    t, y = solver(dirk2, y0, h, t_max)
    y_exact_n = y_exact(t, y0)

    errors2.append(np.max(np.abs(y-y_exact_n)))

    t, y = solver(dirk3, y0, h, t_max)
    y_exact_n = y_exact(t, y0)

    errors3.append(np.max(np.abs(y-y_exact_n)))

C1 = 100000000e-5
C2 = 150000000e-2


plt.loglog(h_values, errors2, "o-", label = "DIRK2", color = "purple")
plt.loglog(h_values, errors3, "o-", label = "DIRKo3", color = "darkblue")

plt.loglog(h_values, C1*h_values, "--", label = r"$h$", color = "red")
plt.loglog(h_values, C2*h_values**2, "--",label= r"$h^2$", color = "cyan")

plt.grid(True, alpha = 0.3)
plt.title("Errors for DIRK2 and DIRKo3")
plt.xlabel(r"$\log(h)$")
plt.ylabel(r"$\log(\max(e))$")
plt.legend()
plt.savefig("prob2b_hw3_3.svg")
plt.show()
