import numpy as np
import matplotlib.pyplot as plt


def f(u):
    return 0.5*u*u

def u0(x):
    return np.where(x < 0, 0.0,
           np.where(x < 1, 2.0,
           np.where(x < 2, 1.0, 0.0)))

def exact_solution(x, t):
    x = np.asarray(x)
    u = np.zeros_like(x, dtype=float)

    if t == 0:
        return u0(x)

    if t < 1:
        x_r = 2*t
        x_s1 = 1 + 1.5*t
        x_s2 = 2 + 0.5*t

        mask = (0 < x) & (x < x_r)
        u[mask] = x[mask]/t

        mask = (x_r <= x) & (x < x_s1)
        u[mask] = 2.0

        mask = (x_s1 <= x) & (x < x_s2)
        u[mask] = 1.0

    elif t < 1.5:
        x_r = 2*t
        x_s = t + 1.5
        mask = (0 < x) & (x < x_r)
        u[mask] = x[mask]/t
        mask = (x_r <= x) & (x < x_s)
        u[mask] = 2.0

    else:
        x_s = np.sqrt(6*t)
        mask = (0 < x) & (x < x_s)
        u[mask] = x[mask]/t

    return u

def lax_friedrichs_step(U, lam):
    Un = U.copy()
    Un[1:-1] = 0.5*(U[2:] + U[:-2]) - 0.5*lam*(f(U[2:]) - f(U[:-2]))
    Un[0] = 0.0
    Un[-1] = 0.0
    return Un

def richtmyer_step(U, lam):
    
    U_half = 0.5*(U[1:] + U[:-1]) - 0.5*lam*(f(U[1:]) - f(U[:-1]))
    Un = U.copy()
    Un[1:-1] = U[1:-1] - lam*(f(U_half[1:]) - f(U_half[:-1]))
    Un[0] = 0.0
    Un[-1] = 0.0
    return Un

def maccormack_step(U, lam):
    
    U_star = U.copy()
    U_star[:-1] = U[:-1] - lam*(f(U[1:]) - f(U[:-1]))
    U_star[-1] = 0.0
    Un = U.copy()
    Un[1:-1] = 0.5*(U[1:-1] + U_star[1:-1]
                    - lam*(f(U_star[1:-1]) - f(U_star[:-2])))
    Un[0] = 0.0
    Un[-1] = 0.0
    return Un

def solve(method_step, x, h, k, output_times):
    lam = k/h
    U = u0(x)
    results = {}
    t = 0.0
    for T in output_times:
        while t < T - 1e-14:
            dt = min(k, T - t)
            U = method_step(U, dt/h)
            t += dt
        results[T] = U.copy()
    return results


output_times = [0.5, 1.5, 2.5, 3.5, 5.0]

xmin, xmax = -1.0, 7.0
h = 0.005
x = np.arange(xmin, xmax + h, h)
k = 0.45*h

methods = {
    "Lax-Friedrichs": lax_friedrichs_step,
    "Richtmyer": richtmyer_step,
    "MacCormack": maccormack_step,
}


for name, step in methods.items():
    sol = solve(step, x, h, k, output_times)

    plt.figure(figsize=(9, 5.6))
    for T in output_times:
        plt.plot(x, sol[T], linewidth=1.4, label=fr"{name}, $t={T}$")
        plt.plot(x, exact_solution(x, T), "--", linewidth=1.0, label=fr"exact, $t={T}$")

    plt.title(fr"Burgers equation: {name} vs exact soln")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u(x,t)$")
    plt.ylim(-0.25, 2.35)
    plt.xlim(xmin, xmax)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{name}_prob2.pdf")
    plt.show()

plt.figure(figsize=(9, 5.6))
for T in output_times:
    plt.plot(x, exact_solution(x, T), linewidth=1.8, label=fr"exact, $t={T}$")

plt.title("Burgers equation: exact entropy solution")
plt.xlabel(r"$x$")
plt.ylabel(r"$u(x,t)$")
plt.ylim(-0.25, 2.35)
plt.xlim(xmin, xmax)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"exact_prob2.pdf")
plt.show()


