import numpy as np
import matplotlib.pyplot as plt

def flux(u):
    return 0.5 * u**2


def initial_condition(x):
    return np.where((x >= 0.0) & (x <= 1.0), 1.0, 0.0)


def exact_solution(x, t):

    u = np.zeros_like(x)
    if t == 0:
        return initial_condition(x)

    if t <= 2.0:

        mask_fan = (x > 0.0) & (x < t)
        u[mask_fan] = x[mask_fan] / t
        mask_const = (x >= t) & (x <= 1.0 + 0.5 * t)
        u[mask_const] = 1.0

    else:
        xs = np.sqrt(2.0 * t)
        mask_fan = (x > 0.0) & (x < xs)
        u[mask_fan] = x[mask_fan] / t

    return u

def lax_friedrichs_flux(uL, uR, h, k):
    return 0.5 * (flux(uL) + flux(uR)) - 0.5 * (h/k) * (uR - uL)


def richtmyer_flux(uL, uR, h, k):

    lam = k / h
    u_half = 0.5 * (uL + uR) - 0.5 * lam * (flux(uR) - flux(uL))
    return flux(u_half)


def godunov_flux(uL, uR):

    F = np.zeros_like(uL)

    rare = uL <= uR
    shock = ~rare

    mask = rare & (uL >= 0)
    F[mask] = flux(uL[mask])


    mask = rare & (uR <= 0)
    F[mask] = flux(uR[mask])
    mask = rare & (uL < 0) & (uR > 0)
    F[mask] = 0.0
    s = 0.5 * (uL + uR)

    mask = shock & (s >= 0)
    F[mask] = flux(uL[mask])

    mask = shock & (s < 0)
    F[mask] = flux(uR[mask])

    return F


def conservative_update(U, h, k, method):
    Uext = np.zeros(len(U) + 2)
    Uext[1:-1] = U
    Uext[0] = 0.0
    Uext[-1] = 0.0

    uL = Uext[:-1]
    uR = Uext[1:]

    if method == "Lax-Friedrichs":
        F = lax_friedrichs_flux(uL, uR, h, k)

    elif method == "Richtmyer":
        F = richtmyer_flux(uL, uR, h, k)

    elif method == "Godunov":
        F = godunov_flux(uL, uR)

    return U - (k / h) * (F[1:] - F[:-1])


def maccormack_update(U,h,k):

    lam = k/h

    Uext = np.zeros(len(U) + 2)
    Uext[1:-1] = U
    Uext[0] = 0.0
    Uext[-1] = 0.0

    Ustar = U - lam * (flux(Uext[2:]) - flux(Uext[1:-1]))

    Ustarext = np.zeros(len(Ustar) + 2)
    Ustarext[1:-1] = Ustar
    Ustarext[0] = 0.0
    Ustarext[-1] = 0.0
    Unew = 0.5 * (
        U + Ustar
        - lam * (flux(Ustarext[1:-1]) - flux(Ustarext[:-2]))
    )

    return Unew


def solve(method, x, h, k, final_time, output_times):
    U = initial_condition(x)

    snapshots = {}
    t = 0.0

    for tout in output_times:
        while t < tout - 1e-12:
            dt = min(k, tout - t)

            if method == "MacCormack":
                U = maccormack_update(U, h, dt)
            else:
                U = conservative_update(U, h, dt, method)

            t += dt

        snapshots[tout] = U.copy()

    return snapshots

x_min = -1.0
x_max = 5.0
h = 0.01

x = np.arange(x_min, x_max + h, h)

CFL = 0.8
max_speed = 1.0
k = CFL * h / max_speed

output_times = [0, 1, 2, 3, 4, 5, 6]

methods = [
    "Lax-Friedrichs",
    "Richtmyer",
    "MacCormack",
    "Godunov",
]

all_solutions = {}

for method in methods:
    all_solutions[method] = solve(
        method=method,
        x=x,
        h=h,
        k=k,
        final_time=max(output_times),
        output_times=output_times,
    )


for method in methods:
    plt.figure(figsize=(9, 5))

    for t in output_times:
        plt.plot(x, all_solutions[method][t], label=f"{method}, t={t}")

    for t in output_times:
        plt.plot(x, exact_solution(x, t), "k--", linewidth=1.0, alpha=0.75)

    plt.title(f"{method} method for Burgers equation")
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.ylim(-0.2, 1.2)
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{method}_prob4.pdf")
    plt.show()


plt.figure(figsize=(11, 8))

for i, method in enumerate(methods, start=1):
    plt.subplot(2, 2, i)
    plt.plot(x, all_solutions[method][6], label=method, color = "red")
    plt.plot(x, exact_solution(x, 6), "k--", label="Exact")
    plt.title(f"{method}, t=6")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.ylim(-0.2, 1.2)
    plt.grid(True)
    plt.legend()
plt.savefig(f"plot_summary.pdf")
plt.tight_layout()
plt.show()