import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def build(n):
    h = 2.0 / (n + 1)
    xi = np.linspace(-1.0, 1.0, n + 2)

    def rhs(t, y):
        U_in = y[:n]
        xL, xR = y[n], y[n+1]

        U = np.zeros(n + 2)
        U[1:n+1] = U_in

        xf = 0.5 * (xR - xL)
        xf = max(xf, 1e-12)

        D_L = (4.0*U[1] - U[2])/(2*h)
        D_R = (-4.0*U[n] + U[n-1])/(2*h)

        dU = np.zeros(n)
        for j in range(1, n + 1):
            ux = (U[j + 1] - U[j - 1])/(2*h)
            uxx = (U[j + 1] - 2*U[j] + U[j - 1]) / (h * h)
            adv_coeff = -0.5 * ((1.0 + xi[j])*D_R + (1.0 - xi[j])*D_L)
            dU[j - 1] = (adv_coeff * ux + U[j]*uxx + ux*ux)/(xf * xf)

        dxL = -D_L / xf
        dxR = -D_R / xf
        return np.concatenate([dU, [dxL, dxR]])
    return rhs, xi
def initial_condition_1(x):
    return np.where(np.abs(x)<1.0, 1 - x**2, 0.0)
def initial_condition_2(x):
    return np.where(np.abs(x)<1.0, 1 - 0.99*np.cos(2*np.pi * x), 0.0)


def solve_problem(u0_fun, n=199, t0=0.0, tf=1.2):
    rhs, xi = build(n)
    U0 = u0_fun(xi[1:-1])
    y0 = np.concatenate([U0, [-1.0, 1.0]])
    t_eval = np.round(np.arange(0, tf + 1e-12, 0.1), 10)

    sol = solve_ivp(
        rhs, [t0, tf], y0,
        method="BDF",
        t_eval=t_eval,
        rtol=1e-6, atol=1e-8, max_step=0.02)
    return sol, xi

def physical_grid(xi, xL, xR):
    x0 = 0.5 * (xL + xR)
    xf = 0.5 * (xR - xL)
    return x0 + xf * xi

def plot_soln(sol, xi, title_prefix):
    n = len(xi) - 2
    times = sol.t

    plt.figure(figsize=(8,5))
    for k, t in enumerate(times[1:]): 
        y = sol.y[:, k + 1]
        U_in = y[:n]
        xL, xR = y[n], y[n + 1]
        U = np.zeros(n + 2)
        U[1:n+1] = U_in
        x = physical_grid(xi, xL, xR)
        plt.plot(x, U, label=f"t={t:.2f}")
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.title(f"{title_prefix}: solution in x coordinates")
    plt.legend(ncol=3, fontsize=8)
    plt.grid(True,alpha=0.2)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,5))
    xi_dense = np.linspace(-1.0, 1.0, 500)
    plt.plot(xi_dense, 1.0 - xi_dense**2, "--", label=r"ref $1-\xi^2$")
    for k, t in enumerate(times[1:]):
        y = sol.y[:, k + 1]
        U_in = y[:n]
        U = np.zeros(n + 2)
        U[1:n+1] = U_in
        umax = np.max(U)
        if umax > 0:
            plt.plot(xi, U / umax, label=f"t={t:.2f}")
    plt.xlabel(r"$\xi$")
    plt.ylabel(r"$u(\xi,t)/u_{\max}(t)$")
    plt.title(f"{title_prefix}: renorm. profiles")
    plt.legend(ncol=3, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

sol1, xi = solve_problem(initial_condition_1, n=199)
plot_soln(sol1, xi, "Initial condition 1")

sol2, xi = solve_problem(initial_condition_2, n=199)
plot_soln(sol2, xi, "Initial condition 2")