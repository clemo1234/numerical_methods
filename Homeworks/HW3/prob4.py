import numpy as np
import time
import matplotlib.pyplot as plt



def f(u, mu):
    x, y = u
    return np.array([
        y,
        mu * (1 - x**2) * y - x
    ])

def jacobian(u, mu):
    x, y = u
    
    return np.array([
        [0.0, 1.0],
        [-2*mu*x*y - 1.0, mu*(1 - x**2)]])

gamma = 1.0 - (1.0/np.sqrt(2))



def solve_stage(U,rhs,h, a, mu, tol=1e-10, maxit=20):
    for _ in range(maxit):
        F = U - rhs - h*a*f(U, mu)
        J = np.eye(2) - h*a*jacobian(U, mu)
        delta = np.linalg.solve(J, -F)
        U = U + delta
        if np.linalg.norm(delta) < tol:
            break
    return U

def dirk2_step(u,h, mu):

    U1 = solve_stage(u.copy(), u, h, gamma, mu)
    k1 = f(U1, mu)

    rhs2 = u + h*(1-gamma)*k1
    U2 = solve_stage(U1.copy(), rhs2, h, gamma, mu)
    k2 = f(U2, mu)


    u2 = u + h*((1-gamma)*k1 + gamma*k2)
    u1 = u + h*k1

    err = np.linalg.norm(u2 - u1)

    return u2, err

def run_fixed(mu, h, T):
    t = 0.0
    u = np.array([2.0, 0.0])

    ts = [t]
    xs = [u[0]]
    ys = [u[1]]

    start = time.time()

    while t < T:
        u, _ = dirk2_step(u, h, mu)
        t += h

        ts.append(t)
        xs.append(u[0])
        ys.append(u[1])

    cpu = time.time() - start
    return np.array(ts), np.array(xs), np.array(ys), cpu

def run_adaptive(mu, T,atol=1e-5, rtol=1e-5,h0=1e-3):

    t = 0.0
    u = np.array([2.0, 0.0])
    h = h0

    ts = [t]
    xs = [u[0]]
    ys = [u[1]]

    start = time.time()

    while t < T:

        if t + h > T:
            h = T - t

        u_new, err = dirk2_step(u, h, mu)
        
        tol = atol + rtol*np.linalg.norm(u_new)
        #logic for algorithm (part (b))
        if err <= tol:
            t = t + h
            u = u_new
            ts.append(t)
            xs.append(u[0])
            ys.append(u[1])

        if err == 0:
            factor = 5.0
        else:
            factor = 0.9*(tol/err)**0.5
            factor = min(5.0, max(0.2, factor))

        h *= factor

    cpu_times = time.time() - start
    
    return np.array(ts), np.array(xs), np.array(ys), cpu_times

ts, xs, ys, cpu_times = run_fixed(mu=1e2, h=1e-3, T=200)


print("CPU fixed:", cpu_times)

plt.figure()
plt.plot(ts, xs, label="x")
plt.plot(ts, ys, label="y")
plt.xlabel(r"t")
plt.legend()
plt.title(r"Time series Soln ($mu$=1e2)")
plt.grid(True, alpha =0.3)
plt.savefig("prob4_hw3_1.svg")
plt.show()

plt.figure()
plt.plot(xs, ys)
plt.title("Phase plot (xy)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True, alpha =0.3)
plt.savefig("prob4_hw3_2.svg")
plt.show()

ts, xs, ys, cpu_times = run_fixed(mu=1e3, h=1e-3, T=2e3)


print("CPU fixed:", cpu_times)

plt.figure()
plt.plot(ts, xs, label="x")
plt.plot(ts, ys, label="y")
plt.xlabel(r"t")
plt.legend()
plt.title(r"Time series Soln ($mu$=1e3)")
plt.grid(True, alpha =0.3)
plt.savefig("prob4_hw3_1-1.svg")
plt.show()

plt.figure()
plt.plot(xs, ys)
plt.title("Phase plot (xy)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True, alpha =0.3)
plt.savefig("prob4_hw3_2-1.svg")
plt.show()


ts, xs, ys, cpu_times2 = run_adaptive(mu=1e6, T=2e6)
print("CPU adaptive:", cpu_times2)

plt.figure()
plt.plot(ts, xs, label="x")
plt.plot(ts, ys, label="y")
plt.legend()
plt.title("Adaptive Soln")
plt.grid(True, alpha =0.3)
plt.savefig("prob4_hw3_3.svg")

plt.show()

plt.figure()
plt.plot(xs, ys)
plt.title("Adaptive method phase plot (xy)")
plt.xlabel(r"t")
plt.grid(True, alpha =0.3)
plt.savefig("prob4_hw3_4.svg")
plt.show()