import numpy as np
import time
import matplotlib.pyplot as plt

def robertson(t, u):
    x, y, z = u
    a = 0.04
    b = 1.0e4
    c = 3.0e7
    dx = -a*x + b*y*z
    dy = a*x - b*y*z - c*y**2
    dz = c*y**2
    return np.array([dx, dy, dz])


def jacobian(y):
    a = 0.04
    b = 1.0e4
    c = 3.0e7
    by = b*y[1]
    bz = b*y[2]
    c2y = 2*c*y[1]
    Jac = np.zeros((3,3))
    Jac[0,0] = -a
    Jac[0,1] = bz
    Jac[0,2] = by
    Jac[1,0] = a
    Jac[1,1] = -bz-c2y
    Jac[1,2] = -by
    Jac[2,1] = c2y
    return Jac

    
def newton(F, J, u0, tol=1e-10, maxit=20):
    u = u0.copy()
    for _ in range(maxit):
        Fu = F(u)
        if np.linalg.norm(Fu) < tol:
            break
        Ju = J(u)
        delta = np.linalg.solve(Ju, -Fu)
        u += delta
    return u

gamma = 1/2 + np.sqrt(3)/6

A = np.array([
    [gamma, 0, 0],
    [1-gamma, gamma, 0],
    [0.5, 0.5, 0]
])
b = np.array([0.5, 0.5, 0])
c = np.array([gamma, 1, 1])

def dirk3(f, Jf, t0, tf, u0, h):
    N = int((tf - t0)/h)
    t = t0
    u = u0.copy()
    sol = [u.copy()]

    for _ in range(1,N):
        k = []

        for i in range(3):
            def F(stage):
                s = u + h*sum(A[i,j]*k[j] for j in range(i)) + h*A[i,i]*stage
                return stage - f(t + c[i]*h, s)

            def J(stage):
                s = u + h*sum(A[i,j]*k[j] for j in range(i)) + h*A[i,i]*stage
                return np.eye(3) - h*A[i,i]*Jf(s)

            ki = newton(F, J, f(t, u))
            k.append(ki)

        u = u + h*sum(b[i]*k[i] for i in range(3))
        t = t + h
        sol.append(u.copy())

    return np.array(sol)

gamma2 = 1 - 1/np.sqrt(2)

#butcher arrray for dirk2
A2 = np.array([[gamma2, 0],
               [1-gamma2, gamma2]])
b2 = np.array([0.5, 0.5])
c2 = np.array([gamma2, 1])

def dirk2_step(f, Jf, t, u, h):
    k = []
    for i in range(2):
        def F(stage):
            s = u + h*sum(A2[i,j]*k[j] for j in range(i)) + h*A2[i,i]*stage
            return stage - f(t + c2[i]*h, s)

        def J(stage):
            s = u + h*sum(A2[i,j]*k[j] for j in range(i)) + h*A2[i,i]*stage
            return np.eye(3) - h*A2[i,i]*Jf(s)

        ki = newton(F, J, f(t, u))
        k.append(ki)

    return u + h*(b2[0]*k[0] + b2[1]*k[1])

def bdf2(f, Jf, t0, tf, u0, h):
    N_step = int((tf - t0)/h)
    
    u1 = dirk2_step(f, Jf, t0, u0, h)

    sol = [u0.copy(), u1.copy()]
    t = t0 + h

    u_nm1 = u0.copy()
    
    u_n = u1.copy()

    for _ in range(2, N_step):
        def F(u):
            return (3*u - 4*u_n + u_nm1)/(2*h) - f(t+h, u)

        def J(u):
            return 3/(2*h)*np.eye(3) - Jf(u)

        u_np1 = newton(F, J, u_n.copy())

        sol.append(u_np1.copy())
        u_nm1, u_n = u_n, u_np1
        t = t + h

    return np.array(sol)



u0 = np.array([1.0, 0.0, 0.0])
t0, tf = 0, 100
hs = [1e-1, 1e-2, 1e-3]

h_int = 2

time_start2 = time.process_time()
sol = bdf2(robertson, jacobian, t0, tf, u0, hs[h_int])
time_end2 = time.process_time()
time_start1 = time.process_time()
sol_dirk3 = dirk3(robertson, jacobian, t0, tf, u0, hs[h_int])
time_end1 = time.process_time()
sol_dirk2 = [u0]



times = np.arange(0,100, hs[h_int])


time_start = time.process_time()

for i,t in enumerate(times[1:]):
    sol_dirk2.append(dirk2_step(robertson, jacobian, t, sol_dirk2[i], hs[h_int]))

time_end = time.process_time()

time_elapsed = time_end - time_start
time_elapsed1 = time_end1 - time_start1
time_elapsed2 = time_end2 - time_start2
times_array = np.array([time_elapsed, time_elapsed1, time_elapsed2])
sort_array = np.argsort(times_array)
cpu_array = times_array[sort_array]
methods = np.array(["DIRK2", "DIRKo3", "BFD"])
print("dirk2, dirk3, bdf ", time_elapsed, time_elapsed1, time_elapsed2)
sol_dirk2 = np.array(sol_dirk2)


pos = 2



plt.plot(methods[sort_array], cpu_array, marker = ".", color = "black")
plt.grid(True, alpha = 0.3)
plt.xlabel("Methods")
plt.ylabel("Times")
plt.savefig("prob3c_hw3_10.svg")
plt.show()