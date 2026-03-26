import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sparse_solve
import matplotlib.pyplot as plt


def f(x):
    f = np.zeros_like(x)
    region = (x >= -np.pi/2) & (x <= np.pi/2)
    f[region] = -np.cos(x[region])
    return f

def solve_tin_temperature(N=160, L=80):
    hx = 2*np.pi/N
    hy = 2/L
    x = -np.pi + hx*np.arange(N)
    y = np.linspace(0, 2, L + 1)

    e = np.ones(N)
    T = sp.diags([e, -2*e, e], offsets=[-1, 0, 1], shape=(N, N), format='lil')
    T[0, -1] = 1.0
    T[-1, 0] = 1.0

    T = T.tocsr()

    I = sp.eye(N, format='csr')

    Ax = T / hx**2
    B = Ax - (2.0 / hy**2) * I
    C = (1.0 / hy**2) * I

    block = [[None for _ in range(L)] for _ in range(L)]
    for j in range(L - 1):
        block[j][j] = B
        if j > 0:
            block[j][j - 1] = C
        block[j][j + 1] = C

    block[L - 1][L - 3] = (1 / ((2 * hy)) *I)
    block[L - 1][L - 2] = (-2 / (hy)) *I
    block[L - 1][L - 1] = (3 / (2 * hy)) *I

    A = sp.bmat(block, format='csr')

    F = f(x)
    rhs = np.concatenate([np.tile(F, L - 1), np.zeros(N)])

    Uvec = sparse_solve.spsolve(A, rhs)
    U_unknown = Uvec.reshape(L, N)   
    U = np.zeros((L + 1,N))
    U[1:, :] = U_unknown
    print(U)
    return x, y, U

x, y, U = solve_tin_temperature(N=160,L=80)
X, Y = np.meshgrid(x,y)

plt.figure(figsize=(8, 4.8))
cs = plt.contourf(X, Y, U, levels=40)
plt.colorbar(cs, label='temperature u(x,y)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of stationary temperature on cylinder')
plt.tight_layout()
plt.savefig("temp.svg")
plt.show()