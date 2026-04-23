import numpy as np
import pygmsh
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


dt = 0.01
Tfinal = 1.0
num_steps = int(Tfinal / dt)
inner_radius = 1
outer_radius = 2
mesh_size = 0.12  

def u_stationary(r):
    return (1.0 - r**2)/4.0 + (3.0*np.log(r))/(4.0*np.log(2.0))


def u0_func(x, y):
    r = np.sqrt(x**2 + y**2)
    return r + x/r

with pygmsh.occ.Geometry() as geom:
    outer = geom.add_disk([0.0, 0.0, 0.0], outer_radius)
    inner = geom.add_disk([0.0, 0.0, 0.0], inner_radius)
    annulus = geom.boolean_difference(outer, inner)
    geom.characteristic_length_min = mesh_size
    geom.characteristic_length_max = mesh_size
    mesh = geom.generate_mesh()

points = mesh.points[:, :2]

if "triangle" in mesh.cells_dict:
    triangles = mesh.cells_dict["triangle"]
else:
    raise ValueError("No triangular cells found in generated mesh.")

n_nodes = len(points)
n_tri = len(triangles)

x = points[:, 0]
y = points[:, 1]
r_all = np.sqrt(x**2 + y**2)

tol = 1e-3
boundary_nodes = np.where(
    (np.abs(r_all - inner_radius) < tol) | (np.abs(r_all - outer_radius) < tol)
)[0]
boundary_nodes = np.unique(boundary_nodes)

all_nodes = np.arange(n_nodes)
interior_nodes = np.setdiff1d(all_nodes, boundary_nodes)

g2i = -np.ones(n_nodes, dtype=int)
g2i[interior_nodes] = np.arange(len(interior_nodes))

n_int = len(interior_nodes)

def triangle_area(coords):
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]
    return 0.5 * abs((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))

def local_matrices(coords):
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]

    area = triangle_area(coords)

    b = np.array([y2 - y3, y3 - y1, y1 - y2], dtype=float)
    c = np.array([x3 - x2, x1 - x3, x2 - x1], dtype=float)

    Ke = (np.outer(b, b) + np.outer(c, c)) / (4.0 * area)

    Me = (area / 12.0) * np.array([
        [2.0, 1.0, 1.0],
        [1.0, 2.0, 1.0],
        [1.0, 1.0, 2.0]
    ])

    fe = (area / 3.0) * np.ones(3)

    return Me, Ke, fe


M = lil_matrix((n_int, n_int))
K = lil_matrix((n_int, n_int))
F = np.zeros(n_int)

for tri in triangles:
    coords = points[tri]
    Me, Ke, fe = local_matrices(coords)
    for a_local in range(3):
        A_global = tri[a_local]
        A_int = g2i[A_global]
        if A_int != -1:
            F[A_int] += fe[a_local]
        for b_local in range(3):
            B_global = tri[b_local]
            B_int = g2i[B_global]
            if A_int != -1 and B_int != -1:
                M[A_int, B_int] += Me[a_local, b_local]
                K[A_int, B_int] += Ke[a_local, b_local]
M = csr_matrix(M)
K = csr_matrix(K)

U0_full = u0_func(points[:, 0], points[:, 1])
U0_full[boundary_nodes] = 0.0
U_n = U0_full[interior_nodes].copy()

U_hist = np.zeros((n_nodes, num_steps + 1))
U_hist[:, 0] = U0_full


A = M + 0.5 * dt * K
B = M - 0.5 * dt * K

for n in range(num_steps):
    rhs = B @ U_n + dt * F
    U_np1 = spsolve(A, rhs)

    U_full = np.zeros(n_nodes)
    U_full[interior_nodes] = U_np1
    U_full[boundary_nodes] = 0.0

    U_hist[:, n+1] = U_full
    U_n = U_np1.copy()


idx_01 = int(0.1/dt)
idx_10 = int(1.0/dt)
U_t01 = U_hist[:, idx_01]
U_t10 = U_hist[:, idx_10]

triang = mtri.Triangulation(points[:, 0], points[:, 1], triangles=triangles)
plt.figure(figsize=(7, 6))
plt.tripcolor(triang, U_t01, shading='gouraud', cmap='plasma')
plt.colorbar(label=r'u(x,y,0.1)')
plt.title("FEM solution t = 0.1")
plt.xlabel(r"x")
plt.ylabel(r"y")
plt.gca().set_aspect('equal')
plt.tight_layout()

plt.figure(figsize=(7, 6))
plt.tripcolor(triang, U_t10, shading='gouraud', cmap='plasma')
plt.colorbar(label=r'u(x,y,1)')
plt.title("FEM solution t = 1")
plt.xlabel(r"x")
plt.ylabel(r"y")
plt.gca().set_aspect('equal')
plt.tight_layout()

r = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
sort_idx = np.argsort(r)

r_sorted = r[sort_idx]
u_sorted = U_t10[sort_idx]

r_exact = np.linspace(1.0, 2.0, 400)
u_exact = u_stationary(r_exact)

plt.figure(figsize=(8, 6))
plt.plot(r_sorted, u_sorted, '.', markersize=4, label='Num. FEM t=1')
plt.plot(r_exact, u_exact, 'k--', linewidth=2, label='Exact stationary soln')
plt.xlabel(r"r")
plt.ylabel(r"u(r)")
plt.title("Solution t=1 vs exact stationary soln")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()