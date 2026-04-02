import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from matplotlib.tri import Triangulation

from distmesh import *


Lx = Ly = 3.0
cx = cy = 1.5
r_include = 1.0

def d_square(p):
    return drectangle(p, 0, 3.0, 0, 3.0)

def make_fixed_pts(n_arc=80, n_s=21):

    xs = np.linspace(0, 3.0, n_s)

    ys = np.linspace(0, 3.0, n_s)
    

    left = np.column_stack([np.zeros_like(ys), ys])
    right = np.column_stack([3.0 * np.ones_like(ys), ys])
    bottom = np.column_stack([xs, np.zeros_like(xs)])
    top = np.column_stack([xs, 3.0 * np.ones_like(xs)])

    theta = np.linspace(0.0, 2 * np.pi, n_arc, endpoint=False)
    circle = np.column_stack([cx + r_include*np.cos(theta),
                              cy + r_include*np.sin(theta)])

    pfix = np.vstack([left, right, bottom, top, circle])
    return np.unique(np.round(pfix, 12), axis=0)


def triangle_gradients(coords):

    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]

    detJ = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
    area = (1/2) * abs(detJ)

    b = np.array([y2-y3, y3-y1, y1-y2], dtype=float)
    c = np.array([x3-x2, x1-x3, x2-x1], dtype=float)

    grads = np.column_stack([b, c]) / (2*area)
    return area, grads


def conductivity_pts(p, a1, a2):
    rr = (p[:,0] - cx)**2 + (p[:,1] - cy)**2
    inside = rr <= r_include**2 + 1e-12
    a = np.full(len(p), a2, dtype=float)
    a[inside] = a1
    return a


def assemble_system(pts, tri, a1, a2):
    npts = len(pts)
    A = lil_matrix((npts, npts), dtype=float)
    b = np.zeros(npts, dtype=float)

    centers = pts[tri].mean(axis=1)
    atri = conductivity_pts(centers, a1, a2)

    for k, nodes in enumerate(tri):
        coords = pts[nodes]
        area, grads = triangle_gradients(coords)

        Ke = atri[k] * area * (grads @ grads.T)

        for i_local, i_global in enumerate(nodes):
            for j_local, j_global in enumerate(nodes):
                A[i_global, j_global] += Ke[i_local, j_local]

    eps = 1e-8

    left = np.isclose(pts[:,0], 0.0, atol=eps)
    right = np.isclose(pts[:,0], 3.0, atol=eps)

    dirichlet = np.where(left | right)[0]
    uD = np.zeros(npts)
    uD[right] = 1.0

    free = np.setdiff1d(np.arange(npts), dirichlet)

    for idx in dirichlet:
        col = A[:, idx].toarray().ravel()
        b[free] -= col[free] * uD[idx]

    for idx in dirichlet:
        A.rows[idx] = [idx]
        A.data[idx] = [1.0]
        b[idx] = uD[idx]

    for i in free:
        for idx in dirichlet:
            A[i, idx] = 0.0

    return csr_matrix(A), b, atri

def solve_fem(pts, tri, a1, a2):
    A, b, atri = assemble_system(pts, tri, a1, a2)
    u = spsolve(A, b)
    return u, atri

def current_density(pts, tri, u, atri):
    ntri = len(tri)
    grad_u = np.zeros((ntri, 2))
    j = np.zeros((ntri, 2))
    centers = pts[tri].mean(axis=1)

    for k, nodes in enumerate(tri):
        coords = pts[nodes]
        area, grads = triangle_gradients(coords)
        u_local = u[nodes]
        grad = u_local @ grads
        grad_u[k] = grad
        j[k] = -atri[k] * grad

    abs_jcenters = np.linalg.norm(j, axis=1)

    npts = len(pts)
    abs_j_verts = np.zeros(npts)
    count = np.zeros(npts)

    for k, nodes in enumerate(tri):
        abs_j_verts[nodes] += abs_jcenters[k]
        count[nodes] += 1.0

    abs_j_verts /= np.maximum(count, 1.0)
    return centers, grad_u, j, abs_jcenters, abs_j_verts


def plotting(pts, tri, u, abs_vert, name):
    triang = Triangulation(pts[:,0], pts[:,1], tri)

    fig1, ax1 = plt.subplots(figsize=(8,5))
    c1 = ax1.tripcolor(triang, u, shading='gouraud')
    ax1.triplot(triang, color='k', linewidth=0.15, alpha=0.3)
    ax1.add_patch(plt.Circle((cx, cy), r_include, fill=False, color='red', linewidth=1.8))
    ax1.set_aspect('equal')
    ax1.set_title(f'Voltage: {name}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    fig1.colorbar(c1, ax=ax1, label='u')
    fig1.tight_layout()
    

    fig2, ax2 = plt.subplots(figsize=(8,5))
    c2 = ax2.tripcolor(triang, abs_vert, shading='gouraud')
    ax2.triplot(triang, color='k', linewidth=0.15, alpha=0.3)
    ax2.add_patch(plt.Circle((cx, cy), r_include, fill=False, color='red', linewidth=1.8))
    ax2.set_aspect('equal')
    ax2.set_title(f'abs(j): {name}')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    fig2.colorbar(c2, ax=ax2, label='abs(j)')
    fig2.tight_layout()
    fig1.savefig(f"plot_voltage_{name}.png")
    fig2.savefig(f"plot_current_{name}.png")
    plt.show()



n = 50
h0 = r_include*2*np.pi/n
pfix = make_fixed_pts()
pts, tri = distmesh2D(d_square, huniform, h0, [0.0, 3.0, 0.0, 3.0], pfix)
u_a, a_tri_a = solve_fem(pts, tri, 1.2, 1.0)
_, _, j_a, absj_cent_a, absj_vert_a = current_density(pts, tri, u_a, a_tri_a)
plotting(pts, tri, u_a, absj_vert_a, 'CASE A')
u_b, a_tri_b = solve_fem(pts, tri, 0.8, 1.0)
_, _, j_b, absj_cent_b, absj_vert_b = current_density(pts, tri, u_b, a_tri_b)
plotting(pts, tri, u_b, absj_vert_b, 'CASE B')

