import numpy as np
import matplotlib.pyplot as plt
from distmesh import (
    distmesh2D, dcircle, drectangle, dline,
    ddiff, dunion, dintersect, huniform
)

def regular_polygon(n, rad=1.0, center=(0.0,0.0), angle0=0.0):
    cx, cy = center
    angle = angle0 + (2*np.pi*np.arange(n))/n
    return np.column_stack([cx + rad*np.cos(angle), cy + rad*np.sin(angle)])

def dpoly_convex(p, verts):
    d = -np.inf * np.ones(len(p))
    m = len(verts)
    for i in range(m):
        x1, y1 = verts[i]
        x2, y2 = verts[(i+1) % m]
        di = dline(p, x1, y1, x2, y2)
        d = np.maximum(d, di)
    return d

def fh(p):
    return huniform(p)

def fd_Ls(p):
    d1 = drectangle(p, -1.0, -0.2, -1.0, 1.0)
    d2 = drectangle(p, -1.0,  1.0, -1.0, -0.2)
    return dunion(d1, d2)

pfix_L = np.array([[-1.0, -1.0],
    [ 1.0, -1.0],
    [ 1.0, -0.2],
    [-0.2, -0.2],
    [-0.2,  1.0],
    [-1.0,  1.0]])

ptsL, triL = distmesh2D(fd_Ls, fh, 0.12, [-1.2, 1.2, -1.2, 1.2], pfix_L)

outer_part = regular_polygon(5, rad=1, angle0=np.pi/2)
inner_part = regular_polygon(5, rad=0.4, angle0=np.pi/2 + np.pi/5)

def fd_pentagon_ring(p):
    return ddiff(dpoly_convex(p, outer_part), dpoly_convex(p, inner_part))

pfix_P = np.vstack([outer_part, inner_part])
ptsP, triP = distmesh2D(fd_pentagon_ring, fh, 0.1, [-1.2, 1.2, -1.2, 1.2], pfix_P)

def fd_semidisk_twoholes(p):
    d_disk = dcircle(p, 0, 0, 1)
    d_half = dline(p, 1, 0.0, -1, 0.0)
    d_outer = dintersect(d_disk, d_half)
    d_h1 = dcircle(p, -0.5, -0.5, 0.2)
    d_h2 = dcircle(p,  0.5, -0.5, 0.2)
    return ddiff(ddiff(d_outer, d_h1), d_h2)

pfix_S = np.array([
    [-1, 0], [1, 0], [0, -1],
    [-0.2, -0.5], [-0.7, -0.5], [-0.5, -0.2], [-0.5, -0.7],
    [ 0.7, -0.5], [ 0.2, -0.5], [ 0.5, -0.2], [ 0.5, -0.7],
])

ptsS, triS = distmesh2D(fd_semidisk_twoholes, fh, 0.10, [-1.3, 1.3, -1.3, 0.2], pfix_S)
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].triplot(ptsL[:,0], ptsL[:,1], triL, color='black', linewidth=0.5)
ax[0].set_aspect('equal')
ax[0].set_title("L")

ax[1].triplot(ptsP[:,0], ptsP[:,1], triP, color='black', linewidth=0.5)
ax[1].set_aspect('equal')
ax[1].set_title("Pentagon")

ax[2].triplot(ptsS[:,0], ptsS[:,1], triS, color='black', linewidth=0.5)
ax[2].set_aspect('equal')
ax[2].set_title("Half Circle")

plt.tight_layout()
plt.savefig("triangulate.svg")
plt.show()