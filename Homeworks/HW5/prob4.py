import numpy as np
import matplotlib.pyplot as plt
import pygmsh
from distmesh import dcircle, drectangle, dline, ddiff, dunion, dintersect


def regular_polygon(n, rad=1.0, center=(0.0,0.0), angle0=0.0):
    cx, cy = center
    angle = angle0 + (2*np.pi*np.arange(n))/n
    return np.column_stack([cx + rad*np.cos(angle), cy + rad*np.sin(angle)])

def add_polygon_loop(geom, verts, mesh_size):
    pts = [geom.add_point([x, y, 0.0], mesh_size) for x, y in verts]
    lines = [geom.add_line(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
    return geom.add_curve_loop(lines)

def add_circle_loop(geom, center, r, mesh_size):
    cx, cy = center
    pc = geom.add_point([cx, cy, 0.0], mesh_size)
    pR = geom.add_point([cx + r, cy, 0.0], mesh_size)
    pT = geom.add_point([cx, cy + r, 0.0], mesh_size)
    pL = geom.add_point([cx - r, cy, 0.0], mesh_size)
    pB = geom.add_point([cx, cy - r, 0.0], mesh_size)

    a1 = geom.add_circle_arc(pR, pc, pT)
    a2 = geom.add_circle_arc(pT, pc, pL)
    a3 = geom.add_circle_arc(pL, pc, pB)
    a4 = geom.add_circle_arc(pB, pc, pR)

    return geom.add_curve_loop([a1, a2, a3, a4])

def get_triangles(mesh):
    if "triangle" in mesh.cells_dict:
        return mesh.cells_dict["triangle"]


def fd_Ls(p):
    d1 = drectangle(p, -1.0, -0.2, -1.0, 1.0)
    d2 = drectangle(p, -1.0,  1.0, -1.0, -0.2)
    return dunion(d1, d2)

with pygmsh.geo.Geometry() as geom:
    mesh_size = 0.10
    L_verts = np.array([[-1.0, -1.0],
        [ 1.0, -1.0],
        [ 1.0, -0.2],
        [-0.2, -0.2],
        [-0.2,  1.0],
        [-1.0,  1.0],])

    outer_loop = add_polygon_loop(geom, L_verts, mesh_size)
    geom.add_plane_surface(outer_loop)

    meshL = geom.generate_mesh(dim=2)


outer = regular_polygon(5, rad=1.0, angle0=np.pi/2)
inner = regular_polygon(5, rad=0.5, angle0=np.pi/2 + np.pi/5)

def dpoly_from_lines(p, verts):
    d = -np.inf * np.ones(len(p))
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        d = np.maximum(d, dline(p, x1, y1, x2, y2))
    return d

def fd_pentagon_ring(p):
    d_outer = dpoly_from_lines(p, outer)
    d_inner = dpoly_from_lines(p, inner)
    return ddiff(d_outer, d_inner)

with pygmsh.geo.Geometry() as geom:
    mesh_size = 0.10

    outer_loop = add_polygon_loop(geom, outer, mesh_size)
    inner_loop = add_polygon_loop(geom, inner, mesh_size)

    geom.add_plane_surface(outer_loop, holes=[inner_loop])

    meshP = geom.generate_mesh(dim=2)



#params
R = 1
c1 = (-0.5, -0.5)
c2 = ( 0.5, -0.5)
rh = 0.2

def fd_semidisk_twoholes(p):
    d_disk = dcircle(p, 0.0, 0.0, R)

    d_half = dline(p, R, 0.0, -R, 0.0)
    d_outer = dintersect(d_disk, d_half)
    d_h_1 = dcircle(p, c1[0], c1[1], rh)
    d_h_2 = dcircle(p, c2[0], c2[1], rh)

    return ddiff(ddiff(d_outer, d_h_1), d_h_2)

with pygmsh.geo.Geometry() as geom:
    mesh_size = 0.08

    p_left   = geom.add_point([-R,  0.0, 0.0], mesh_size)
    p_right  = geom.add_point([ R,  0.0, 0.0], mesh_size)
    p_bottom = geom.add_point([ 0.0, -R, 0.0], mesh_size)
    p_center = geom.add_point([ 0.0,  0.0, 0.0], mesh_size)

    top_line = geom.add_line(p_left, p_right)
    arc1 = geom.add_circle_arc(p_right, p_center, p_bottom)
    arc2 = geom.add_circle_arc(p_bottom, p_center, p_left)

    outer_loop = geom.add_curve_loop([top_line, arc1, arc2])

    hole1_loop = add_circle_loop(geom, c1, rh, mesh_size)
    hole2_loop = add_circle_loop(geom, c2, rh, mesh_size)
    geom.add_plane_surface(outer_loop, holes=[hole1_loop, hole2_loop])

    meshS = geom.generate_mesh(2)

fig, ax = plt.subplots(1, 3,figsize=(15, 5))


def plot_mesh(ax, mesh, title, color):
    pts = mesh.points[:, :2]
    tri = get_triangles(mesh)
    ax.triplot(pts[:, 0], pts[:, 1], tri, linewidth=0.5, color = color)
    ax.set_aspect("equal")
    ax.set_title(title)

plot_mesh(ax[0], meshL, "L", color = "darkblue")
plot_mesh(ax[1], meshP, "Pentagon", color = "red")
plot_mesh(ax[2], meshS, "Half Circle", color = "Green")

plt.tight_layout()
plt.savefig("pygmsh_triang.svg")
plt.show()