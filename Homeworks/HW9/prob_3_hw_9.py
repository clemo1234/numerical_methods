import numpy as np
import matplotlib.pyplot as plt

t1 = np.linspace(0, 1, 200)
t2 = np.linspace(1, 1.5, 200)
t3 = np.linspace(1.5, 6, 400)

plt.figure(figsize=(8, 6))

t = np.linspace(0, 6, 300)
for u in np.linspace(0, 2, 20):
    x = u * t
    plt.plot(x, t, color="gray", linewidth=0.8, alpha=0.5)


plt.plot(0*t, t, "k--", linewidth=1.5, label=r"Rarefaction boundary $x=0$")

#x = 2t
plt.plot(2*t, t, "k--", linewidth=1.5, label=r"Rarefaction boundary $x=2t$")

#u = 2 region
for x0 in np.linspace(0.05, 0.95, 8):
    x = x0 + 2*t
    plt.plot(x, t, color="blue", linewidth=0.8, alpha=0.6)


for x0 in np.linspace(1.05, 1.95, 8):
    x = x0 + t
    plt.plot(x, t, color="purple", linewidth=0.8, alpha=0.6)


xs1 = 1 + 1.5*t1
plt.plot(xs1, t1, "r", linewidth=2.5, label=r"Shock $2\to 1$")


xs2 = 2 + 0.5*t1
plt.plot(xs2, t1, "m", linewidth=2.5, label=r"Shock $1\to 0$")
xs_combined = t2 + 1.5
plt.plot(xs_combined, t2, "r", linewidth=2.5, label=r"Combined shock $2\to 0$")

xs_fan = np.sqrt(6*t3)
plt.plot(xs_fan, t3, "r", linewidth=2.5, label=r"Shock in fan")

plt.scatter([2.5, 3], [1, 1.5], color="black", zorder=5, marker='s')
plt.text(2.55, 1.03, r"$(5/2,1)$", fontsize=11,bbox=dict(facecolor='white', alpha=1.0, edgecolor='black'))
plt.text(3.05, 1.53, r"$(3,3/2)$", fontsize=11,bbox=dict(facecolor='white', alpha=1.0, edgecolor='black'))

plt.xlabel(r"$x$")
plt.ylabel(r"$t$")
plt.title("Characteristics and shock curves for non-viscous Burgers equation")
plt.xlim(-0.1, 3.5)
plt.ylim(0, 3)
plt.grid(True)
plt.legend(loc="upper right")
plt.savefig("prob3_burger.pdf")
plt.show()