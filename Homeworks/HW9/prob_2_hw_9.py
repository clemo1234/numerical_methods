import numpy as np
import matplotlib.pyplot as plt

def f(rho):
    return -rho*np.log(rho)

def fp(rho):
    return -(np.log(rho) + 1)

rho_L = 0.1
rho_R = 0.9

c_L = fp(rho_L)
c_R = fp(rho_R)

s = (f(rho_R) - f(rho_L))/(rho_R - rho_L)

t = np.linspace(0, 3, 300)

x0_left = np.linspace(-3, -0.1, 12)
x0_right = np.linspace(0.1, 3, 12)

plt.figure(figsize=(8, 6))

for x0 in x0_left:
    x = x0 + c_L*t
    plt.plot(x, t, color="blue", alpha=0.8)

for x0 in x0_right:
    x = x0 + c_R*t
    plt.plot(x, t, color="red", alpha=0.8)

x_shock = s*t
plt.plot(x_shock, t, color="darkgreen", linewidth=3, label="shock line")

plt.xlabel(r"$x$")
plt.ylabel(r"$t$")
plt.title("Characteristics and Shock Line for Traffic Model")
plt.legend()
plt.grid(True)
plt.savefig("prob2_green_b.pdf")
plt.show()
print("Left characteristic speed =", c_L)
print("Right characteristic speed =", c_R)
print("Shock speed =", s)