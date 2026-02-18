import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time


mu = 0.012277471

y_init = [0.994, 0, 0, -2.00158510637908252240537862224]

y_1_init, y_2_init, y_3_init, y_4_init = y_init

def arenstorf(t, y):
    y_1, y_2, y_3, y_4 = y
    
    r_1 = ((y_1+mu)**2 + y_2**2)**(1/2)
    r_2 = ((y_1 - 1 + mu)**2 + y_2**2)**(1/2)
    
    return [y_3, y_4, y_1 + 2*y_4 - (1-mu)*(y_1 + mu)/r_1**3 - mu*(y_1 - 1 + mu)/r_2**3, y_2 - 2*y_3 - 
            (1-mu)*y_2/r_1**3 - mu*y_2/r_2**3]

start_cpu_time = time.process_time()

sol = solve_ivp(arenstorf, [0, 100], y_init,
                method= "DOP853", rtol = 1e-12, atol = 1e-12)

end_cpu_time = time.process_time()


y_1, y_2, y_3, y_4 = sol.y

elapsed_cpu_time = end_cpu_time - start_cpu_time

print(f"Elapsed CPU time: {elapsed_cpu_time:.4f} seconds")

plt.scatter(0,0, color = "blue", label = "Earth")
plt.plot(y_1, y_2, color = "black", label = "Trajectory")
plt.grid(True, alpha = 0.4)
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc = "upper right")
plt.savefig("prob1bdop853.svg")
plt.show()


