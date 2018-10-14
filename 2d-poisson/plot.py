import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("error.dat")
plt.grid()
plt.loglog(data[:,0],data[:,1], marker='o')
plt.show()


