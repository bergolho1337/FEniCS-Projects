import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("output.dat")

plt.title("Error L2",fontsize=14)
plt.xlabel(u"log h",fontsize=15)
plt.ylabel(u"log E",fontsize=15)
plt.loglog(data[:,0],data[:,1], marker='o')
plt.grid()
plt.show()

plt.clf()
plt.title("Error L2",fontsize=14)
plt.xlabel(u"h",fontsize=15)
plt.ylabel(u"E",fontsize=15)
plt.plot(data[:,0],data[:,1], marker='o')
plt.grid()
plt.show()

