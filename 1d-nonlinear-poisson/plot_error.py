import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("output/error.dat")

log_h = np.log10(data[:,0])
log_E = np.log10(data[:,1])
coeff = np.polyfit(log_h,log_E,1)
a = coeff[0]
b = coeff[1]
print("Error function")
print("y = %g . x + %g" % (a,b))

plt.title("Error L2",fontsize=14)
plt.xlabel(u"log h",fontsize=15)
plt.ylabel(u"log E",fontsize=15)
plt.loglog(data[:,0],data[:,1], marker='o')
plt.grid()
plt.show()