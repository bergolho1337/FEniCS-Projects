import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("output.dat")

log_h = np.log10(data[:,0])
log_E = np.log10(data[:,1])
coeff = np.polyfit(log_h,log_E,1)
a = coeff[0]
b = coeff[1]
print("Error function")
print("y = %.10lf . x + %.10lf" % (a,b))

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




