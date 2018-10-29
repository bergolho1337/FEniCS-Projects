import sys
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def main():
    timestep_plot = 300
    xmin = -2
    xmax = 2
    nelem = 100

    data_analit = np.genfromtxt("output/analit.dat")
    data_aprox = np.genfromtxt("output/aprox.dat")
    x = np.linspace(xmin,xmax,nelem+1)

    plt.clf()
    plt.grid()
    plt.plot(x,data_analit[timestep_plot][1:],label="analit",c="red")
    plt.plot(x,data_aprox[timestep_plot][1:],label="aprox",c="blue",linestyle='--')
    plt.xlabel("x",fontsize=15)
    plt.ylabel("u",fontsize=15)
    plt.title("Analitical x Aproximation",fontsize=14)
    plt.legend(loc=0,fontsize=14)
    plt.show()

	

if __name__ == "__main__":
    main()
