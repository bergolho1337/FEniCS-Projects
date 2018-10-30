import sys
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

D0 = 0.05
TAU = 1.0
w0 = 0.05
w1 = 0.0 

def print_message (ndiv,h):
	print("*********************************************************************")
	print("Solving Allen-Cahn equation using %d divisions -- h = %g" % (ndiv,h))
	print("*********************************************************************")

def f_bar ():
    return 6.0*(w0 - w1)

def V ():
    return math.sqrt(2.0*D0)*f_bar() / TAU

def solve_problem (xmax,tmax):
    x = np.linspace(-2,2,50)
    y = np.linspace(0,10,20)
    
    X, Y = np.meshgrid(x,y)
    Z = (1.0 - np.tanh( (X- V()*Y)/(2.0 * np.sqrt(2.0*D0)) )) / 2.0
    #Z = X**2 + Y**2

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_surface(X,Y,Z,color='b')
    plt.show()
    

def main():

	if (len(sys.argv) != 3):
		print("========================================================")
		print("Usage:> python analitical.py <xmax> <tmax>")
		print("========================================================")
		sys.exit(1)

	xmax = float(sys.argv[1])
	tmax = float(sys.argv[2])

	solve_problem(xmax,tmax)
	

if __name__ == "__main__":
    main()
