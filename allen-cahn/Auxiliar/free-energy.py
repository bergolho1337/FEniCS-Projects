import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

w0 = 0
w1 = 0

def h (u):
    return (u**2.0) * (3.0 - 2.0*u)

def W (u):
    return (0.25 * u**2.0 * (1.0 - u)**2.0) + (w1*h(u)) + (w0*(1.0-h(u)))

def plot_solution (x,y):
    plt.clf()
    plt.grid()
    plt.plot(x,y,label="aprox",c="blue",linestyle='--')
    plt.xlabel("x",fontsize=15)
    plt.ylabel("y",fontsize=15)
    plt.title("Free-Energy",fontsize=14)
    plt.legend(loc=0,fontsize=14)
    plt.show()

def main():
    
    x = np.linspace(-0.3,1.3,100)
    y = W(x)
    plot_solution(x,y)
    #print(asol)

if __name__ == '__main__':
    main()