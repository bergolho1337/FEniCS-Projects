import sys
import numpy as np
import matplotlib.pyplot as plt

def calc_error (argv):
    max_degree = int(argv[1])
    for degree in range(1,max_degree+1):
        filename = "output/error_cg_%d.dat" % degree
        data = np.genfromtxt(filename)
        plt.loglog(data[:,0],data[:,1], marker='o', label='degree=%d' % degree)

	plt.title("Error L2",fontsize=14)
	plt.xlabel(u"log h",fontsize=15)
	plt.ylabel(u"log E",fontsize=15)
    plt.grid()
    plt.legend(loc="best")
    #plt.show()
    plt.savefig("output/error.pdf")

def main():
	if (len(sys.argv) != 2):
		print("==============================================================")
		print("Usage:> python plot_convergeance.py <max_degree>")
		print("\t<max_degree> = Maximum degree of the unitary test")
		print("==============================================================")
		sys.exit(1)

	calc_error(sys.argv)

if __name__ == "__main__":
	main()
