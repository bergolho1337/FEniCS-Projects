import sys
import numpy as np
import matplotlib.pyplot as plt

def calc_error (argv):
    	filename = sys.argv[1]
    
	data = np.genfromtxt(filename)

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

def main():
	if (len(sys.argv) != 2):
		print("==============================================================")
		print("Usage:> python plot_error.py <error_filename>")
		print("\t<error_filename> = Path to the error filename")
		print("==============================================================")
		sys.exit(1)

	calc_error(sys.argv)

if __name__ == "__main__":
	main()
