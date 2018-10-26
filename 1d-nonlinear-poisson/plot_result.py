import numpy as np
from matplotlib import pyplot

analit = np.genfromtxt(open("output/analit.dat","r")) 
aprox = np.genfromtxt(open("output/aprox.dat","r"))

pyplot.clf()
pyplot.title("Analitical x Aproximation")
pyplot.xlabel("x")
pyplot.ylabel("u")
pyplot.plot(analit[:,0],analit[:,1],label="analit",linewidth=2,color="r")
pyplot.plot(aprox[:,0],aprox[:,1],label="aprox",linewidth=2,linestyle="--",color="b",marker='o')
pyplot.grid()
pyplot.legend(loc=0,fontsize=15)
pyplot.show()
#pyplot.savefig(output_filename)
