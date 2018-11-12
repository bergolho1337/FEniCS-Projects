import numpy as np
from pylab import *

# Definindo o conjunto de dados a ser plotado
u = np.linspace(0,1,100)
h = u**2 * (3 - 2*u)

# Plotando os dados
plot(u,h)

# Visualizando
grid()
show()
