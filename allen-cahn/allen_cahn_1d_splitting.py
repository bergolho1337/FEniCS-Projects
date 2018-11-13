# Looks like its okay !!!

import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dolfin import *

tmax = 2.0         # final time
dt = 0.001         # size of the timestep
lmbda = 0.05          # diffusion constant
tau = 1.0          # parameter tau
nelem = 32         # Number of elements
xmin = -2.0         # Inferior limit of the domain
xmax = 2.0         # Superior limit of the domain
w0 = 0.5           # Parameter
w1 = 0.0           # Parameter
print_rate = 10    # print rate of the solution

# Class representing the initial conditions
# Type 1 = Analitical solution
class InitialConditions_Type1(UserExpression):
    def __init__(self, **kwargs):
        random.seed(2 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)
    def eval(self, values, x):
        values[0] = ( 1.0 - np.tanh(x / (2.0*np.sqrt(2.0*lmbda))) ) / 2.0
    
# Class representing the initial conditions
# Type 2 = Single pulse
class InitialConditions_Type2(UserExpression):
    def __init__(self, **kwargs):
        random.seed(2 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)
    def eval(self, values, x):
        if (x >= 2.0) and (x <= 3.0):
            values[0] = 1.0
        else:
            values[0] = 0.0

def compute_f0 ():
    return 6.0*(w0-w1)

def compute_F (u):
    return u*(1.0-u)*(u-0.5+compute_f0())

def plot_aprox_timestep (x,data_aprox,timestep_plot):
    plt.plot(x,data_aprox[timestep_plot][1:],label="aprox-%d" % timestep_plot,linestyle='--')

def compare_aproximation_timesteps ():
    
    # Get the data
    data_aprox = np.genfromtxt("output/aprox.dat")
    x = np.linspace(xmin,xmax,nelem+1)
    timesteps = [1,100,1000,1999]
    #timesteps = [1,1000,10000,19999]

    for k in range(len(timesteps)):
        plot_aprox_timestep(x,data_aprox,timesteps[k])
    plt.grid()
    plt.xlabel("x",fontsize=15)
    plt.ylabel("u",fontsize=15)
    plt.title("Aproximation Timesteps",fontsize=14)
    plt.legend(loc=0,fontsize=14)
    plt.savefig("output/aprox_timesteps.pdf")

# Create mesh and build function space
mesh = IntervalMesh(nelem,xmin,xmax)
P1 = FiniteElement("Lagrange",mesh.ufl_cell(),1)
ME = FunctionSpace(mesh,P1)
V = FunctionSpace(mesh, 'P', 1)

# Define functions
#u   = Function(ME)  # current solution
u_n = Function(V)  # solution from previous converged step

# Define initial value
u_init = InitialConditions_Type1()
u_n.interpolate(u_init)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0)

# Difussion phase
F1 = u*v*dx + dt*lmbda*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx
a1, L1 = lhs(F1), rhs(F1)

# Time-stepping
u = Function(V)
t = 0
num_timesteps = int(tmax/dt)
h = float((xmax-xmin)/nelem)

file = File("vtu2/output.pvd", "compressed")
aprox_file = open("output/aprox.dat","w")

for i in range(num_timesteps):

    # Update current time
    t = i*dt

    # Write current solution into a file
    if (i % print_rate == 0):
        file << (u_n, t)

    # Compute the diffusion solution
    solve(a1 == L1, u)

    # Compute reaction phase
    for i in range(len(u.vector())):
        u.vector()[i] = u.vector()[i] + dt*compute_F(u.vector()[i])

    # Save current solution to the aproximation file
    vertex_values_u = u.compute_vertex_values(mesh)
    aprox_file.write("%g " % t)
    for i in range(len(vertex_values_u)-1):
        aprox_file.write("%g " % vertex_values_u[i])
    aprox_file.write("%g\n" % vertex_values_u[len(vertex_values_u)-1])

    # Update previous solution
    #u_n.assign(u)

    # Get the diffusion solution
    #vertex_values_u = u.compute_vertex_values(mesh)
    
    # Solve the reaction phase using explicit euler
    #for j in range(len(vertex_values_u)):
    #    x = j*h
    #    F = compute_F(vertex_values_u[j])
    #    vertex_values_u[j] = vertex_values_u[j] + dt*F
    
    # Update previous solution
    u_n.assign(u)

aprox_file.close()
compare_aproximation_timesteps()