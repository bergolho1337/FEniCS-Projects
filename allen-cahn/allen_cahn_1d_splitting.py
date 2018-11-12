import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dolfin import *

tmax = 2.0         # final time
dt = 0.001         # size of the timestep
D0 = 0.05          # diffusion constant
tau = 1.0          # parameter tau
nelem = 32         # Number of elements
xmin = 0.0         # Inferior limit of the domain
xmax = 4.0         # Superior limit of the domain
w0 = 0.5           # Parameter
w1 = 0.0           # Parameter
print_rate = 10    # print rate of the solution

# Class representing the initial conditions
# Type 1 = Single pulse
class InitialConditions_Type1(UserExpression):
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

file = File("vtu/output.pvd", "compressed")
 
# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0)

# Difussion phase
F1 = u*v*dx + dt*D0*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx
a1, L1 = lhs(F1), rhs(F1)

# Time-stepping
u = Function(V)
t = 0
num_timesteps = int(tmax/dt)
h = float((xmax-xmin)/nelem)

file = File("vtu/output.pvd", "compressed")
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