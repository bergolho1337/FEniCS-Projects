# Simple program to change the value of point in the solution vector

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
w0 = 0.0           # Parameter
w1 = 0.0           # Parameter
print_rate = 10    # print rate of the solution

# Create mesh and build function space
mesh = IntervalMesh(nelem,xmin,xmax)
P1 = FiniteElement("Lagrange",mesh.ufl_cell(),1)
ME = FunctionSpace(mesh,P1)
V = FunctionSpace(mesh, 'P', 1)

# Define initial value
u_D = Expression('x[0]*x[0]',degree=2)
u_n = interpolate(u_D, V)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0)

F = u*v*dx + dt*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx
a, L = lhs(F), rhs(F)

u = Function(V)
t = 0
file = File("vtu/output.pvd", "compressed")

# Compute solution
solve(a == L, u)

print("Before")
for i in range(len(u.vector())):
    print(u.vector()[i])
    u.vector()[i] = 1.0

print("After")
for i in range(len(u.vector())):
    print(u.vector()[i])

