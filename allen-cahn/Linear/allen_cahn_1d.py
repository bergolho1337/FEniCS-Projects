import random
import sympy
import numpy as np
from dolfin import *

# Model parameters
lmbda  = 5.0e-02    # surface parameter
dt     = 5.0e-03    # time step
theta  = 0.5        # time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson
M = 1.0
xmin = -2.0
xmax = 2.0
nelem = 100
w0 = 0.5
w1 = 0.0

# Class representing the initial conditions
class InitialConditions(UserExpression):
    def __init__(self, **kwargs):
        random.seed(2 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)
    def eval(self, values, x):
        values[0] = ( 1.0 - np.tanh(x / (2.0*np.sqrt(2.0*lmbda))) ) / 2.0

# Class for interfacing with the Newton solver
class AllenCahnEquation(NonlinearProblem):
    def __init__(self, a, L):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
    def F(self, b, x):
        assemble(self.L, tensor=b)
    def J(self, A, x):
        assemble(self.a, tensor=A)

# Form compiler options
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True

# Create mesh and build function space
mesh = IntervalMesh(nelem,xmin,xmax)
V = FunctionSpace(mesh, 'P', 1)

# Create initial condition
u_init = InitialConditions(degree=1)
#u_n = interpolate(u_init,V)
#u_init = Expression('x[0]*x[0] + x[1]*x[1]',degree=2)

# Define initial value
u_n = interpolate(u_init,V)

# Define the free-energy function
u = Function(V)
u = variable(u)
h = (u**2) * (3 - 2*u)
f    = (0.25*u**2*(1-u)**2) + (w1*h) + (w0*(1 - h))
dfdu = diff(f, u)
#dfdu = u*(1-u)*(u - 0.5 + 6.0*(w0 - w1))

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

F = u*v*dx + dt*lmbda*dot(grad(u), grad(v))*dx - (u_n + dt*dfdu)*v*dx
a, L = lhs(F), rhs(F)

# Output file
file = File("vtu/output.pvd", "compressed")
aprox_file = open("output/aprox.dat","w")

# Time-stepping
u = Function(V)
t = 0.0
T = 400.0*dt
while (t < T):
    print("Time = %.10lf" % t)

    t += dt
    solve(a == L, u)
    
    file << (u,t)
    vertex_values_u = u.compute_vertex_values(mesh)
    aprox_file.write("%g " % t)
    for i in range(len(vertex_values_u)-1):
        aprox_file.write("%g " % vertex_values_u[i])
    aprox_file.write("%g\n" % vertex_values_u[len(vertex_values_u)-1])

    # Update previous solution
    u_n.assign(u)
aprox_file.close()

analit_file = open("output/analit.dat","w")
x = np.linspace(xmin,xmax,nelem+1)
t = 0.0
while (t < T):

    t += dt

    analit_file.write("%g " % t)
    for i in range(len(x)-1):
        f_bar = 6.0*(w0 - w1)
        v = np.sqrt(2.0*lmbda)*f_bar
        y = (1.0 - np.tanh( (x[i] - v*t)/(2.0 * np.sqrt(2.0*lmbda)) )) / 2.0
        analit_file.write("%g " % y)
    f_bar = 6.0*(w0 - w1)
    v = np.sqrt(2.0*lmbda)*f_bar
    y = (1.0 - np.tanh( (x[len(x)-1] - v*t)/(2.0 * np.sqrt(2.0*lmbda)) )) / 2.0
    analit_file.write("%g\n" % y)

    # Update previous solution
    u_n.assign(u)
analit_file.close()

