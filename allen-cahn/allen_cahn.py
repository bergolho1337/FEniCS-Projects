import random
import sympy
import numpy as np
from dolfin import *

center_1 = np.array([0.3,0.5])
center_2 = np.array([0.7,0.5])
radius = 0.2

w0 = 0.0
w1 = 0.05

# Class representing the initial conditions
class InitialConditions(UserExpression):
    def __init__(self, **kwargs):
        random.seed(2 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)
    def eval(self, values, x):
        dist_1 = np.linalg.norm(x-center_1)
        dist_2 = np.linalg.norm(x-center_2)
        if (dist_1 <= radius):
            values[0] = 1.0
        elif (dist_2 <= radius):
            values[0] = 1.0
        else:
            values[0] = 0.0

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

# Model parameters
lmbda  = 5.0e-02  # surface parameter
dt     = 5.0e-04    # time step
theta  = 0.5        # time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson
M = 1.0

# Form compiler options
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True

# Create mesh and build function space
mesh = UnitSquareMesh.create(100, 100, CellType.Type.quadrilateral)
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
h = u**2 * (3 - 2*u)
f    = 0.25*u**2*(1-u)**2 + w1*h + w0*(1 - h)
dfdu = diff(f, u)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

F = u*v*dx + dt*lmbda*dot(grad(u), grad(v))*dx - u_n*v*dx - dt*dfdu*v*dx
a, L = lhs(F), rhs(F)

# Output file
file = File("vtu/output.pvd", "compressed")

# Time-stepping
u = Function(V)
t = 0.0
T = 200.0*dt
while (t < T):
    print("Time = %.10lf" % t)

    t += dt
    solve(a == L, u)
    
    file << (u,t)

    # Update previous solution
    u_n.assign(u)