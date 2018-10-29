import random
import sympy
import numpy as np
from dolfin import *

def buildCircles (init_pos,d,n_circles,radius):
    centers = []
    for i in range(n_circles):
        x = init_pos[0] + d[0]*2*i*radius
        y = init_pos[1] + d[1]*2*i*radius
        center = [x,y]
        centers.append(center)
    return np.array(centers)

n_circles_1 = 5
radius_1 = 0.1
init_pos_1 = [0.1,0.5]
d_1 = [1.0,0.0]
centers_1 = buildCircles(init_pos_1,d_1,n_circles_1,radius_1)

n_circles_2 = 5
radius_2 = 0.1
init_pos_2 = [0.5,0.1]
d_2 = [0.0,1.0]
centers_2 = buildCircles(init_pos_2,d_2,n_circles_2,radius_2)

n_circles_3 = 0
radius_3 = 0.08
init_pos_3 = [0.3,0.4]
d_3 = [0.85090352453,0.52532198881]
centers_3 = buildCircles(init_pos_3,d_3,n_circles_3,radius_3)

n_circles_4 = 0
radius_4 = 0.08
init_pos_4 = [0.7,0.4]
d_4 = [0.0883686861,-0.99608783514]
centers_4 = buildCircles(init_pos_4,d_4,n_circles_4,radius_4)


w0 = 0.0
w1 = 0.05

# Class representing the initial conditions
class InitialConditions(UserExpression):
    def __init__(self, **kwargs):
        random.seed(2 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)
    def eval(self, values, x):
        values[0] = 0.0
        for i in range(n_circles_1):
            dist = np.linalg.norm(x-centers_1[i])
            if (dist <= radius_1):
                values[0] = 1.0
        for i in range(n_circles_2):
            dist = np.linalg.norm(x-centers_2[i])
            if (dist <= radius_2):
                values[0] = 1.0
        for i in range(n_circles_3):
            dist = np.linalg.norm(x-centers_3[i])
            if (dist <= radius_3):
                values[0] = 1.0
        for i in range(n_circles_4):
            dist = np.linalg.norm(x-centers_4[i])
            if (dist <= radius_4):
                values[0] = 1.0

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
T = 400.0*dt
while (t < T):
    print("Time = %.10lf" % t)

    t += dt
    solve(a == L, u)
    
    file << (u,t)

    # Update previous solution
    u_n.assign(u)