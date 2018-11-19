# =============================================================================================
# Program that solves the 2d Allen-Cahn equation using the splitting operator
# =============================================================================================

import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dolfin import *

tmax = 4.0         # final time
dt = 0.001         # size of the timestep
lmbda = 0.0001          # diffusion constant
tau = 1.0          # parameter tau
nelem = 4         # Number of elements
w0 = 0.2           # Parameter
w1 = 0.0           # Parameter
print_rate = 10    # print rate of the solution

# Circles configuration parameters
n_circles_1 = 2
radius_1 = 0.15
init_pos_1 = [0.2,0.5]
d_1 = [1.0,0.0]

# Build circles function
def buildCircles (init_pos,d,n_circles,radius):
    centers = []
    for i in range(n_circles_1):
        x = init_pos[0] + d[0]*2*i*radius
        y = init_pos[1] + d[1]*2*i*radius
        center = [x,y]
        centers.append(center)
    return np.array(centers)

centers_1 = buildCircles(init_pos_1,d_1,n_circles_1,radius_1)

# Class representing the initial conditions
class InitialConditions_Type1(UserExpression):
    def __init__(self, **kwargs):
        random.seed(2 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)
    def eval(self, values, x):
        values[0] = 0.0
        for i in range(n_circles_1):
            dist = np.linalg.norm(x-centers_1[i])
            if (dist <= radius_1):
                values[0] = 1.0

# Class representing the initial conditions
# Custom initial condition
class InitialConditions_Type2(UserExpression):
    def __init__(self, **kwargs):
        random.seed(2 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)
    def eval(self, values, x):
        values[0] = x[0]+x[1]

def compute_f0 ():
    return 6.0*(w0-w1)

def compute_F (u):
    return u*(1.0-u)*(u-0.5+compute_f0())

def get_weight_xy (x,y):
    if (x == 0.0):
        if (y == 0.0 or y == 1.0):
            return 1.0
        else:
            return 2.0
    elif (x == 1.0):
        if (y == 0.0 or y == 1.0):
            return 1.0
        else:
            return 2.0
    else:
        if (y == 0.0 or y == 1.0):
            return 2.0
        else:
            return 4.0

def compute_weights (mesh):
    weights = []
    coordinates = mesh.coordinates()
    for i in range(len(coordinates)):
        x, y = coordinates[i][0], coordinates[i][1]
        weight = get_weight_xy(x,y)
        print("(%lf,%lf) = %lf" % (x,y,weight))
        weights.append(weight)
    return weights

def compute_aproximation ():
    # Create mesh and build function space
    mesh = UnitSquareMesh(nelem,nelem)
    P1 = FiniteElement("Lagrange",mesh.ufl_cell(),1)
    ME = FunctionSpace(mesh,P1*P1)
    V = FunctionSpace(mesh, 'P', 1)

    weights = compute_weights(mesh)

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

    file = File("vtu/output.pvd", "compressed")

    for i in range(num_timesteps):

        # Update current time
        t = i*dt
        print("[Timestep] t = %.10lf" % t)

        # Write current solution into a file
        if (i % print_rate == 0):
            file << (u_n, t)

        # Compute the diffusion solution
        solve(a1 == L1, u)

        # Compute reaction phase
        for i in range(len(u.vector())):
            u.vector()[i] = u.vector()[i] + dt*compute_F(u.vector()[i])

        # Update previous solution
        u_n.assign(u)

def print_boundary ():
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, 'CG', 1)
    bc = DirichletBC(V, Constant(0), 'near(x[0], 0) || near(x[0], 1) || near(x[1], 0) || near(x[1], 1)')
    not_bc = DirichletBC(V, Constant(0), '!(near(x[0], 0) || near(x[0], 1) || near(x[1], 0) || near(x[1], 1))')

    gdim = mesh.geometry().dim()
    d2v = dof_to_vertex_map(V)
    print("Number of points in boundary = %d" % (len(bc.get_boundary_values())))
    print("Number of points not in boundary = %d" % (len(not_bc.get_boundary_values())))
    for dof in bc.get_boundary_values():
        vertex = Vertex(mesh, d2v[dof])

        # Filter only the 4 corners
        if (vertex.x(0) == 0.0 and vertex.x(1) == 0.0):
            in_corner = True
        elif (vertex.x(0) == 1.0 and vertex.x(1) == 0.0):
            in_corner = True
        elif (vertex.x(0) == 0.0 and vertex.x(1) == 1.0):
            in_corner = True
        elif (vertex.x(0) == 1.0 and vertex.x(1) == 1.0):
            in_corner = True
        else:
            in_corner = False

        if (in_corner):    
            print("(%.10lf,%.10lf) - Coeff = 1" % (vertex.x(0),vertex.x(1)))
        else:
            print("(%.10lf,%.10lf) - Coeff = 2" % (vertex.x(0),vertex.x(1)))
        #for i in range(gdim):
        #    print("i = %d -- %.10lf" % (i,vertex.x(i)))


def edit_boundary_points ():
    mesh = UnitSquareMesh(2, 2)

    V = FunctionSpace(mesh, 'CG', 1)
    dof_v = dof_to_vertex_map(V)
    v_dof = vertex_to_dof_map(V)

    f = Expression(('x[0]+x[1]'),degree=1)
    u = interpolate(f,V)
    u_v = u.vector()

    file = File("teste.pvd", "compressed")

    for i in range(len(u_v)):
        print(u_v[i])
    print(dof_v)
    print(v_dof)

    file << (u, 0)

def main ():
    # Supressing outputs from the solver
    set_log_level(50)

    edit_boundary_points()
    #compute_aproximation()

if __name__ == "__main__":
    main()