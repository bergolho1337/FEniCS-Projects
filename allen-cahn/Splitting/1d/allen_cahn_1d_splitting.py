# =============================================================================================
# Program that solves the 1-d Allen-Cahn equation using the splitting operator
# =============================================================================================

import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dolfin import *

# -----------------------------------------------------------------------------------------------
# Allen Cahn equation parameters
tmax = 10.0         # final time
dt = 0.001         # size of the timestep
lmbda = 0.001          # diffusion constant
tau = 1.0          # parameter tau
nelem = 128         # Number of elements
xmin = -2.0         # Inferior limit of the domain
xmax = 2.0         # Superior limit of the domain
w0 = 0.0           # Parameter
w1 = 0.0           # Parameter
print_rate = 10    # print rate of the solution
alpha = 5.0        # Volume rate
V = 0.5            # Maximum cell volume
# -----------------------------------------------------------------------------------------------
# Space discretization
h = float((xmax-xmin)/nelem)

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
        if (x >= 0.0) and (x <= 1.0):
            values[0] = 1.0
        else:
            values[0] = 0.0

# h
def compute_h (u):
    return u**2 * (3 - 2*u)

# Compute the numerical integral using the trapeziodal rule
def compute_volume (u):
    n = len(u)
    A = (compute_h(u[0]) + compute_h(u[n-1])) / 2.0
    for i in range(1,n-1):
        A = A + compute_h(u[i])
    return A*h

# f_bar
def compute_fbar (u,vu):
    return alpha * (V - vu)

# f_0
def compute_f0 ():
    return 6.0*(w0-w1)

# F (reaction equation)
def compute_F (u,vu):
    #return u*(1.0-u)*(u-0.5+compute_f0())
    return u*(1.0-u)*(u-0.5+compute_fbar(u,vu))

def plot_aprox_timestep (x,data_aprox,timestep_plot):
    plt.plot(x,data_aprox[timestep_plot][1:],label="aprox-%d" % timestep_plot,linestyle='--')

def compare_aproximation_timesteps ():
    
    # Get the data
    data_aprox = np.genfromtxt("output/aprox.dat")
    x = np.linspace(xmin,xmax,nelem+1)
    total_steps = int(tmax/dt)
    t_steps = total_steps / 6
    timesteps = []
    for i in range(6):
        timesteps.append( int(t_steps*i) )
    

    for k in range(len(timesteps)):
        plot_aprox_timestep(x,data_aprox,timesteps[k])
    plt.grid()
    plt.xlabel("x",fontsize=15)
    plt.ylabel("u",fontsize=15)
    plt.title("Aproximation Timesteps",fontsize=14)
    plt.legend(loc=0,fontsize=14)
    plt.savefig("output/aprox_timesteps.pdf")

def compute_aproximation ():
    # Create mesh and build function space
    mesh = IntervalMesh(nelem,xmin,xmax)
    P1 = FiniteElement("Lagrange",mesh.ufl_cell(),1)
    ME = FunctionSpace(mesh,P1)
    V = FunctionSpace(mesh, 'P', 1)

    # Define functions
    #u   = Function(ME)  # current solution
    u_n = Function(V)  # solution from previous converged step

    # Define initial value
    u_init = InitialConditions_Type2()
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
    #h = float((xmax-xmin)/nelem)

    file = File("vtu/output.pvd", "compressed")
    aprox_file = open("output/aprox.dat","w")

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
        # Calculate the current volume of the cell
        vu = compute_volume(u.vector())
        print("Current volume = %.10lf" % vu)

        for i in range(len(u.vector())):
            u.vector()[i] = u.vector()[i] + dt*compute_F(u.vector()[i],vu)

        # Save current solution to the aproximation file
        vertex_values_u = u.compute_vertex_values(mesh)
        aprox_file.write("%g " % t)
        for i in range(len(vertex_values_u)-1):
            aprox_file.write("%g " % vertex_values_u[i])
        aprox_file.write("%g\n" % vertex_values_u[len(vertex_values_u)-1])

        # Update previous solution
        u_n.assign(u)

    aprox_file.close()

def main ():
    # Supressing outputs from the solver
    set_log_level(50)

    # Compute aproximation
    compute_aproximation()
    compare_aproximation_timesteps()

if __name__ == "__main__":
    main()