# ============================================================================================================
# Program that solves the 2-D Allen-Cahn equation using the FEniCS library.
# ------------------------------------------------------------------------------------------------------------
# This program uses the NewtonMethod to solve the nonlinear system of equations that appear when we apply 
# a variable substitution over the term related to the Laplacian operator. 
# (For more information about the solution steps read the 'solution_guide.pdf' file).
# ------------------------------------------------------------------------------------------------------------
# Remember to clean up the files from the 'vtu' folder when you are going to do another simulation. Use the
# script 'clean_previous_results.sh' to do this.
# ============================================================================================================
 
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dolfin import *

# Model parameters
lmbda  = 5.0e-02    # Surface parameter
dt     = 1.0e-03    # Time step
tmax = 1.0          # Maximum time of the simulation
theta  = 0.5        # Time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson
M = 1.0             # Diffusive factor
nelem = 128         # Number of finite elements to use
w0 = 0.0            # Weight related to the free-energy density
w1 = 0.0            # Weight related to the free-energy density
print_rate = 10     # Rate which the VTU file will be saved

# Build circles function
def buildCircles (init_pos,d,n_circles,radius):
    centers = []
    for i in range(n_circles):
        x = init_pos[0] + d[0]*2*i*radius
        y = init_pos[1] + d[1]*2*i*radius
        center = [x,y]
        centers.append(center)
    return np.array(centers)

# Circles configuration parameters
n_circles_1 = 2
radius_1 = 0.2
init_pos_1 = [0.3,0.5]
d_1 = [1.0,0.0]
centers_1 = buildCircles(init_pos_1,d_1,n_circles_1,radius_1)

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
        values[1] = 0.0
    def value_shape(self):
        return (2,)

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

def compute_aproximation ():
    print("[!] Computing approximate solution ...")

    # Form compiler options
    parameters["form_compiler"]["optimize"]     = True
    parameters["form_compiler"]["cpp_optimize"] = True

    # Create mesh and build function space
    mesh = UnitSquareMesh(nelem,nelem)
    P1 = FiniteElement("Lagrange",mesh.ufl_cell(),1)
    ME = FunctionSpace(mesh,P1*P1)

    # Define trial and test functions
    du    = TrialFunction(ME)
    q, v  = TestFunctions(ME)

    # Define functions
    u   = Function(ME)  # current solution
    u0  = Function(ME)  # solution from previous converged step

    # Split mixed functions
    dc, dmu = split(du)
    c,  mu  = split(u)
    c0, mu0 = split(u0)

    # Create intial conditions and interpolate
    u_init = InitialConditions()
    u.interpolate(u_init)
    u0.interpolate(u_init)

    # Compute the chemical potential df/dc
    c = variable(c)
    h = (c**2) * (3 - 2*c)
    f    = (0.25*c**2*(1-c)**2) + (w1*h) + (w0*(1 - h))
    dfdc = diff(f, c)

    # mu_(n+theta)
    mu_mid = (1.0-theta)*mu0 + theta*mu

    # Weak statement of the equations
    L0 = c*q*dx - c0*q*dx - dt*mu_mid*q*dx
    L1 = mu*v*dx + dfdc*v*dx + lmbda*dot(grad(c),grad(v))*dx
    L = L0 + L1

    # Compute directional derivative about u in the direction of du (Jacobian)
    a = derivative(L, u, du)

    # Create nonlinear problem and Newton solver
    problem = AllenCahnEquation(a,L)
    solver = NewtonSolver()
    solver.parameters["linear_solver"] = "lu"
    solver.parameters["convergence_criterion"] = "incremental"
    solver.parameters["relative_tolerance"] = 1e-6

    # Output file
    file = File("vtu/output-2d.pvd", "compressed")

    # Step in time
    k = 0
    n_timesteps = int(tmax/dt)
    while (k <= n_timesteps):
        t = dt*k
        print("Timestep: t = %g" % t)

        # Get the reference of the variable 'u' from the PDE system
        u0.vector()[:] = u.vector()
        solver.solve(problem, u.vector())

        # Save VTU when we reach the a multiple of the 'print_rate'
        if (k % print_rate == 0):
            file << (u.split()[0], t)

        # Next timestep
        k = k + 1

def main ():
    # Supressing outputs from the solver
    set_log_level(50)

    compute_aproximation()
    
if __name__ == "__main__":
    main()