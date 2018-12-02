# ============================================================================================================
# Program that solves the 1-D Cahn-Hilliard equation using the FEniCS library.
# ------------------------------------------------------------------------------------------------------------
# All the output files for pos-processing (error analysis, etc ...) are being store over the 'output' folder.
# ------------------------------------------------------------------------------------------------------------
# This program uses the NewtonMethod to solve the nonlinear system of equations that appear when we apply 
# a variable substitution over the term related to the Laplacian operator. 
# (For more information about the solution steps read the 'solution_guide.pdf' file).
# ------------------------------------------------------------------------------------------------------------
# Furthermore, the program also generates a file 'aprox.dat' that represents the 
# approximate solution.
# ============================================================================================================
 
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dolfin import *

# ==========================================================================================================
# Model parameters
lmbda  = 5.0e-04    # Surface parameter
dt     = 1.0e-02    # Time step
tmax = 10.0         # Maximum time of the simulation
theta  = 0.5        # Time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson
M = 1.0             # Diffusive factor
xmin = -2.0         # Limits of the interval
xmax = 2.0          # Limits of the interval
nelem = 128          # Number of finite elements to use
w0 = 0.0            # Weight related to the free-energy density
w1 = 0.0            # Weight related to the free-energy density
timestep_plot = 0   # Timestep of the plot
print_rate = 10     # Rate which the VTU file will be saved
# ==========================================================================================================

# Class representing the initial conditions
# Type 1 = Middle pulse
class InitialConditions_Type1(UserExpression):
    def __init__(self, **kwargs):
        random.seed(2 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)
    def eval(self, values, x):
        if (x >= -0.5 and x <= 0.5):
            values[0] = 1.0
        else:
            values[0] = 0.0
        values[1] = 0.0
    def value_shape(self):
        return (2,)

# Class for interfacing with the Newton solver
class CahnHilliardEquation(NonlinearProblem):
    def __init__(self, a, L):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
    def F(self, b, x):
        assemble(self.L, tensor=b)
    def J(self, A, x):
        assemble(self.a, tensor=A)

def solve_problem ():
    print("[!] Solving 1d Cahn Hilliard equation ...")

    # Form compiler options
    parameters["form_compiler"]["optimize"]     = True
    parameters["form_compiler"]["cpp_optimize"] = True

    # Create mesh and build function space
    mesh = IntervalMesh(nelem,xmin,xmax)
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
    u_init = InitialConditions_Type1()
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
    L0 = c*q*dx - c0*q*dx + dt*dot(grad(mu_mid), grad(q))*dx
    L1 = mu*v*dx - dfdc*v*dx - lmbda*dot(grad(c),grad(v))*dx
    L = L0 + L1

    # Compute directional derivative about u in the direction of du (Jacobian)
    a = derivative(L, u, du)

    # Create nonlinear problem and Newton solver
    problem = CahnHilliardEquation(a,L)
    solver = NewtonSolver()
    solver.parameters["linear_solver"] = "lu"
    solver.parameters["convergence_criterion"] = "incremental"
    solver.parameters["relative_tolerance"] = 1e-6

    # Output file
    file = File("vtu/output-CH.pvd", "compressed")
    aprox_file = open("output/CH.dat","w")

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

        # Save current solution to the aproximation file
        vertex_values_u = u.split()[0].compute_vertex_values(mesh)
        aprox_file.write("%g " % t)
        for i in range(len(vertex_values_u)-1):
            aprox_file.write("%g " % vertex_values_u[i])
        aprox_file.write("%g\n" % vertex_values_u[len(vertex_values_u)-1])

        # Next timestep
        k = k + 1
    aprox_file.close()

def plot_solution ():

    # Get the data
    data_aprox = np.genfromtxt("output/CH.dat")
    x = np.linspace(xmin,xmax,nelem+1)
    timesteps = [0,250,500,750,1000]

    # Plot the solution for each timestep
    for k in range(len(timesteps)):
        timestep = timesteps[k]
        plt.plot(x,data_aprox[timestep][1:],label="t = %.3lf" % float(timestep*dt),linestyle='--')
    
    plt.grid()
    plt.xlabel("x",fontsize=15)
    plt.ylabel("u",fontsize=15)
    plt.title("Aproximation Timesteps (D0 = %.5lf)" % (lmbda),fontsize=14)
    plt.legend(loc=0,fontsize=14)
    plt.savefig("output/aprox_timesteps.pdf")

def main ():
    # Supressing outputs from the solver
    set_log_level(50)

    # Calling the solver
    solve_problem()

    # Ploting the solution
    plot_solution()

if __name__ == "__main__":
    main()
