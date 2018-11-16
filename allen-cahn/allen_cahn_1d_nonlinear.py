# ============================================================================================================
# Program that solves the 1-D Allen-Cahn equation using the FEniCS library.
# ------------------------------------------------------------------------------------------------------------
# All the output files for pos-processing (error analysis, etc ...) are being store over the 'output' folder.
# ------------------------------------------------------------------------------------------------------------
# This program uses the NewtonMethod to solve the nonlinear system of equations that appear when we apply 
# a variable substitution over the term related to the Laplacian operator. 
# (For more information about the solution steps read the 'solution_guide.pdf' file).
# ------------------------------------------------------------------------------------------------------------
# Another feature of the program is the comparison between the approximate and analitical solutions. The user
# needs to specify a timestep as a reference for the plot that overlaps the two solutions then a 'pdf' file
# will be generated at the 'output' folder with the name 'comparison.pdf'. 
# ------------------------------------------------------------------------------------------------------------
# Furthermore, the program also generates two files 'aprox.dat' and 'analit.dat' that represents the 
# approximate and analitical solution respectevely.
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
#lmbda  = 5.0e-02    # Surface parameter
lmbda  = 5.0e-02    # Surface parameter
dt     = 1.0e-03    # Time step
tmax = 2.0          # Maximum time of the simulation
theta  = 0.5        # Time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson
M = 1.0             # Diffusive factor
xmin = -2.0         # Limits of the interval
xmax = 2.0          # Limits of the interval
nelem = 32         # Number of finite elements to use
w0 = 0.0            # Weight related to the free-energy density
w1 = 0.0            # Weight related to the free-energy density
timestep_plot = 0 # Timestep of the plot
print_rate = 10     # Rate which the VTU file will be saved

# Class representing the initial conditions
# Type 1 = Analitical solution
class InitialConditions_Type1(UserExpression):
    def __init__(self, **kwargs):
        random.seed(2 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)
    def eval(self, values, x):
        values[0] = ( 1.0 - np.tanh(x / (2.0*np.sqrt(2.0*lmbda))) ) / 2.0
        values[1] = 0.0
    def value_shape(self):
        return (2,)

# Class representing the initial conditions
# Type 2 = Single pulse
class InitialConditions_Type2(UserExpression):
    def __init__(self, **kwargs):
        random.seed(2 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)
    def eval(self, values, x):
        if (x <= 0.5):
            values[0] = 1.0
        else:
            values[0] = 0.0
        #values[0] = ( 1.0 - np.tanh(x / (2.0*np.sqrt(2.0*lmbda))) ) / 2.0
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
    file = File("vtu/output.pvd", "compressed")
    aprox_file = open("output/aprox.dat","w")

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

def compute_analitical ():
    print("[!] Computing analitical solution ...")

    analit_file = open("output/analit.dat","w")
    x = np.linspace(xmin,xmax,nelem+1)
    k = 0
    n_timesteps = int(tmax/dt)
    while (k <= n_timesteps):

        t = dt*k

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
        k = k + 1

    analit_file.close()

def plot_aprox_timestep (x,data_aprox,timestep_plot):
    plt.plot(x,data_aprox[timestep_plot][1:],label="t = %.3lf" % float(timestep_plot*dt),linestyle='--')

def plot_analit_timestep (x,data_analit,timestep_plot):
    plt.plot(x,data_analit[timestep_plot][1:],label="t = %.3lf" % float(timestep_plot*dt),linestyle='-')

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

def compare_analitical_timesteps ():
    
    # Get the data
    data_analit = np.genfromtxt("output/analit.dat")
    x = np.linspace(xmin,xmax,nelem+1)
    timesteps = [1,100,1000,1999]
    #timesteps = [1,1000,10000,19999]

    plt.clf()
    for k in range(len(timesteps)):
        plot_analit_timestep(x,data_analit,timesteps[k])
    plt.grid()
    plt.xlabel("x",fontsize=15)
    plt.ylabel("u",fontsize=15)
    plt.title("Analitical Timesteps",fontsize=14)
    plt.legend(loc=0,fontsize=14)
    plt.savefig("output/analit_timesteps.pdf")


def compare_aproximation_analitical ():
    
    # Get the data
    data_analit = np.genfromtxt("output/analit.dat")
    data_aprox = np.genfromtxt("output/aprox.dat")
    x = np.linspace(xmin,xmax,nelem+1)
    t = dt*2000

    plt.clf()
    plt.plot(x,data_analit[timestep_plot][1:],label="analit",c="red")
    plt.plot(x,data_aprox[timestep_plot][1:],label="aprox-2000",linestyle='--')
    plt.grid()
    plt.xlabel("x",fontsize=15)
    plt.ylabel("u",fontsize=15)
    plt.ylim([0,1])
    plt.title("Analitical x Aproximation (t = %g)" % t,fontsize=14)
    plt.legend(loc=0,fontsize=14)
    plt.savefig("output/comparison.pdf")

def main ():
    # Supressing outputs from the solver
    set_log_level(50)

    compute_aproximation()
    compute_analitical()
    compare_aproximation_timesteps()
    compare_analitical_timesteps()
    compare_aproximation_analitical()

if __name__ == "__main__":
    main()
