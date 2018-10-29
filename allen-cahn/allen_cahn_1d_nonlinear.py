import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dolfin import *

# Model parameters
lmbda  = 5.0e-02    # Surface parameter
dt     = 5.0e-03    # Time step
theta  = 0.5        # Time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson
M = 1.0             # Diffusive factor
xmin = -2.0         # Limits of the interval
xmax = 2.0          # Limits of the interval
nelem = 100         # Number of finite elements to use
w0 = 0.5            # Weight related to the free-energy density
w1 = 0.0            # Weight related to the free-energy density
n_timesteps = 500   # Number of timesteps

# Class representing the intial conditions
class InitialConditions(UserExpression):
    def __init__(self, **kwargs):
        random.seed(2 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)
    def eval(self, values, x):
        values[0] = ( 1.0 - np.tanh(x / (2.0*np.sqrt(2.0*lmbda))) ) / 2.0
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
    L1 = mu*v*dx + dfdc*v*dx + lmbda*dot(c,v)*dx
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
    t = 0.0
    T = n_timesteps*dt
    while (t < T):
        t += dt
        u0.vector()[:] = u.vector()
        solver.solve(problem, u.vector())

        file << (u.split()[0], t)
        vertex_values_u = u.split()[0].compute_vertex_values(mesh)
        aprox_file.write("%g " % t)
        for i in range(len(vertex_values_u)-1):
            aprox_file.write("%g " % vertex_values_u[i])
        aprox_file.write("%g\n" % vertex_values_u[len(vertex_values_u)-1])
    aprox_file.close()

def compute_analitical ():
    analit_file = open("output/analit.dat","w")
    x = np.linspace(xmin,xmax,nelem+1)
    t = 0.0
    T = n_timesteps*dt
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

    analit_file.close()

def compare_analitical_aproximation ():
    # Specify the timestep to plot
    timestep_plot = 200
    
    # Get the data
    data_analit = np.genfromtxt("output/analit.dat")
    data_aprox = np.genfromtxt("output/aprox.dat")
    x = np.linspace(xmin,xmax,nelem+1)

    plt.clf()
    plt.grid()
    plt.plot(x,data_analit[timestep_plot][1:],label="analit",c="red")
    plt.plot(x,data_aprox[timestep_plot][1:],label="aprox",c="blue",linestyle='--')
    plt.xlabel("x",fontsize=15)
    plt.ylabel("u",fontsize=15)
    plt.title("Analitical x Aproximation (t = %g)" % timestep_plot,fontsize=14)
    plt.legend(loc=0,fontsize=14)
    plt.savefig("output/comparison.pdf")
    #plt.show()

def main ():
    compute_aproximation()
    compute_analitical()
    compare_analitical_aproximation()

if __name__ == "__main__":
    main()