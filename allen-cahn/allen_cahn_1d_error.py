import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dolfin import *

# Model parameters
lmbda  = 5.0e-02    # Surface parameter
dt     = 5.0e-03    # Time step
tmax   = 2.5
theta  = 0.5        # Time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson
M = 1.0             # Diffusive factor
xmin = -2.0         # Limits of the interval
xmax = 2.0          # Limits of the interval
#nelem = 4         # Number of finite elements to use
w0 = 0.0            # Weight related to the free-energy density
w1 = 0.0            # Weight related to the free-energy density
n_timesteps = 500   # Number of timesteps
ref_timestep = 2    # Reference timestep for error calculation
n_simulations = 3   # Number of simulations

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

def solve_problem (nelem):

    start_h = float((xmax-xmin)/nelem)

    # Form compiler options
    parameters["form_compiler"]["optimize"]     = True
    parameters["form_compiler"]["cpp_optimize"] = True

    # Create mesh and build function space
    mesh = IntervalMesh(nelem,xmin,xmax)
    P1 = FiniteElement("Lagrange",mesh.ufl_cell(),2)
    ME = FunctionSpace(mesh,P1*P1)

    # Define analitical solution
    u_analit = Expression('(1.0 - tanh( (x[0] - (sqrt(2.0*lmbda)*(6.0*(w0 - w1)))*t)/(2.0 * sqrt(2.0*lmbda)) )) / 2.0',
                 degree=2, lmbda=lmbda, w0=w0, w1=w1, t=0)

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
    #file = File("vtu/output.pvd", "compressed")

    # Step in time
    k = 0
    T = tmax
    while (k <= n_timesteps):
        t = k*dt

        u0.vector()[:] = u.vector()
        solver.solve(problem, u.vector())

        #file << (u.split()[0], t)
        #vertex_values_u = u.split()[0].compute_vertex_values(mesh)
        
        # Compute error in L2 norm
        if (k == ref_timestep):
            error_L2 = errornorm(u_analit, u.split()[0], 'L2')
        
        k = k + 1
    return start_h, error_L2

def main ():
    error_file = open("output/l2_error.dat","w")
    # Starting number of elements
    nelem = 1
    
    for k in range(n_simulations):
        h, error_L2 = solve_problem(nelem)
        error_file.write("%g %g\n" % (h,error_L2))
        nelem = nelem * 2
    error_file.close()

if __name__ == "__main__":
    main()