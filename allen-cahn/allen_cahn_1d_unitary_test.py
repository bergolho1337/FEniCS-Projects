from dolfin import *
import sys
import os
import random
import numpy as np

# Allen-Cahn equation parameters
lmbda  = 5.0e-02    # Surface parameter
dt     = 1.0e-03    # Time step
tmax = 2.0          # Maximum time of the simulation
theta  = 0.5        # Time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson
M = 1.0             # Diffusive factor
xmin = -2.0         # Limits of the interval
xmax = 2.0          # Limits of the interval
#nelem = 64         # Number of finite elements to use
w0 = 0.0            # Weight related to the free-energy density
w1 = 0.0            # Weight related to the free-energy density
#timestep_plot = 2000 # Timestep of the plot
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

def run_unitary_test (toler,degrees):
    # Read the output files and check convergeance using a Unitary test
    print("[!] Running Unitary Test")
    for degree in degrees:
        print("------------------------------------------------------------------------------------")
        print("[!] Checking solution using Continous Galerkin (degree = %d)" % degree)
        # Read the data
        filename = "output/error_cg_%d.dat" % degree
        data = np.genfromtxt(filename)
        
        log_h = np.log10(data[:,0])
        log_E = np.log10(data[:,1])

        # Use the 2 last points to interpolate the error function
        coeff = np.polyfit(log_h,log_E,1)
        a = coeff[0]
        b = coeff[1]
        print("Error function")
        print("y = %g . x + %g" % (a,b))

        # Calculate reference coefficient using the Finite Element error formula ...
        a_ref = degree+1
        diff = np.abs(a-a_ref)
        if (diff < toler):
            print("[+] Sucess ! Continuos Galerkin with degree %d has passed in the Unitary test!" % degree)
        else:
            print("[-] ERROR ! Continuos Galerkin with degree %d has failed in the Unitary test!" % degree)
        print("|| a = %.10lf || a_ref = %.10lf || diff = %.10lf || toler = %.10lf" % (a,a_ref,diff,toler))
        print("------------------------------------------------------------------------------------")

def solve_allen_cahn (nelem,degree,error_timestep):

    start_h = float((xmax-xmin)/nelem)

    # Form compiler options
    parameters["form_compiler"]["optimize"]     = True
    parameters["form_compiler"]["cpp_optimize"] = True

    # Create mesh and build function space
    mesh = IntervalMesh(nelem,xmin,xmax)
    P1 = FiniteElement("CG",mesh.ufl_cell(),degree)
    ME = FunctionSpace(mesh,P1*P1)

    # Define analitical solution
    u_analit = Expression('(1.0 - tanh( (x[0] - (sqrt(2.0*lmbda)*(6.0*(w0 - w1)))*t)/(2.0 * sqrt(2.0*lmbda)) )) / 2.0',
                 degree=degree, lmbda=lmbda, w0=w0, w1=w1, t=0)

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
    #file = File("vtu/output.pvd", "compressed")

    # Step in time
    k = 0
    T = tmax
    n_timesteps = int(tmax/dt)
    while (k <= n_timesteps):
        t = k*dt

        u0.vector()[:] = u.vector()
        solver.solve(problem, u.vector())

        #file << (u.split()[0], t)
        #vertex_values_u = u.split()[0].compute_vertex_values(mesh)
        
        # Compute error in L2 norm
        if (k == error_timestep):
            error_L2 = errornorm(u_analit, u.split()[0], 'L2')
        
        k = k + 1
    return start_h, error_L2

def run_simulations (nelem,nsimulations,error_timestep,degrees):
    # Run the simulations for each degree of the Continuous Galerkin elements
    for degree in degrees:
        print("*****************************************************************************")
        print("[!] Running tests with Continous Galerkin Elements (degree = %d)" % degree)
        filename = "output/error_cg_%d.dat" % degree
        file = open(filename,"w")
        elem = nelem
        for k in range(nsimulations):
            print("---------------------------------------------------------------------------------")
            print("[!] Solving Allen-Cahn-1D using %d elements" % elem)
            h, error_L2 = solve_allen_cahn(elem,degree,error_timestep)
            file.write("%g %g\n" % (h,error_L2))
            elem = elem * 2
            print("---------------------------------------------------------------------------------")
        print("*****************************************************************************")
        file.close()

def unitary_test (argv):
    # Parameters configuration
    toler = 5.0e-01
    nelem = int(argv[1])
    nsimulations = int(argv[2])
    error_timestep = int(argv[3])
    degrees = [1,2,3]

    # Run all the simulations
    run_simulations(nelem,nsimulations,error_timestep,degrees)

    # Run the Unitary Test
    run_unitary_test(toler,degrees)

def main():
    if (len(sys.argv) != 4):
        print("==========================================================================")
        print("Usage:> python3 allen_cahn_1d_unitary_test.py <nelem> <nsimulations>")
        print("                                              <error_timestep>")
        print("\t<nelem> = Initial number of elements")
        print("\t<nsimulations> = Number of simulations")
        print("\t<error_timestep> = Timestep the L2-norm error will be calculated")
        print("==========================================================================")
        sys.exit(1)

    # Supressing outputs from the solver
    set_log_level(50)

    # Calling Unitary test function
    unitary_test(sys.argv)

if __name__ == "__main__":
    main()