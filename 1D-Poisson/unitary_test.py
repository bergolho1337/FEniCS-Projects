"""
    Solves a 1D-Poisson equation using FEniCS

    Problem 1:
        -u'' = 2*cos(x)/exp(x)              (equation)
        u_D = g(x) = sin(x) / exp(x)        (boundary condition)

    Analitical solution:
        u(x) = sin(x) / exp(x)
"""

from dolfin import *
import sys
import os
import numpy as np

def write_aproximation (xmin,xmax,nelem,u):
    file = open("output/aprox.dat","w")
    x = np.linspace(xmin,xmax,nelem+1)
    for i in range(len(x)):
        file.write("%.10lf %.10lf\n" % (x[i],u[i]))

def write_analitical (xmin,xmax):
    file = open("output/analit.dat","w")
    x = np.linspace(xmin,xmax,100)
    for i in range(len(x)):
        file.write("%.10lf %.10lf\n" % (x[i],analitical_solution(x[i])))

# Analitical solution of teh current problem
def analitical_solution (x):
    return sin(x) / exp(x)

# Solve the Boundary Value Problem using FEniCS
def solve_BVP (nelem,degree):

    # Define the boundaries
    xmin = 0.0
    xmax = 10.0
    h = (xmax-xmin) / nelem

    # Create the mesh
    mesh = IntervalMesh(nelem,xmin,xmax)

    # Create the test and trial spaces
    # Here we are using Continuous Galerkin
    V = FunctionSpace(mesh,"CG",degree)

    # Define the boundary condition function
    u_D = Expression('sin(x[0]) / exp(x[0])', degree=degree)

    def boundary(x, on_boundary):
            return on_boundary

    bc = DirichletBC(V,u_D,boundary)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Expression('2*cos(x[0])/exp(x[0])',degree=degree)

    # Bilinear and linear formulas
    a = dot(grad(u),grad(v))*dx
    L = f*v*dx

    # Compute solution
    u = Function(V)
    solve(a == L, u, bc)

    # Compute error in L2 norm
    error_L2 = errornorm(u_D, u, 'L2')

    # Compute values of the analitical and aproximate solution
    vertex_values_u_D = u_D.compute_vertex_values(mesh)
    vertex_values_u = u.compute_vertex_values(mesh)

    # Write results into a file for pos-processing
    write_analitical(xmin,xmax)
    write_aproximation(xmin,xmax,nelem,vertex_values_u)

    # Dump solution to file in VTK format
    file = File("vtk/aprox.pvd")
    file << u

    return h, error_L2

def run_simulations (nelem,nsimulations,degrees):
    # Run the simulations for each degree of the Continuous Galerkin elements
    for degree in degrees:
        print("*****************************************************************************")
        print("[!] Running tests with Continous Galerkin Elements (degree = %d)" % degree)
        filename = "output/error_cg_%d.dat" % degree
        file = open(filename,"w")
        elem = nelem
        for k in range(nsimulations):
            print("---------------------------------------------------------------------------------")
            print("[!] Solving BVP using %d elements" % elem)
            h, error_L2 = solve_BVP(elem,degree)
            file.write("%g %g\n" % (h,error_L2))
            elem = elem * 2
            print("---------------------------------------------------------------------------------")
        print("*****************************************************************************")
        file.close()

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

def unitary_test (argv):
    # Parameters configuration
    toler = 5.0e-01
    nelem = int(argv[1])
    nsimulations = int(argv[2])
    degrees = [1,2]

    # Run all the simulations
    run_simulations(nelem,nsimulations,degrees)

    # Run the Unitary Test
    run_unitary_test(toler,degrees)

def main():
    if (len(sys.argv) != 3):
        print("==============================================================")
        print("Usage:> python3 solver.py <nelem> <nsimulations>")
        print("\t<nelem> = Initial number of elements")
        print("\t<nsimulations> = Number of simulations")
        print("==============================================================")
        sys.exit(1)

    # Supressing outputs from the solver
    set_log_level(50)

    # Calling Unitary test function
    unitary_test(sys.argv)

if __name__ == "__main__":
    main()
