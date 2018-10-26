"""
    Solves a 1D-Poisson equation using FEniCS

    1D Cahn-Hilliard:
        u'' = H*u*(1-u)*(1-2u) / eps**2              (equation)
        2 * eps**2 * u' = 0                          (boundary)

    Analitical solution:
        u(x) = 0.5 * (1 + tanh(x/(2*l))),
        l = sqrt(2/H)*eps
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

# Function to get a string parse to an integer, double its value and return
# this value as a string
def double_string (value):
    int_value = int(value)
    int_value = int_value * 2
    return str(int_value)

# [Unitary Test]
# Function to solve a number of problems using diferent configurations 
# to check the convergeance of the method
def check_convergeance (argv):
    nsimulations = 10
    file = open("output/error.dat","w")
    for i in range(nsimulations):
        print("[!] Solving BVP using %d elements")
        
        h, error_L2 = solve_BVP(argv)
        
        file.write("%g %g\n" % (h,error_L2))
        
        argv[1] = double_string(argv[1])

# Solve the Boundary Value Problem using FEniCS
def solve_BVP (argv):
    # Input parameters
    nelem = int(argv[1])               # Number of elements
    degree = int(argv[2])              # Degree of polynomial
    # Constants
    H = 1.0
    EPS = 0.05

    # Define the boundaries
    xmin = -1.0
    xmax = 1.0
    h = (xmax-xmin) / nelem

    # Create the mesh
    mesh = IntervalMesh(nelem,xmin,xmax)

    # Create the test and trial spaces
    # Here we are using Continuous Galerkin
    V = FunctionSpace(mesh,"CG",degree)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Expression('H*u*(1-u)*(1-2u) / eps**2',degree=2)

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

    #print("Analitical")
    #print(vertex_values_u_D)

    #print("Aproximation")
    #print(vertex_values_u)

    # Dump solution to file in VTK format
    file = File("vtk/aprox.pvd")
    file << u

    print("******************************************************************")
    print("Nelem = %d" % nelem)
    print("Degree = %d" % degree)
    print("h = %.10lf" % h)
    print("L2 norm = %.10lf" % error_L2)
    print("******************************************************************")

    return h, error_L2

def main():
    if (len(sys.argv) != 3):
        print("==============================================================")
        print("Usage:> python3 solver.py <nelem> <degree>")
        print("\t<nelem> = Number of elements")
        print("\t<degree> = Degree of the interpolation polynomial")
        print("==============================================================")
        sys.exit(1)

    solve_BVP(sys.argv)
    #check_convergeance(sys.argv)

if __name__ == "__main__":
    main()
