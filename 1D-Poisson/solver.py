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
    u_D = Expression('sin(x[0]) / exp(x[0])', degree=1)

    def boundary(x, on_boundary):
            return on_boundary

    bc = DirichletBC(V,u_D,boundary)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Expression('2*cos(x[0])/exp(x[0])',degree=1)

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

    #solve_BVP(sys.argv)
    check_convergeance(sys.argv)

if __name__ == "__main__":
    main()
