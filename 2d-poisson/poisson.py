from fenics import *
import sys
import math

def print_message (ndiv,h):
	print("*********************************************************************")
	print("Solving Poisson equation using %d divisions -- h = %g" % (ndiv,h))
	print("*********************************************************************")

def solve_problem (ndiv):

	# Calculate spatial discretization
	h = float(1.0/ndiv)
	# Output a message to the user
	print_message(ndiv,h)
	# _________________________________________________________________
	# Create mesh and define function space
	# Square mesh consist of ndiv by ndiv squares, where each square is divided in two triangles 
	mesh = UnitSquareMesh(ndiv, ndiv)
	# Create the finite element function space V
	# Function space consist of standard Lagrange elements with degree 1
	V = FunctionSpace(mesh, 'P', 2)

	# __________________________________________________________________
	# Define boundary condition
	u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)

	def boundary(x, on_boundary):
	    return on_boundary

	bc = DirichletBC(V, u_D, boundary)

	# __________________________________________________________________
	# Define variational problem
	u = TrialFunction(V)
	v = TestFunction(V)
	f = Constant(-6.0)
	a = dot(grad(u), grad(v))*dx
	L = f*v*dx

	# __________________________________________________________________
	# Compute solution
	u = Function(V)
	solve(a == L, u, bc)

	# __________________________________________________________________
	# Plot solution and mesh
	plot(u)
	plot(mesh)

	# __________________________________________________________________
	# Save solution to file in VTK format
	vtkfile = File('poisson/solution%d.pvd' % ndiv)
	vtkfile << u

	# __________________________________________________________________
	# Compute error in L2 norm
	error_L2 = errornorm(u_D, u, 'L2')

	# __________________________________________________________________
	# Compute maximum error at vertices
	vertex_values_u_D = u_D.compute_vertex_values(mesh)
	vertex_values_u = u.compute_vertex_values(mesh)
	import numpy as np
	error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))

	return h, error_L2, error_max

def main():

	if (len(sys.argv) != 2):
		print("========================================================")
		print("Usage:> python poisson.py <number_of_simulations>")
		print("========================================================")
		sys.exit(1)

	n_simulations = int(sys.argv[1])
	outFile = open("output.dat","w")

	# Number of divisions
	ndiv = 2
	for i in range(n_simulations):
		h, error_L2, error_max = solve_problem(ndiv)
		outFile.write("%g %g %g\n" % (h,error_L2,error_max))
		ndiv = ndiv * 2 
	
	outFile.close()


if __name__ == "__main__":
    main()

