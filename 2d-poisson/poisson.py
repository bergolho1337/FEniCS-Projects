from fenics import *
import math

# Number of divisions
ndiv = 2
h = float(1.0/ndiv)
# _________________________________________________________________
# Create mesh and define function space
# Square mesh consist of 8 by 8 squares, where each square is divided in two triangles 
mesh = UnitSquareMesh(ndiv, ndiv)
# Create the finite element function space V
# Function space consist of standard Lagrange elements with degree 1
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition
u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6.0)
a = dot(grad(u), grad(v))*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Plot solution and mesh
plot(u)
plot(mesh)

# Save solution to file in VTK format
vtkfile = File('poisson/solution.pvd')
vtkfile << u

# Compute error in L2 norm
error_L2 = errornorm(u_D, u, 'L2')

# Compute maximum error at vertices
vertex_values_u_D = u_D.compute_vertex_values(mesh)
vertex_values_u = u.compute_vertex_values(mesh)
import numpy as np
error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))

# Print errors
print('h = ', h)
print('log h = ', math.log(h))
print('error_L2  =', error_L2)
print('log error_L2 = ', math.log(error_L2))
print('error_max =', error_max)

# Hold plot
#interactive()
