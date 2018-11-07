from dolfin import *
from mshr import*
from fenics import *
import numpy as np
import scipy.io

# Class representing the intial conditions
class InitialConditions(Expression):
    def eval(self, values, x):
       if( pow((x[0] - 12.8)/2.1, 2) + pow((x[1] - 12.8)/1.9, 2) == 1.0):
             values[0] = 1.0
       else:
             values[0] = 0.0


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

def Boundary(x, on_boundary):
   return on_boundary

#----------------------
# Parametros do modelo
#----------------------
lmbda  = 5.0e-03
invlmda= 0.2e+03
dt     = 1.0e-02  #1.0e-03
theta  = 1.0      # Euler implicito, theta=0.5 -> Crank-Nicolson.
difus  = 100.0
sigma_o= 0.0
P      = 1.0
Ap     = 0.5

#-----------------------
# Form compiler options
#-----------------------
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "quadrature"

#-----------------------------------------------------------------
# Crea a malha, outro importa malha criada no Gmsh em formato xml
#-----------------------------------------------------------------

domain = Rectangle(Point(0,0), Point(25.6,25.6))
mesh = generate_mesh(domain,80)
#mesh = UnitSquareMesh(64, 64)


#P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
#ME = FunctionSpace(mesh, P1*P1*P1)
P1 = FiniteElement("CG", triangle, 1)
W_elem = MixedElement([P1, P1, P1])
ME = FunctionSpace(mesh, W_elem)

#------------------------------------
# Define funcoes admissiveis e teste
#------------------------------------
#du = TrialFunction(ME)
du = Function(ME)
q  = TestFunctions(ME)
v  = TestFunctions(ME)
w  = TestFunctions(ME)

#----------------
# Define funcoes
#----------------
u   = Function(ME)  # current solution
u0  = Function(ME)  # solution from previous converged step

#-----------------------
# Split mixed functions
#-----------------------
dc  = split(du)
dmu = split(du)
dphi= split(du)

c  = split(u)  
mu = split(u)
phi= split(u)

c0 = split(u0) 
mu0= split(u0)


#------------------------------------------
# Create intial conditions and interpolate
#------------------------------------------
u_init = Expression('pow((x[0]-12.8)/2.1,2)+pow((x[1]-12.8)/1.9,2) ==1.0 ? 1.0 : 0',
                    P1, degree=2)

#u_init = InitialConditions(degree=2)
u =interpolate(u_init, ME.sub(0).collapse())
u0=interpolate(u_init, ME.sub(0).collapse())

# Dirichlet bc
g = Constant(1.0)
bc= DirichletBC(ME.sub(2), g, Boundary)


#--------------------------------------
# Compute the chemical potential df/dc
#--------------------------------------
#c    = variable(c)
#f    = 0.25*0.18*pow(c,2)*pow((c-1),2)
#dfdc = diff(f, c)

# Weak statement of the equations
# Erro ta aki !!!!
L0 = c*q*dx - c0*q*dx + dt*invlmda*dot( grad(pow(c,2)*mu),grad(q) )*dx - dt*P*phi*c*q*dx + dt*Ap*c*q*dx
L1 = mu*v*dx - 0.09*(c-1.0)*(2.0*c-1.0)*c*v*dx - lmbda*lmbda*dot(grad(c), grad(v))*dx
L2 = (c+difus*(1.0-c))*dot(grad(phi),grad(w))*dx + c*phi*w*dx
L = L0 + L1 + L2

# Compute directional derivative about u in the direction of du (Jacobian)      
a = derivative(L, u, du)

#--------------------------------------------
# Create nonlinear problem and Newton solver
#--------------------------------------------

problem = CahnHilliardEquation(a, L)
solver = NewtonSolver()

#solver.parameters["linear_solver"] = "lu"
#solver.parameters["convergence_criterion"] = "incremental"
#solver.parameters["relative_tolerance"] = 1e-6

#-------------------  Novo -----------------------------------

solver.parameters["linear_solver"] = "gmres"
solver.parameters["preconditioner"] ="ilu"
solver.parameters["convergence_criterion"] = "incremental"
solver.parameters["krylov_solver"]["absolute_tolerance"] =1e-6
solver.parameters["krylov_solver"]["relative_tolerance"] =1e-6


#-------------
# Output file
#-------------

file = File("output.pvd", "compressed")

#--------------
# Step in time
#--------------

t = 0.0
T = 20.0 #1.0e-01
while (t < T):
    t += dt
    u0.vector()[:] = u.vector()
    solver.solve(problem, u.vector(), bc)  
    file << (u.split()[0], t)

plot(u.split()[0])
interactive()
