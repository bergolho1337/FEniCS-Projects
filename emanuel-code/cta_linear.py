from dolfin import *
from mshr import*
from fenics import *
import numpy as np
import scipy.io


#----------------------
# Parametros do modelo
#----------------------
lmbda  = 5.0e-03
invlmda= 0.2e+03
dt     = 1.0e-02  #1.0e-03
difus  = 100.0
sigma_o= 0.0
P      = 1.0
Ap     = 0.5


''' Cria a malha '''
domain = Rectangle(Point(0,0), Point(25.6,25.6))
mesh = generate_mesh(domain,80)

''' Define Elementos triang. coninuo de grau 1  e
    espaco das funcoes mistas '''
Q = FiniteElement('CG',triangle,1)
V = FunctionSpace(mesh, MixedElement([Q,Q,Q]))

''' Cria funcoes admissiveis e testes '''
w = Function(V)
dw = Function(V)
(u_1, u_2, u_3) = split(w)
v_1, v_2, v_3 = TestFunctions(V)

 
''' Define cond de contorno Dirichlet '''
bcs = DirichletBC(V.sub(2),Constant(1.0),"on_boundary")
#bcs = DirichletBC(V.sub(2),Constant(1.0),DomainBoundary())

''' Cria as funcoes do passo anterior '''
w_n = Function(V)
(u_n1,u_n2,u_n3) = split(w_n)

''' Cria cond. inicial '''
u_01 = Expression('pow((x[0]-12.8)/2.1,2)+pow((x[1]-12.8)/1.9,2) <=1.0 ? 1.0 : 0',
                   Q, degree=2)

#u_1  = interpolate(u_01, V.sub(0).collapse())
u_n1 = interpolate(u_01, V.sub(0).collapse())

#assign(w.sub(0), u_1)
assign(w_n.sub(0), u_n1)


''' Forma Variacional com o metodo separador convexo '''
''' de Eyre '''

L0 = (u_n1+difus*(1.0-u_n1))*dot(grad(u_3),grad(v_3))*dx + u_n1*u_3*v_3*dx 

L1 = (u_1-u_n1)*v_1*dx + dt*invlmda*u_n1**2*dot(grad(u_2), grad(v_1))*dx \
     -dt*P*u_3*u_1*v_1*dx + dt*Ap*u_1*v_1*dx 

L2 = u_2*v_2*dx - 0.045*(u_1-0.5)*v_2*dx - 18.0*(u_n1-0.25)**3*v_2*dx \
    - lmbda*lmbda*dot(grad(u_1), grad(v_2))*dx

F = L0 + L1 + L2

#F = (u_1+difus*(1.0-u_1))*dot(grad(u_3),grad(v_3))*dx + u_1*u_3*v_3*dx \
#   +u_1*v_1*dx + dt*invlmda*u_1**2*dot(grad(u_2), grad(v_1))*dx \
#   -dt*P*u_3*u_1*v_1*dx+dt*Ap*u_1*v_1*dx \
#   +u_2*v_2*dx-0.09*(u_1-1.0)*(2.0*u_1**2-u_1)*v_2*dx\
#   -lmbda*lmbda*dot(grad(u_1),grad(v_2))*dx

''' Saida da solucao em VTU '''
vtkfile_u_1 = File('Solucao_CTA/u_1.pvd')
vtkfile_u_2 = File('Solucao_CTA/u_2.pvd')
vtkfile_u_3 = File('Solucao_CTA/u_3.pvd')


J = derivative(F, w, dw)

''' Inicializa o metodo de solucao '''
problem = NonlinearVariationalProblem(F, w, bcs=bcs, J=J)
solver = NonlinearVariationalSolver(problem)
solver.parameters['newton_solver']['relative_tolerance'] = 1e-6
solver.parameters['newton_solver']['linear_solver'] = 'mumps'

''' Resolve o problema '''
t = 0.0
T = 20.0 
num_steps = int(T/dt)
progress = Progress('Time-stepping')

for n in range(num_steps):
    t += dt
#    solve(F == 0, w, bcs)  
    
    solver.solve()
    # Save solution to file (VTK)
    if n%(num_steps/10)==0 :
        _u_1, _u_2, _u_3 = w.split(True)
        vtkfile_u_1 << (_u_1, t)
        vtkfile_u_2 << (_u_2, t)
        vtkfile_u_3 << (_u_3, t)

    # Update previous solution
    w_n.assign(w)
    
    # Update progress bar
    progress.update(t / T)


interactive()





