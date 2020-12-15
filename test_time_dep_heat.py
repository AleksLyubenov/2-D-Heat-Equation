from dolfin import *
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import matplotlib.pyplot as plt
import numpy.linalg as npl
    
'''
    NOTATION:
    x[0] == x
    x[1] == y
    
    BOUNDARIES:
    LEFT:   1
    TOP:    2
    RIGHT:  3
    BOTTOM: 4
'''

#----------------------------------------------------------------------------------------------------------------------------

# Define mesh (which we assume to be a 2D square with dx =dy) 
n_space = 100
#mesh = UnitSquareMesh(n_space, n_space)
mesh = RectangleMesh(Point(0., 0.), Point(pi, pi), n_space, n_space)

# Define the function space
V = FunctionSpace(mesh, "Lagrange", 1) 

# Create classes to define the subdomains top, bottom, left and right within our 2D box domain
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0)
        # Or equivalent to the above: return x[0] < 5* DOLFIN_EPS

class Right(SubDomain):
    def inside(self,x, on_boundary):
        return near(x[0], np.pi)
        
class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0)
        
class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], np.pi)
        
# Instantialize the subdomain objects
left = Left()
top = Top()
right = Right()
bottom = Bottom()

# Create mesh functions for the parts of our domain on the boundaries
boundaries = MeshFunction('size_t',mesh,mesh.topology().dim()-1)
boundaries.set_all(0)

# Mark (name) our boundaries 
left.mark(boundaries, 1)
top.mark(boundaries, 2)
right.mark(boundaries, 3)
bottom.mark(boundaries, 4)

#----------------------------------------------------------------------------------------------------------------------------

# Define time parameters
dt = 0.1
t=dt
t_end = 5

# Define the initial condition (u at t=0)
u_init = Expression('0', degree=2, t=0)
u_prev = interpolate(u_init, V)

# Define Dirichlet boundary conditions
u_d = Expression('sin(pow(x[1],2))*pow(t,2)', degree=2, t=dt)
u_d = interpolate(u_d, V)
bcs = [DirichletBC(V, u_d, boundaries, 1)]

# Define a new unit of measurement 's' for the boundary mesh
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# g_i are the Neumann boundary conditions (Set unused conditions to: Constant(0.0))
g1 = Constant(0.0)
g2 = Expression("2*pow(t,2)*x[1]*cos(pow(x[0],2)+pow(x[1],2))", degree=2, t=dt) 
g3 = Expression("2*pow(t,2)*x[0]*cos(pow(x[0],2)+pow(x[1],2))", degree=2, t=dt) 
g4 = Expression("-2*pow(t,2)*x[1]*cos(pow(x[0],2)+pow(x[1],2))", degree=2, t=dt)  

#----------------------------------------------------------------------------------------------------------------------------

# Define the Variational Form of the problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression('2*t*sin(pow(x[0],2)+pow(x[1],2)) - 4*pow(t,2)*cos(pow(x[0],2)+pow(x[1],2)) + 4*pow(t,2)*(pow(x[0],2)+pow(x[1],2))*sin(pow(x[0],2)+pow(x[1],2))', 
degree=2, t=dt)

# a(u,v) in the variational form (Note: we omit the integral(s))
a = u*v*dx + dt*inner(grad(u),grad(v))*dx
# l(v) in the variational form (Note: we omit the intergral(s))
L = u_prev*v*dx  + dt*g1*v*ds(1) + dt*g2*v*ds(2) + dt*g3*v*ds(3) + dt*g4*v*ds(4) + dt*f*v*dx

#----------------------------------------------------------------------------------------------------------------------------   
    
# Time Loop
while t <= t_end + dt/2:
    # Update time in all functions 
    u_d.t=t
    g1.t=t
    g2.t=t
    g3.t=t
    g4.t=t
    f.t=t
    
    # Compute solution
    u = Function(V)
    
    #A, b = assemble_system(a, L, bcs)
    A = assemble(a)
    b = assemble(L)
    
    for i in bcs:
        i.apply(A)
        i.apply(b)

    # Am is our stiffness matrix A
    A = A.array()
    Am = sp.csc_matrix(A)

    #Create the ILU preconditioner
    M2 = spl.spilu(Am)
    M_x = lambda x: M2.solve(x)     
    M = spl.LinearOperator(((n_space+1)**2,(n_space+1)**2), M_x)
    
    # bm is our RHS vector b
    bm = b.get_local()
    
    u_prevm=u_prev.vector()
    
    # Recall, bicgstab returns the solution vector u && information about the algorithm execution
    uvec, info = spl.bicgstab(Am,bm,x0=u_prevm,M=M) 
    
    u.vector()[:] = uvec
    u_prev.assign(u) 
    
    # Ensure that solution is computed at t_end regardless of dt chosen
    if t == t_end:
         break
    t += dt
    if t > t_end:
        t = t_end

#----------------------------------------------------------------------------------------------------------------------------   

# Save pieces of the solution
# Note: ParaView v. 5.6 is required to view these pvd files
File("u_final.pvd") << u
File("mesh.pvd") << mesh
File("boundaries.pvd") << boundaries

plot(u)
plt.savefig('u.png')
