from dolfin import *
import numpy as np
import scipy.sparse.linalg as spl

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

# Define mesh (which we assume to be a 2D square with dx =dy) 
n_space = 32
mesh = UnitSquareMesh(n_space, n_space)
# Define the function space (the finite dimensional vector space V_h)
V = FunctionSpace(mesh, "Lagrange", 1) 


# Create classes to define the subdomains top, bottom, left and right within our 2D box domain (As well as the interior??)
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0)
        # Or equivalent to the above: return x[0] < 5* DOLFIN_EPS

class Right(SubDomain):
    def inside(self,x, on_boundary):
        return near(x[0], 1.0)
        
class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0)
        
class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 1.0)
        
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

# Define the boundary condition
u0 = Expression("x[0]*x[1]", degree=2)

# Define Dirichlet boundary conditions on the TOP and BOTTOM boundaries --> u==u0 TOP and BOTTOM
bcs = [DirichletBC(V, u0, boundaries, 2), DirichletBC(V, u0, boundaries, 4)]

# Define a new unit of measurement 's' for the mesh
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# Define the Variational Form of the problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("0", degree=2) # For u=xy

# g_i are the Neumann boundary conditions (Set unused conditions to: Constant(0.0)
g1 = Expression("-x[1]", degree=2)                                  #left   --> g1=-y
g2 = Constant(0.0)                  #Expression("x[0]", degree=2)   #top    --> g2=x
g3 = Expression("x[1]", degree=2)                                   #right  --> g3=y
g4 = Constant(0.0)                  #Expression("-x[0]", degree=2)  #bottom --> g4=-x

# a(u,v) in the variational form (Note: we omit the integral(s))
a = inner(grad(u), grad(v))*dx
# l(v) in the variational form (Note: we omit the intergral(s))
L = f*v*dx + g1*v*ds(1) + g2*v*ds(2) + g3*v*ds(3) + g4*v*ds(4) 

# Compute solution
u = Function(V)

A, b = assemble_system(a, L, bcs)
# Am is our stiffness matrix A
Am = A.array()
# bm is our RHS vector b
bm = b.get_local()
# Recall, bicgstab returns the solution vector u && information about the algorithm execution
uvec, info = spl.bicgstab(Am,bm)

u.vector()[:] = uvec

# Save pieces of the solution
# Note: ParaView v. 5.6 is required to view these pvd files
file = File("poisson.pvd")
file << u
file = File("mesh.pvd")
file << mesh
file = File("boundary.pvd")
file << boundaries

# Plot the solution
import matplotlib.pyplot as plt
plot(u)
plt.savefig('u.png')