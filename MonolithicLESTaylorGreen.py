"""A demo FEniCS script for solving the unsteady form of Navier-Stokes equations
using Q1P1 elements with streamwise Upwind Petrov-Galerkin stabilization for
convection instabilities, and Petrov Galerkin Pressure Stabilzation for linear
velocity-pressure combination.

The demo pertains to unsteady viscous flow through a channel with a step inside.
If the step geometry is modified, this demo can also be used to implement the
classical Backward Facing Step problem. The step geometry is specified in the
file steppedChannel.geo provided along with this script.

Note:
-----
This demo script is accompanied also by a short in-class code implementation
activity, and is paired with an assignment. Focus of this demo and the tutorial
is to illustrate how unsteady incompressible flow problems can be solved using FEniCS.

Last Revised:
-------------
Spring 2023

Disclaimer:
-----------
Developed for computational fluid dynamics class taught at the Paul M Rady
Department of Mechanical Engineering at the University of Colorado Boulder by
Prof. Debanjan Mukherjee.

All inquiries addressed to Prof. Mukherjee directly at debanjan@Colorado.Edu

"""
import fenics as fe
import matplotlib.pyplot as plt
import numpy as np
import fenics as fe
from dolfin import *
from ufl import *
from ufl.classes import *
from ufl.algorithms import *
from ufl.algorithms.formtransformations import compute_form_arities

#-------------------------------------------------------------------
# Definition of some global FEniCS optimization parameters
#
# WE WILL DISCUSS THE MEANING AND IMPLICATIONS OF THESE OPTIMIZATION
# OPERATIONS - AND THEIR ROLES IN GENERATING EFFICIENT COMPILED CODE
#-------------------------------------------------------------------
fe.set_log_level(fe.LogLevel.INFO)
fe.parameters['form_compiler']['representation']    = 'uflacs'
fe.parameters['form_compiler']['optimize']          = True
fe.parameters['form_compiler']['cpp_optimize']      = True
fe.parameters["form_compiler"]["cpp_optimize_flags"]= '-O2 -funroll-loops'

#--------------------------------------------
# definition of problem parameters and inputs
#--------------------------------------------
meshFile    = "steppedChannel.xml"
facetFile   = "steppedChannel_facet_region.xml"
outFileV    = "results/vel-.pvd"
outFileP    = "results/pres-.pvd"
outFileW    = "results/vorticity-.pvd"
outFileF    = "results/forces.dat"
outPlot     = "results/forces.png"
U0          = 3
viscosity   = 1.4e-05
dt          = 0.5e-08
t_end       = 0.2

#----------------------------------------------------------------------------
# Identification of all correct boundary markers needed for the domain
#
# WE WILL DISCUSS THE NEED FOR SYSTEMATIC PRE-PROCESSING AND MODEL GENERATION
# TO ENSURE THE BOUNDARY MARKERS ARE PROPERLY IDENTIFIED
#----------------------------------------------------------------------------
ID_INLET    = 1
ID_TOP      = 2
ID_OUTLET   = 3
ID_BOTTOMR  = 4
ID_BOTTOML  = 8
ID_STEPR    = 5
ID_STEPL    = 7
ID_STEPTOP  = 6

#-------------------------------------------------------
# problem parameters defined in FEniCS compatible syntax
#-------------------------------------------------------
mu          = fe.Constant(viscosity)
idt         = fe.Constant(1.0/dt)
theta       = fe.Constant(1)
a = fe.Constant(1)
nx = 16
ny = 8
nz = 16
#------------------------------------------------
# create the mesh and import mesh into the solver
#------------------------------------------------
mesh = fe.BoxMesh(fe.Point(0,0,0),fe.Point(a,a,a),nx,ny,nz)
class PeriodicBoundary(SubDomain):

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two slave edges
        return bool ((near(x[0], 0) or near(x[1], 0) or near(x[2], 0)) and 
            (not ((near(x[0], a) and near(x[2], a)) or 
                  (near(x[0], a) and near(x[1], a)) or
                  (near(x[1], a) and near(x[2], a)))) and on_boundary)

    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
    	#### define mapping for a single point in the box, such that 3 mappings are required
        if near(x[0], a) and near(x[1], a) and near(x[2],a):
            y[0] = x[0] - a
            y[1] = x[1] - a
            y[2] = x[2] - a
        ##### define mapping for edges in the box, such that mapping in 2 Cartesian coordinates are required
        if near(x[0], a) and near(x[2], a):
            y[0] = x[0] - a
            y[1] = x[1] 
            y[2] = x[2] - a      
        elif near(x[1], a) and near(x[2], a):
            y[0] = x[0] 
            y[1] = x[1] - a
            y[2] = x[2] - a
        elif near(x[0], a) and near(x[1], a):
            y[0] = x[0] - a
            y[1] = x[1] - a
            y[2] = x[2]         
        #### right maps to left: left/right is defined as the x-direction
        elif near(x[0], a):
            y[0] = x[0] - a
            y[1] = x[1]
            y[2] = x[2]
        ### back maps to front: front/back is defined as the y-direction    
        elif near(x[1], a):
            y[0] = x[0]
            y[1] = x[1] - a
            y[2] = x[2] 
        #### top maps to bottom: top/bottom is defined as the z-direction        
        elif near(x[2], a):
            y[0] = x[0]
            y[1] = x[1]
            y[2] = x[2] - a

## Define boundary condition
pbc = PeriodicBoundary()

#----------------------------
# Collect boundary conditions
#----------------------------


#------------------------------------------------------
# Define body forces and some other relevant quantities
# n is the normals to the surfaces/edges of the domain
# I is the 2x2 identity matrix
#------------------------------------------------------

n       = fe.FacetNormal(mesh)
I       = fe.Identity(2)
zero    = fe.Constant(0.0)

#-----------------------------------------------------------------------
# Define the mixed vector function space operating on this meshed domain
#
# TO BE COMPLETED DURING IN CLASS DEMO
#-----------------------------------------------------------------------
V = fe.VectorElement("Lagrange", mesh.ufl_cell(),1)
P = fe.FiniteElement("Lagrange", mesh.ufl_cell(),1)
M = fe.MixedElement([V,P])
W = fe.FunctionSpace(mesh, M, constrained_domain = pbc)

#-------------------------------------
# Define unknown and test function(s)
#
# TO BE COMPLETED DURING IN CLASS DEMO
#-------------------------------------
(v,q) = fe.TestFunctions(W)
#-------------------------------------------------------
# Defining essential/Dirichlet boundary conditions
# Step 1: Identify all boundary segments forming Gamma_d
#-------------------------------------------------------
#domainBoundaries    = fe.MeshFunction("size_t", mesh, facetFile)
#ds                  = fe.ds(subdomain_data=domainBoundaries)

#---------------------------------------------------------
# Step 2: Define all boundary values (u_D) and assign them
# to the appropriate Gamma_d
#---------------------------------------------------------


eps_u = fe.interpolate(fe.Constant((U0,0,0)),W.sub(0).collapse())
eps_p = fe.interpolate(fe.Constant(1.01325e+05),W.sub(1).collapse())

#noSlip      = fe.Constant((0.0,0.0))
#pOut        = fe.Constant(0.0)
#inletFlow   = fe.Constant((U0, 0.0))

#inletBC     = fe.DirichletBC(W.sub(0), inletFlow, domainBoundaries, ID_INLET)
#outletBC    = fe.DirichletBC(W.sub(1), pOut, domainBoundaries, ID_OUTLET)
#topBC       = fe.DirichletBC(W.sub(0), noSlip, domainBoundaries, ID_TOP)
#botRBC      = fe.DirichletBC(W.sub(0), noSlip, domainBoundaries, ID_BOTTOMR)
#botLBC      = fe.DirichletBC(W.sub(0), noSlip, domainBoundaries, ID_BOTTOML)
#stepRBC     = fe.DirichletBC(W.sub(0), noSlip, domainBoundaries, ID_STEPR)
#stepLBC     = fe.DirichletBC(W.sub(0), noSlip, domainBoundaries, ID_STEPL)
#tepTopBC   = fe.DirichletBC(W.sub(0), noSlip, domainBoundaries, ID_STEPTOP)
bcs = []
#----------------------------
# Collect boundary conditions
#----------------------------

h = fe.CellDiameter(mesh)
#------------------------------------------------------
# Define body forces and some other relevant quantities
# n is the normals to the surfaces/edges of the domain
# I is the 2x2 identity matrix
#------------------------------------------------------
forc       = fe.Expression(('x[0]','x[1]','x[2]'),degree = 3)
n       = fe.FacetNormal(mesh)
I       = fe.Identity(2)
zero    = fe.Constant(0.0)

#------------------------------------------------------------------
# Define variational forms without time derivative in current time
# In theta-Galerkin formulation this is corresponding to the t_n+1
#
# TO BE COMPLETED DURING IN CLASS DEMO
#------------------------------------------------------------------
w = fe.Function(W)

(u,p) = (fe.split(w))
fe.assign(w.sub(0),eps_u)
fe.assign(w.sub(1),eps_p)
strain = 0.5*(fe.grad(u)+fe.grad(u).T)
strainmag = fe.sqrt(fe.inner(strain,strain))
b = 2*((0.17*h)**2)*strainmag*strain

T1_1 = fe.inner(v, fe.grad(u)*u)*fe.dx
T2_1 = mu*fe.inner(fe.grad(v), fe.grad(u))*fe.dx
T3_1 = p*fe.div(v)*fe.dx
T4_1 = q*fe.div(u)*fe.dx
T5_1 = fe.dot(v,fe.div(b))*fe.dx
T6_1 = fe.dot(v,(forc))*fe.dx
L_1 = T1_1 + T2_1 - T3_1 - T4_1 + T5_1 +T6_1

#------------------------------------------------------------------
# Define variational forms without time derivative in previous time
# In theta-Galerkin formulation this is corresponding to the t_n
#
# TO BE COMPLETED DURING IN CLASS DEMO
#------------------------------------------------------------------
w0 = fe.Function(W)

(u0, p0) = (fe.split(w0))
fe.assign(w0.sub(0),eps_u)
strain0 = 0.5*(fe.grad(u0)+fe.grad(u0).T)
strainmag0 = fe.sqrt(fe.inner(strain0,strain0))
b0 = 2*((0.17*h)**2)*strainmag0*strain0
T1_0 = fe.inner(v, fe.grad(u0)*u0)*fe.dx
T2_0 = mu*fe.inner(fe.grad(v), fe.grad(u0))*fe.dx
T3_0 = p*fe.div(v)*fe.dx # Since there is no pressure time derivative we can safely use p instead of p0
T4_0 = q*fe.div(u0)*fe.dx
T5_0 = fe.dot(v,fe.div(b0))*fe.dx
T6_0 = fe.dot(v,(forc))*fe.dx
L_0 = T1_0 + T2_0 - T3_0 - T4_0+ T5_0 +T6_0 
#------------------------------------------------
# Combine variational forms with time derivative
# As discussed for the theta-Galerkin formulation
#
#  dw/dt + F(t) = 0 is approximated as
#  (w-w0)/dt + (1-theta)*F(t0) + theta*F(t) = 0
#
# TO BE COMPLETED DURING IN CLASS DEMO
#-----------------------------------------------
F = idt*fe.inner((u-u0),v)*fe.dx + (1.0-theta)*L_0 + theta*L_1
#------------------------------------------
# Definition of the stabilization parameter
#
# TO BE COMPLETED DURING IN CLASS DEMO
#------------------------------------------
velocity = u0
vnorm = fe.sqrt(fe.dot(velocity,velocity))
tau = ((2*theta*idt)**2 + (2*vnorm/h)**2 + (4*mu/h**2)**2)**(-0.5)
#------------------------------------------------------------
# Residual of the strong form of Navier-Stokes and continuity
#
# TO BE COMPLETED DURING IN CLASS DEMO
#------------------------------------------------------------
r = idt*(u-u0) + \
theta*(fe.grad(u)*u - mu*fe.div(fe.grad(u)) + fe.grad(p) +fe.div(b) - forc) +\
(1-theta)*(fe.grad(u0)*u0 - mu*fe.div(fe.grad(u0)) + fe.grad(p) + fe.div(b0) - forc)
#-------------------------------------
# Add SUPG stabilization
#
# TO BE COMPLETED DURING IN CLASS DEMO
#-------------------------------------
F += tau*fe.inner(fe.grad(v)*u, r)*fe.dx(metadata={'quadrature_degree':4})
#--------------------------------------
# Add PSPG stabilization
#
# TO BE COMPLETED DURING IN CLASS DEMO
#-------------------------------------
F += -tau*fe.inner(fe.grad(q), r)*fe.dx(metadata={'quadrature_degree':4})
#----------------------------------------------------
# Define Jacobian or derivative for the Newton method
#
# TO BE COMPLETED DURING IN CLASS DEMO
#----------------------------------------------------
dW = fe.TrialFunction(W)
J = fe.derivative(F,w,dW)
#--------------------------------------
# Create variational problem and solver
#--------------------------------------
prm = fe.NonlinearVariationalProblem(F, w, bcs, J=J)
solver  = fe.NonlinearVariationalSolver(prm)
prm = solver.parameters
prm['newton_solver']['absolute_tolerance'] = 1E-8
prm['newton_solver']['relative_tolerance'] = 1E-7
prm['newton_solver']['maximum_iterations'] = 500
prm['newton_solver']['relaxation_parameter'] = 0.7


#----------------------------------
# Create files for storing solution
#----------------------------------
ufile = fe.File(outFileV)
pfile = fe.File(outFileP)
omegaFile = fe.File(outFileW)

#----------------------------------------------------
# Time-stepping loop
#
# DISCUSS THE CONCEPT OF AN INNNER AND AN OUTER LOOP
# INNER LOOP - NEWTON ITERATION LOOP AT EACH STEP
# OUTER LOOP - LOOP THROUGH THE TIME STEPS
#---------------------------------------------------
t   = dt
tn  = 0

Fx_listValues = []
Fy_listValues = []
T_listValues = []

while t < t_end:

    print("t =", t)

    #--------
    # Compute
    #--------
    #fe.assign(w.sub(0),eps_u)
    #fe.assign(w.sub(1),eps_p)
    #fe.assign(w0.sub(0),eps_u)
    print("Solving ....")
    solver.solve()

    #-------------------
    # Extract solutions:
    #-------------------
    (u, p) = w.split()

    #--------------------------------------------------------------
    # Save to file but only once every 4 steps
    # strategies like these reduce the amount of data files written
    # on to the computer memory
    #--------------------------------------------------------------
    if tn%4 == 0:

        u.rename("vel", "vel")
        p.rename("pres", "pres")
        ufile << u
        pfile << p
        print("Written Velocity And Pressure Data")

        #-------------------------------------------------------------------
        # Basic post-processing action: TASK 1:
        # We will compute the vorticity directly from within the solver code
        #
        # TO BE COMPLETED DURING IN CLASS DEMO
        #-------------------------------------------------------------------


    #----------------------------------------------------------------------
    # Basic post-processing action: TASK 2:
    # Compute the force on the step wall using integration of the traction
    #
    # TO BE COMPLETED DURING IN CLASS DEMO
    #----------------------------------------------------------------------

    #-----------------------
    # Move to next time step
    #-----------------------
    w0.assign(w)
    t   += dt
    tn  += 1

#---------------------------------------------------
# Plot and save the force data into a png file with
# appropriately formatted plotting commands
#
# TO BE COMPLETED DURING IN CLASS DEMO
#---------------------------------------------------
