import os
#import mshr
import numpy as np
from time import time
from fenics import *
from dolfin import *
import matplotlib.pyplot as plt
import os
from mpi4py import MPI
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from V0 import *
from galpha_U import *
#from mshr import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

xmin, xmax = 0, 1
ymin, ymax = 0, 1
zmin, zmax = 0, 1

pp0 = Point(0, 0)
pp1 = Point(1, 1)

File_name = 'Solution'

# Adaptive data
REFINE_RATIO = 0.3
Degree = 8
MAX_ITER = 10
REF_TYPE = 1

TIMINGVEC = np.zeros(MAX_ITER)
LevelStep = np.zeros(MAX_ITER)
L2vect = np.zeros(MAX_ITER)
L2qvect = np.zeros(MAX_ITER)
Evect = np.zeros(MAX_ITER)
Dofsvect = np.zeros(MAX_ITER)
hvect = np.zeros(MAX_ITER)

TrialTypeDir = os.path.join(File_name, 'Test')

if not os.path.isdir(TrialTypeDir): os.makedirs(TrialTypeDir)

epsilon = 0.5

element_degree = 1
time_step=0

# GA parameters
dt =1e-3
final_time=0.25
final_ts=500
ro=0.9
alphaf=1/(1+ro)
alpham=1/2*(3-ro)/(1+ro)
gamma=0.5+alpham-alphaf
eta=dt*gamma*alphaf
data_filename = XDMFFile('solution.xdmf')

for time_step in range(final_ts):
    time=(time_step)*dt
    mesh = RectangleMesh(pp0,pp1,1,1)


    for level in range(MAX_ITER):
        level_step = level

        # Define the necessary function spaces:
        # Discontinuous polynomial space for vector valued test function
        #t(TestType, mesh.ufl_cell(), Ptest)
        V1 = VectorElement( "DG", mesh.ufl_cell() ,element_degree, dim=2)

        # Discontinuous polynomial space for scalar valued test function
        V2 = FiniteElement( "DG", mesh.ufl_cell() ,element_degree)

        # Raviart-Thomas space for the flux (vector valued) trial function
        #V3 = FiniteElement( "RT", mesh.ufl_cell() , element_degree)
        V3 = VectorElement( "CG", mesh.ufl_cell() ,element_degree, dim=2)
        # Alternatively a continuous space if one wants

        # Continuous polynomial space for the base variable trial function
        V4 = FiniteElement( "CG", mesh.ufl_cell() ,element_degree)

        # The 'total' space is the product of all four functions spaces
        # defined above

        V = FunctionSpace(mesh, MixedElement(V1,V2,V3,V4))

        # Define the test and trial functions on the total space,
        # e.g. phi and w belong to the discontinuous vector valued space V1
        # and u and r belong to the space of continuous polynomials V4 etc.
        phi, psi, q, u = TrialFunctions(V)
        w, v, s, r = TestFunctions(V)

        # To compute the inner product we need the element diameter
        h = Circumradius(mesh)
        hmax = mesh.hmax()

        # n is the element normal vector needed to compute the boundary
        # integrals
        n = FacetNormal(mesh)
        nn = as_vector([n[0],n[1]])

        # Convection vector
        b = as_vector([Expression('-x[1]',degree = Degree,domain = mesh),Expression('x[0]',degree = Degree,domain = mesh)])

        # Analytical solution
        uexact = Expression('(t+0.1)*(1+tanh(0.5*(0.1-abs(0.5-sqrt(x[0]*x[0]+x[1]*x[1]))   ) )  )',t=time,degree = Degree,domain = mesh)

        gradu = as_vector([uexact.dx(0),uexact.dx(1)])
        # special solve at initial time
        if time_step==0:
            U_nn = Expression('(0.1)*(1+tanh(0.5*(0.1-abs(0.5-sqrt(x[0]*x[0]+x[1]*x[1]))   ) )  )',degree = Degree,domain = mesh)

            u_ = Expression('(t+0.1)*(1+tanh(0.5*(0.1-abs(0.5-sqrt(x[0]*x[0]+x[1]*x[1]))   ) )  )',t=0,degree = Degree-1,domain = mesh)
            u_t = Expression('(1+tanh(0.5*(0.1-abs(0.5-sqrt(x[0]*x[0]+x[1]*x[1]))   ) )  )',degree = Degree,domain = mesh)

            f= u_t + dot(b,nabla_grad(u_))-epsilon*div(grad(u_))
            VV,phi0,psi0,V,Ndofs,NdofsV,NdofsU,qq=V0(Degree,V,f,mesh,epsilon,b,u_t,U_nn)
            UU=U_nn


        if time_step>0:

            u_ = Expression('(t+0.1)*(1+tanh(0.5*(0.1-abs(0.5-sqrt(x[0]*x[0]+x[1]*x[1]))   ) )  )',t=time-(1-alphaf)*dt,degree = Degree,domain = mesh)
            u_t = Expression('(1+tanh(0.5*(0.1-abs(0.5-sqrt(x[0]*x[0]+x[1]*x[1]))   ) )  )',degree = Degree,domain = mesh)
            u_bc = Expression('(t+0.1)*(1+tanh(0.5*(0.1-abs(0.1-sqrt(x[0]*x[0]+x[1]*x[1]))   ) )  )',t=dt,degree = Degree,domain = mesh)

            f= u_t + dot(b,grad(u_))-epsilon*div(grad(u_))
            VV,phi0,psi0,V,Ndofs,UU,qq=galpha(Degree,V,f,mesh,epsilon,b,u_,U_nn,V_nn,Q_nn,dt,ro,time)




        # error in u
        e0r = UU - uexact
        q0r = qq - epsilon*gradu
        # compute error indicators
        PC = FunctionSpace(mesh,"DG", 0)
        c  = TestFunction(PC)

        gplot = Function(PC)

        ge = ( (phi0[0]*phi0[0] + phi0[1]*phi0[1])*c + h**2 *( phi0[0].dx(0)*phi0[0].dx(0) + phi0[0].dx(0)*phi0[1].dx(1) + phi0[1].dx(1)*phi0[0].dx(0) + phi0[1].dx(1)*phi0[1].dx(1) )*c + alpham*inner(psi0, psi0)*c+ eta*h**2*(psi0.dx(0)*psi0.dx(0)+psi0.dx(1)*psi0.dx(1))*c )*dx

        
        #only mass matrix at initial time
        if time_step==0:

           ge = ( (phi0[0]*phi0[0] + phi0[1]*phi0[1])*c +inner(psi0, psi0)*c)*dx


        g = assemble(ge)
        gplot.vector()[:]=g #if we want to plot this

        L2r = assemble(inner(e0r,e0r)*dx)
        L2q = assemble(inner(q0r,q0r)*dx)

        Ee = assemble(( phi0[0]*phi0[0] + phi0[1]*phi0[1] + h**2 *( phi0[0].dx(0)*phi0[0].dx(0) + phi0[0].dx(0)*phi0[1].dx(1) + phi0[1].dx(1)*phi0[0].dx(0) + phi0[1].dx(1)*phi0[1].dx(1) ) + alpham*inner(psi0, psi0)+eta* h**2*(psi0.dx(0)*psi0.dx(0)+psi0.dx(1)*psi0.dx(1)) )*dx)
        if time_step==0:

           Ee = assemble(( phi0[0]*phi0[0] + phi0[1]*phi0[1]+ inner(psi0, psi0) )*dx)


        E = sqrt(Ee)
        L2e = sqrt(L2r)
        L2eq = sqrt(L2q)
        LevelStep[level] = level_step
        L2vect[level] = L2e
        L2qvect[level] = L2eq
        Evect[level] = E
        Dofsvect[level] = Ndofs
        hvect[level] = hmax



        # meshark cells for refinement
        cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
        #Dorfler
        if REF_TYPE == 1:
            rg = np.sort(g)
            rg = rg[::-1]
            rgind = np.argsort(g)
            rgind = rgind[::-1]
            scum = 0.
            g0 = REFINE_RATIO**2*E**2
            Ntot = mesh.num_cells()
            for cellm in cells(mesh):
                if cellm.index() == rgind[0]:
                    break

            cell_markers[cellm] = True


            for nj in range(1,Ntot):
                scum += g[rgind[nj]]
                for cellm in cells(mesh):
                    if cellm.index() == rgind[nj]:
                        break
                cell_markers[cellm] = scum < g0
                if scum > g0:
                    break



        mesh = refine(mesh,cell_markers)

    plt.style.use('classic')

    t = dt*time_step

    U_nn =UU
    V_nn =VV
    Q_nn=qq



    #some plots
    if level > 0:
        fig, ax1  = plt.subplots()
        ax1.set_ylabel('Error Norm')
        ax1.set_xlabel('dofs')
        plt.loglog(Dofsvect[:level],L2vect[:level])
        plt.loglog(Dofsvect[:MAX_ITER],L2qvect[:MAX_ITER])
        plt.loglog(Dofsvect[:level], Evect[:level])
        ax1.legend(['L2 norm u', 'L2 norm q','Energy norm u,q'], loc='best')
        data_filename = os.path.join(TrialTypeDir, 'l2err%s.eps'%(level))
        fig.savefig(data_filename, format='pdf', transparent=True)
        plt.close()

    if time_step > 0:
        fig, ax1 = plt.subplots()
        cf = plot(U_nn)
        fig.colorbar(cf,ax=ax1)
        data_filename = os.path.join(TrialTypeDir, 'sol%s%s.png'%(level,time_step))
        fig.savefig(data_filename, format='png')
        plt.close()
        fig, ax1 = plt.subplots()
        cf = plot(mesh)
        data_filename = os.path.join(TrialTypeDir, 'mesh%s%s.png'%(level,time_step))
        fig.savefig(data_filename, format='png')
        plt.close()

        fig, ax1  = plt.subplots()

        ax1.set_ylabel('Error Norm')

        ax1.set_xlabel('dofs')

        plt.loglog(Dofsvect[:level],L2vect[:level])

        plt.loglog(Dofsvect[:MAX_ITER],L2qvect[:MAX_ITER])

        plt.loglog(Dofsvect[:level], Evect[:level])

        ax1.legend(['L2 norm u', 'L2 norm q','Energy norm u,q'], loc='best')

        data_filename = os.path.join(TrialTypeDir, 'l2err%s.pdf'%(time_step))

        fig.savefig(data_filename, format='pdf', transparent=True)

        plt.close()

        all_data = np.array(4,dtype=object)
        all_data = Dofsvect, L2vect, Evect,L2qvect
        data_filename = os.path.join(TrialTypeDir, 'data%s'%(time_step))
        np.save(data_filename, all_data)

    print("l2 error", L2vect[level])
    print("err rep error", E)

fig, ax1  = plt.subplots()
ax1.set_ylabel('Error Norm')
ax1.set_xlabel('dofs')
plt.loglog(Dofsvect[:MAX_ITER],L2vect[:MAX_ITER])
plt.loglog(Dofsvect[:MAX_ITER],L2qvect[:MAX_ITER])
plt.loglog(Dofsvect[:MAX_ITER],Evect[:MAX_ITER])
ax1.legend(['L2 norm u','L2 norm q', 'Energy norm u,q'], loc='best')
#ax1.legend(['L2 norm u'], loc='best')
data_filename = os.path.join(TrialTypeDir, 'l2err%s.pdf'%(MAX_ITER))
fig.savefig(data_filename, format='pdf', transparent=True)
plt.close()


data_filename = os.path.join(TrialTypeDir, 'L2_err_u_Tf.txt')
np.savetxt(data_filename, L2vect)


