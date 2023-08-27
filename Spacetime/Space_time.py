import os
import numpy as np
import time
import dolfin
from fenics import *
from dolfin import *
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


parameters['ghost_mode']='shared_facet'

if has_linear_algebra_backend("Epetra"):
        parameters["linear_algebra_backend"] = "Epetra"
        
  
# Mesh
N = 1
p0 = Point(0.0, 0.0, 0.0)
p1 = Point(1.0,1.0,0.25)
M = BoxMesh(p0,p1, N, N, N)
xmin, xmax = 0, 1
ymin, ymax = 0, 1
zmin, zmax = p0[2], p1[2]


File_name = 'Test'
# Adaptive data
REFINE_RATIO = 0.3
Degree = 5
MAX_ITER = 10
REF_TYPE = 1
degreee = 1

trialdeg = degreee
testdeg = trialdeg

if REF_TYPE ==3:
    TrialTypeDir = os.path.join(File_name, 'Uniform_AVS_Ptrial' + str(trialdeg)+ '_Ptest'+ str(testdeg))

if REF_TYPE == 1:
        TrialTypeDir = os.path.join(File_name, 'Adapt_AVS_Ptrial' + str(trialdeg)+ '_Ptest'+ str(testdeg))
        
#define arrays
LevelStep = np.zeros(MAX_ITER)
L2vect = np.zeros(MAX_ITER)
L2vectRate = np.zeros(MAX_ITER)
L2vectQ = np.zeros(MAX_ITER)
L2vectRateQ = np.zeros(MAX_ITER)
L2vectTone = np.zeros(MAX_ITER)
L2vectTone = np.zeros(MAX_ITER)
L2vectRateTone = np.zeros(MAX_ITER)
L2vectRateToneQ = np.zeros(MAX_ITER)
L2vectQTone = np.zeros(MAX_ITER)
L2vectRateQTone = np.zeros(MAX_ITER)
H1vect = np.zeros(MAX_ITER)
H1vectRate = np.zeros(MAX_ITER)
H1vectQ = np.zeros(MAX_ITER)
H1vectRateQ = np.zeros(MAX_ITER)
Evect = np.zeros(MAX_ITER)
Evect_rate = np.zeros(MAX_ITER)
Dofsvect = np.zeros(MAX_ITER)
hvect = np.zeros(MAX_ITER)
TIMINGVEC = np.zeros(MAX_ITER)
Evect_tone = np.zeros(MAX_ITER)
Evect_toneRate = np.zeros(MAX_ITER)


if not os.path.isdir(TrialTypeDir): os.makedirs(TrialTypeDir)




for level in range(MAX_ITER):
    level_step = level
    
    if level == 0:
       tstart = time.time()
       
    print("Generating function spaces at step %d" %level)

    # Define the necessary function spaces:
    # Discontinuous polynomial space for vector valued test function
    #t(TestType, mesh.ufl_cell(), Ptest)
    V1 = VectorElement( "DG", M.ufl_cell() ,testdeg, dim=2)

    # Discontinuous polynomial space for scalar valued test function
    V2 = FiniteElement( "DG", M.ufl_cell() ,testdeg)

    # Raviart-Thomas space for the flux (vector valued) trial function
    #V3 = FiniteElement( "RT", M.ufl_cell() , 1)
    V3 = VectorElement( "CG", M.ufl_cell() ,trialdeg, dim=2)
    # Alternatively a continuous space if one wants

    # Continuous polynomial space for the base variable trial function
    V4 = FiniteElement( "CG", M.ufl_cell() ,trialdeg  )

    # The 'total' space is the product of all four functions spaces
    # defined above

    V = FunctionSpace(M, MixedElement(V1,V2,V3,V4))
    # Define the test and trial functions on the total space,
    # e.g. phi and w belong to the discontinuous vector valued space V1
    # and u and r belong to the space of continuous polynomials V4 etc.
    phi, psi, q, u = TrialFunctions(V)
    w, v, s, r = TestFunctions(V)
    print("finished function spaces at step %d" %level)
    # To compute the inner product we need the element diameter
    h = Circumradius(M)
    hmax = M.hmax()
    
    x, y, z = SpatialCoordinate(M)

    # n is the element normal vector needed to compute the boundary
    # integrals
    n = FacetNormal(M)
    nn = as_vector([n[0],n[1]])

    # Convection vector b

    # Convection vector b = Constant((1,0,-1))
    b = as_vector([1,0,1])

    #normal vector in left edge
    nx = Constant((-1,0,0))

    # Diffusion coefficient
    epsilon = 0.5
    
    #time exponent 'l'
    ell = 2.0



    # Analytical solution
    q_initial = Constant((0,0))
    b = Expression(('-x[1]', 'x[0]','1'), degree=Degree, domain=M)

    uexact = Expression('(x[2]+0.1)*(1+tanh(0.5*(0.1-abs(0.5-sqrt(x[0]*x[0]+x[1]*x[1]))   ) )  )' ,degree = Degree,domain = M)

    gradu = as_vector([uexact.dx(0),uexact.dx(1),0.0])
    uexact_gradient= as_vector([uexact.dx(0),uexact.dx(1)])
    initial=uexact
    uinitial = uexact

    # Source term f
    f = -epsilon*div(gradu) + dot(nabla_grad(uexact),b)

    innerp = ( w[0]*phi[0] + w[1]*phi[1] + inner(v, psi)  + h**2 *( v.dx(0)*psi.dx(0) + v.dx(1)*psi.dx(1) +( w[0].dx(0)+w[1].dx(1) )*( phi[0].dx(0)+phi[1].dx(1) )  ) )*dx

    # Bilinear form acting on the trial functions (u,q) and the
    # optimal test functions (v,w)
    #Here we have integrated back to the equivalent "strong form"
    buv =( ( w[0]*q[0] + w[1]*q[1] - w[0]*epsilon*u.dx(0) - w[1]*epsilon*u.dx(1)
             - v*q[0].dx(0) - v*q[1].dx(1)
             + b[0]*v*u.dx(0) + b[1]*v*u.dx(1) + b[2]*v*u.dx(2)  )*dx )

    # Bilinear form acting on the functions (r,s) and the error
    # representation functions (psi, phi)
    brs =( ( phi[0]*s[0] + phi[1]*s[1] - phi[0]*epsilon*r.dx(0) - phi[1]*epsilon*r.dx(1)
             - psi*s[0].dx(0) - psi*s[1].dx(1)
             + b[0]*psi*r.dx(0) + b[1]*psi*r.dx(1) + b[2]*psi*r.dx(2)  )*dx )


    # Add all contributions to the LHS
    a = buv + brs + innerp

    # Define the load functional
    L = 1*inner(v, f)*dx

    def Space_boundary(x, on_boundary):
        return on_boundary and (near(x[0], xmin, 1e-14) or near(x[0], xmax, 1e-14) or near(x[1], ymin, 1e-14) or near(x[1], ymax, 1e-14) )
        
    def Time_boundary(x, on_boundary):
        return on_boundary and (near(x[2], 0, 1e-14))

    # Dirichlet BC is applied only on the space V4 ( on trial function u )
    bc = DirichletBC(V.sub(3), uexact,Space_boundary)
    ic = DirichletBC(V.sub(3), uexact, Time_boundary)

    # Define the solution
    sol0 = Function(V)
    print("total dofs", sol0.vector().size())
    
    print("trial elems ", M.num_cells())

    #Ndofs = dumbsol.vector().size()
    Ndofs = len(V.sub(3).dofmap().dofs()) + len(V.sub(2).dofmap().dofs())
    print("trial dofs", Ndofs)

    # Call the solver
    solve(a==L, sol0, bcs=[bc,ic], solver_parameters = {'linear_solver' : 'mumps'})


    # Split the solution vector
    phi0, psi0, q0, u0 = sol0.split(True)

    e0r = u0 - uexact
    eqr = q0 - epsilon*uexact_gradient
   

    # compute error indicators from the DPG mixed variable e
    PC = FunctionSpace(M,"DG", 0)
    c  = TestFunction(PC)             # p.w constant fn

    gplot = Function(PC)
    # ge = e_disc*e_disc*c*dx
    ge = ( (phi0[0]*phi0[0] + phi0[1]*phi0[1])*c+ inner(psi0, psi0)*c + h**2 *( psi0.dx(0)*psi0.dx(0)*c + psi0.dx(1)*psi0.dx(1)*c +(phi0[0].dx(0)+phi0[1].dx(1))*(phi0[0].dx(0)+phi0[1].dx(1))*c  )  )*dx

    g = assemble(ge)
    gplot.vector()[:]=g

    L2r = assemble(inner(e0r,e0r)*dx)
    L2Qr = assemble(inner(eqr,eqr)*dx)
    Gamma_Tone = conditional(z > zmax-1e-14, 1,0)
    L2r_Tone = assemble(Gamma_Tone*inner(e0r,e0r)*ds) 
    L2Qr_Tone = assemble(Gamma_Tone*inner(eqr,eqr)*ds)
    PC2 = FunctionSpace(mesh,"DG", 0)

    # Ee = assemble(e_disc*e_disc*dx)
    Ee = assemble(( phi0[0]*phi0[0] + phi0[1]*phi0[1] + inner(psi0, psi0)+  h**2 *( psi0.dx(0)*psi0.dx(0) + psi0.dx(1)*psi0.dx(1) +(phi0[0].dx(0)+phi0[1].dx(1))*(phi0[0].dx(0)+phi0[1].dx(1))  ) )*dx)
    
    Ee_tone = assemble(Gamma_Tone*( phi0[0]*phi0[0] + phi0[1]*phi0[1] + inner(psi0, psi0)+  h**2 *( psi0.dx(0)*psi0.dx(0) + psi0.dx(1)*psi0.dx(1) +(phi0[0].dx(0)+phi0[1].dx(1))*(phi0[0].dx(0)+phi0[1].dx(1))  ) )*ds)
    
    E = sqrt(Ee)
    E_tone = sqrt(Ee_tone)
    L2e = sqrt(L2r)
    L2Qe = sqrt(L2Qr)
    L2e_Tone = sqrt(L2r_Tone)
    L2Qe_Tone = sqrt(L2Qr_Tone)

    LevelStep[level] = level_step
    L2vect[level] = L2e
    L2vectQ[level] = L2Qe
    Evect[level] = E
    Evect_tone[level] = E_tone
    Dofsvect[level] = Ndofs
    hvect[level] = hmax
    L2vectTone[level] = L2e_Tone
    L2vectQTone[level] = L2Qe_Tone

    # Mark cells for refinement
    cell_markers = MeshFunction("bool", M, M.topology().dim())
    if REF_TYPE == 1:
        rg = np.sort(g)
        rg = rg[::-1]
        rgind = np.argsort(g)
        rgind = rgind[::-1]
        scum = 0.
        g0 = REFINE_RATIO**2*E**2
        Ntot = M.num_cells()
        for cellm in cells(M):
            if cellm.index() == rgind[0]:
                break

        cell_markers[cellm] = True


        for nj in range(1,Ntot):
            scum += g[rgind[nj]]
            for cellm in cells(M):
                if cellm.index() == rgind[nj]:
                    break
            cell_markers[cellm] = scum < g0
            if scum > g0:
                break


        # Refine mesh

    if REF_TYPE == 1:
        M = refine(M,cell_markers)
    if REF_TYPE == 3:
        M = refine(M)
        
    plt.style.use('classic')
    file = File("sol_u0_%s.pvd"%(level));
    file << u0;
    file2 = File("sol_ex_%s.pvd"%(level));
    file2 << interpolate(uexact,FunctionSpace(M,"CG", 1));



    #some plots
    if level > 0:
        fig, ax1  = plt.subplots()
        ax1.set_ylabel('Error Norm')
        ax1.set_xlabel('dofs')
        plt.loglog(Dofsvect[:level],L2vect[:level])
        plt.loglog(Dofsvect[:level], L2vectQ[:level])
        ax1.legend(['$||u-u^h||_{L^2(\Omega_T)}$', '$||\mathbf{q}-\mathbf{q}^h||_{L^2(\Omega_T)}$'], loc='best')
        data_filename = os.path.join(TrialTypeDir, 'l2err%s.pdf'%(level))
        fig.savefig(data_filename, format='pdf', transparent=True)
        plt.close()
        
        fig, ax1  = plt.subplots()
        ax1.set_ylabel('Error Norm')
        ax1.set_xlabel('dofs')
        plt.loglog(Dofsvect[:level],Evect[:level])
        ax1.legend(['Energy norm'], loc='best')
        data_filename = os.path.join(TrialTypeDir, 'Energy_err%s.pdf'%(level))
        fig.savefig(data_filename, format='pdf', transparent=True)
        plt.close()


    

    print("l2 error u ", L2vect[level])
    print("l2 error q ", L2vectQ[level])
    print("l2 error u T one ", L2e_Tone)
    print("l2 error Q T one ", L2Qe_Tone)
    print("Energy error", E)
    print("Energy error T one", E_tone)
    if level > 0 and REF_TYPE ==3:
        L2vectRate[level] =ln(L2vect[level]/L2vect[level-1])/ln(hvect[level]/hvect[level-1])
        print("l2 rate u", L2vectRate[level])
        L2vectRateTone[level] =ln(L2vectTone[level]/L2vectTone[level-1])/ln(hvect[level]/hvect[level-1])
        print("l2 rate u Tone", L2vectRateTone[level])
        L2vectRateQTone[level] =ln(L2vectQTone[level]/L2vectQTone[level-1])/ln(hvect[level]/hvect[level-1])
        print("l2 rate Q Tone", L2vectRateQTone[level])
        L2vectRateQ[level] =ln(L2vectQ[level]/L2vectQ[level-1])/ln(hvect[level]/hvect[level-1])
        print("l2 rate Q", L2vectRateQ[level])
        Evect_rate[level] =ln(Evect[level]/Evect[level-1])/ln(hvect[level]/hvect[level-1])
        print("Energy rate ", Evect_rate[level])
        Evect_toneRate[level] =ln(Evect_tone[level]/Evect_tone[level-1])/ln(hvect[level]/hvect[level-1])
        print("Energy rate T one", Evect_toneRate[level])


    print("Success at step %d" %level)
    
    
fig, ax1  = plt.subplots()
ax1.set_ylabel('Error Norm')
ax1.set_xlabel('dofs')
plt.loglog(Dofsvect[:MAX_ITER],L2vect[:MAX_ITER])
plt.loglog(Dofsvect[:MAX_ITER],L2vectQ[:MAX_ITER])
ax1.legend(['$||u-u^h||_{L^2(\Omega_T)}$', '$||\mathbf{q}-\mathbf{q}^h||_{L^2(\Omega_T)}$'], loc='best')
data_filename = os.path.join(TrialTypeDir, 'l2err_final.pdf')
fig.savefig(data_filename, format='pdf', transparent=True)
plt.close()

fig, ax1  = plt.subplots()
ax1.set_ylabel('Error Norm')
ax1.set_xlabel('dofs')
plt.loglog(Dofsvect[:MAX_ITER],L2vectTone[:MAX_ITER])
plt.loglog(Dofsvect[:MAX_ITER],L2vectQTone[:MAX_ITER])
ax1.legend(['$||u-u^h||_{L^2(\Omega_T)}$', '$||\mathbf{q}-\mathbf{q}^h||_{L^2(\Omega_T)}$'], loc='best')
data_filename = os.path.join(TrialTypeDir, 'l2err_finaltime.pdf')
fig.savefig(data_filename, format='pdf', transparent=True)
plt.close()

fig, ax1  = plt.subplots()
ax1.set_ylabel('Error Norm')
ax1.set_xlabel('dofs')
plt.loglog(Dofsvect[:MAX_ITER],L2vectTone[:MAX_ITER])
ax1.legend(['$||u-u^h||_{L^2(\Omega_T)}$',], loc='best')
data_filename = os.path.join(TrialTypeDir, 'l2err_finaltime-U.pdf')
fig.savefig(data_filename, format='pdf', transparent=True)
plt.close()

fig, ax1  = plt.subplots()
ax1.set_ylabel('Error Norm')
ax1.set_xlabel('dofs')
plt.loglog(Dofsvect[:MAX_ITER],Evect[:MAX_ITER])
ax1.legend(['Energy norm'], loc='best')
data_filename = os.path.join(TrialTypeDir, 'Energy_err_final.pdf')
fig.savefig(data_filename, format='pdf', transparent=True)
plt.close()
        



data_filename = os.path.join(TrialTypeDir, 'dofs.txt')
np.savetxt(data_filename, Dofsvect)
data_filename = os.path.join(TrialTypeDir, 'L2_err_u.txt')
np.savetxt(data_filename, L2vect)
data_filename = os.path.join(TrialTypeDir, 'L2_rate_u.txt')
np.savetxt(data_filename, L2vectRate)
data_filename = os.path.join(TrialTypeDir, 'L2_err_u_Tf.txt')
np.savetxt(data_filename, L2vectTone)
data_filename = os.path.join(TrialTypeDir, 'L2_err_q_Tf.txt')
np.savetxt(data_filename, L2vectQTone)
data_filename = os.path.join(TrialTypeDir, 'L2_rate_q.txt')
np.savetxt(data_filename, L2vectRateQ)
data_filename = os.path.join(TrialTypeDir, 'L2_err_q.txt')
np.savetxt(data_filename, L2vectQ)
data_filename = os.path.join(TrialTypeDir, 'L2_rate_u_Tf.txt')
np.savetxt(data_filename, L2vectRateTone)
data_filename = os.path.join(TrialTypeDir, 'L2_rate_q_Tf.txt')
np.savetxt(data_filename, L2vectRateQTone)
data_filename = os.path.join(TrialTypeDir, 'Energy_error.txt')
np.savetxt(data_filename, Evect)
data_filename = os.path.join(TrialTypeDir, 'Energy_error_Tf.txt')
np.savetxt(data_filename, Evect_tone)
data_filename = os.path.join(TrialTypeDir, 'elem_diam.txt')
np.savetxt(data_filename, hvect)



tstop = time.time()
#if rank == 0:
print("\n total time [s] :", tstop-tstart)

TIMINGVEC[0] = tstop-tstart


data_filename = os.path.join(TrialTypeDir, 'Timing.txt')
np.savetxt(data_filename, TIMINGVEC)
