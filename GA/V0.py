from dolfin import *
from fenics import *

def V0(Degree,V,f,M,epsilon,beta_vec,uexact,U_n):

    phi, psi, q, u = TrialFunctions(V)
    w, v, s, r = TestFunctions(V)

    # To compute the inner product we need the element diameter
    h = Circumradius(M)
    hmax = M.hmax()

    # n is the element normal vector needed to compute the boundary
    # integrals
    n = FacetNormal(M)
    nn = as_vector([n[0],n[1]])


    # Inner product on V of the "optimal" test functions (v,w) and the error
    # representation functions (psi, phi) note only mass matrix at t0
    innerp = ( w[0]*phi[0] + w[1]*phi[1]  +inner(v,psi))*dx

    # Bilinear form acting on the trial functions (u,q) and the
    # optimal test functions (v,w)

    def b(w,q,u,v):
        buv =( ( w[0]*q[0]+ w[1]*q[1]
                 + v.dx(0)*q[0]+ v.dx(1)*q[1])*dx
        # Integral over the mesh skeleton
                 - (inner(dot(q('+'),nn('+')),v('+')-v('-')))*dS
        # Integral over the global boundary
                 -(inner(dot(q,nn),v))*ds+(beta_vec[0]*nn[0]+beta_vec[1]*nn[1])*u*v*ds - epsilon*( u*(w[0]*nn[0]+w[1]*nn[1]) )*ds  )
        return buv+inner(u,v)*dx


    a=-b(w,q,u,v)+b(phi,s,r,psi)+ innerp


    def b_RH(w,u,v):
        buv =( + w[0]*epsilon*u.dx(0) +w[1]*epsilon*u.dx(1)
                 - beta_vec[0]*v*u.dx(0) -beta_vec[1]*v*u.dx(1))*dx
        return buv
    # Define the load functional

    L = -1*inner(v, f)*dx
    L-=b_RH(w,U_n,v)

    def boundary(x):
        return x[1] >1.0 - DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

    # Dirichlet BC is applied only on the space V4 ( on trial function u )
    bc = DirichletBC(V.sub(3), uexact,"on_boundary")


    # Define the solution
    sol0 = Function(V)
    print("dofs", sol0.vector().size())

    Ndofs = sol0.vector().size()
    PETScOptions.set("mat_mumps_icntl_24", 1)
    PETScOptions.set("mat_mumps_icntl_14", 50.0)
    # Call the solver
    solve(a==L, sol0,bcs=bc, solver_parameters = {'linear_solver' : 'mumps'})

    # Split the solution vector
    phi0, psi0, q0, u0 = sol0.split(True)

    Ndofs=sol0.vector().size()

    return u0,phi0,psi0,V,Ndofs,Ndofs,Ndofs,q0
