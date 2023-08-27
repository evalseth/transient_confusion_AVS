
import dolfin
from fenics import *

def galpha(Degree,V,f,M,epsilon,beta_vec,uexact,U_nn,V_nn,Q_nn,dt,ro,time):

    b=beta_vec
    alphaf=1/(1+ro)
    alpham=1/2*(3-ro)/(1+ro)
    gamma=0.5+alpham-alphaf
    eta=dt*gamma*alphaf

    phi, psi, q, u = TrialFunctions(V)
    w, v, s, r = TestFunctions(V)

    # To compute the inner product we need the element diameter
    h = Circumradius(M)
    hmax = M.hmax()

    # n is the element normal vector needed to compute the boundary
    # integrals
    n = FacetNormal(M)
    nn = as_vector([n[0],n[1]])

    SOL=Function(V)
    SOL2=Function(V)
    a1,a2,Q_n,U_n = SOL.split(True)
    a1,a2,a3,V_n = SOL2.split(True)

    U_n.interpolate(U_nn)
    V_n.interpolate(V_nn)
    Q_n.interpolate(Q_nn)

    innerp = ( w[0]*phi[0] + w[1]*phi[1]  + h**2 *( w[0].dx(0)*phi[0].dx(0) + w[0].dx(0)*phi[1].dx(1) + w[1].dx(1)*phi[0].dx(0) + w[1].dx(1)*phi[1].dx(1) ) + alpham/(alphaf*dt*gamma)*inner(v, psi) + h**2 *(v.dx(0)*psi.dx(0)+v.dx(1)*psi.dx(1)) )*dx

# Ultra Weak form

    def b(w,q,u,v,alphaf,eta):
        buv =( ( w[0]*q[0] + w[1]*q[1] + w[0].dx(0)*epsilon*u + w[1].dx(1)*epsilon*u
                 + v.dx(0)*q[0] + v.dx(1)*q[1]
                 - beta_vec[0]*v.dx(0)*u - beta_vec[1]*v.dx(1)*u   )*dx
        # Integral over the mesh skeleton
                 - (inner(dot(q('+'),nn('+')),v('+')-v('-')))*dS
                 + ( (beta_vec[0]*nn[0]('+')+beta_vec[1]*nn[1]('+'))*u('+')*(v('+')-v('-')) )*dS
                 - epsilon*( u('+')*((w[0]('+')-w[0]('-'))*nn[0]('+')+(w[1]('+')-w[1]('-'))*nn[1]('+') ) )*dS
        # Integral over the global boundary
                 - (inner(dot(q,nn),v))*ds +(beta_vec[0]*nn[0]+beta_vec[1]*nn[1])*u*v*ds - epsilon*( u*(w[0]*nn[0]+w[1]*nn[1]) )*ds )
        return buv+alpham/(alphaf*dt*gamma)*inner(u,v)*dx

    def b_RH(w,q,u,V,v,alphaf):

        return inner(alpham/gamma*V,v)*dx-inner(V,v)*dx+inner(alpham/(alphaf*dt*gamma)*u,v)*dx

    # Add all contributions to the LHS
    a=b(w,q,u,v,alphaf,eta)+b(phi,s,r,psi,alphaf,eta)+ innerp
    # Define the load functional




    #L = - inner(dot(epsilon*nabla_grad(uexact),n),v)*ds

    def boundary(x):
        return x[1] >1.0 - DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

    # Dirichlet BC is applied only on the space V4 ( on trial function u )

    bc = DirichletBC(V.sub(3), uexact,"on_boundary")

    # f=Constant(0)
    L = 1*inner(v, f)*dx
    L+=b_RH(w,Q_n,U_n,V_n,v,alphaf)


    # Define the solution
    sol0 = Function(V)
    print("dofs", sol0.vector().size())

    Ndofs = sol0.vector().size()



    # Call the solver
    solve(a==L, sol0,bcs=bc, solver_parameters = {'linear_solver' : 'mumps'})

    # Split the solution vector
    phi0, psi0, q0, u0 = sol0.split(True)
    SOL5=Function(V)
    SOL6=Function(V)
    a1,a2,a4,UU = SOL5.split(True)
    a1,a2,a4,VV = SOL6.split(True)

    Ndofs = sol0.vector().size()
    # UU = U_n.copy()
    UU.vector()[:] = (u0.vector()-U_n.vector())/alphaf+U_n.vector()


    # VV = U_n.copy()
    VV.vector()[:] =V_n.vector()+(-dt*V_n.vector()+(UU.vector()-U_n.vector()))/(dt*gamma)


    SOL7=Function(V)
    a1,a2,qq,a7 = SOL7.split(True)
    qq.vector()[:] =  q0.vector()

    SOL3=Function(V)
    SOL4=Function(V)
    a1,a2,a4,f_int = SOL3.split(True)
    SOL3=Function(V)
    a1,a2,a5,u_int = SOL4.split(True)

    return VV,phi0,psi0,V,Ndofs,UU,qq#,f_int,u_int
