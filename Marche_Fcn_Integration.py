import biorbd
from casadi import *
from Fcn_forward_dynamic import *


# Integration Euler
def int_Euler_swing(T, nbNoeuds, neuler, x, u):
    dn = T / nbNoeuds                                                        # Time step for shooting point
    dt = dn / neuler                                                         # Time step for iteration
    xj = x
    for j in range(neuler):
        dxj = ffcn_no_contact(xj, u)
        xj += dt * dxj                                                       # xj = x(t+dt)
    return xj

# Integration Runge Kutta 4
def int_RK4_swing(T, nbNoeuds, nkutta, x, u):
    dn = T / nbNoeuds                # Time step for shooting point
    dt = dn / nkutta                 # Time step for iteration
    xj = x
    for i in range(nkutta):
        k1 = ffcn_no_contact(xj, u)
        x2 = xj + (dt/2)*k1
        k2 = ffcn_no_contact(x2, u)
        x3 = xj + (dt/2)*k2
        k3 = ffcn_no_contact(x3, u)
        x4 = xj + dt*k3
        k4 = ffcn_no_contact(x4, u)

        xj += dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return xj

# Integration Runge Kutta 4
def int_RK4_stance(T, nbNoeuds, nkutta, x, u):
    dn = T / nbNoeuds                # Time step for shooting point
    dt = dn / nkutta                 # Time step for iteration
    xj = x
    for i in range(nkutta):
        k1 = ffcn_contact(xj, u)
        x2 = xj + (dt/2)*k1
        k2 = ffcn_contact(x2, u)
        x3 = xj + (dt/2)*k2
        k3 = ffcn_contact(x3, u)
        x4 = xj + dt*k3
        k4 = ffcn_contact(x4, u)

        xj += dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return xj