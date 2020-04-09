import numpy as np
import biorbd
from casadi import DM, Function, MX

import Fcn_plot_states_controls as psu
import LoadData
from Fcn_Objective import Fcn_Objective
from Fcn_forward_dynamic import Dynamics

from Define_parameters import Parameters
from Marche_Fcn_Integration import int_RK4

# ----------------------------- Load Results from txt file -------------------------------------------------------------
file = '/home/leasanchez/programmation/Simu_Marche_Casadi/Resultats/equincocont01/RES/Swing/equincocont01_sol_swing.txt'
f = open(file, 'r')
content = f.read()
content_divide = content.split('\n')

params = Parameters()

# FIND STATE
# s = np.zeros((params.nbX, (params.nbNoeuds_swing + 1)))
X = DM.zeros(params.nbX, (params.nbNoeuds_swing + 1))
idx_init = 2
for n in range(params.nbNoeuds_swing + 1):
    idx = idx_init + params.nbX * n
    for x in range(params.nbX):
        a = content_divide[idx + x]
        X[x, n] = float(a)

# FIND CONTROL
# u = np.zeros((params.nbU, (params.nbNoeuds_swing)))
U = DM.zeros(params.nbU, (params.nbNoeuds_swing))
idx_u = idx_init + (params.nbNoeuds_swing + 1) * params.nbX + 4
for n in range(params.nbNoeuds_swing):
    idx = idx_u + params.nbU * n
    for u_id in range(params.nbU):
        a = content_divide[idx + u_id]
        U[u_id, n] = float(a)

# FIND PARAMETERS
# p = np.zeros((params.nP))
P = DM.zeros(params.nP)
idx_p = idx_init + (params.nbNoeuds_swing + 1) * params.nbX + 4 + params.nbNoeuds_swing * params.nbU + 4
for p_id in range(params.nP):
    idx = idx_p + p_id
    P[p_id] = float(content_divide[idx])
f.close()

# ----------------------------- Load Data from c3d file ----------------------------------------------------------------
# Swing
[GRF_real_swing, params.T, params.T_stance, params.T_swing] = LoadData.load_data_GRF(params, 'swing')                   # GROUND REACTION FORCES & SET TIME
M_real_swing  = LoadData.load_data_markers(params, 'swing')                                                             # MARKERS POSITION
U_real_swing  = LoadData.load_data_emg(params, 'swing')                                                                 # MUSCULAR EXCITATION
# Stance
[GRF_real_stance, params.T, params.T_stance, params.T_swing] = LoadData.load_data_GRF(params, 'stance')
M_real_stance = LoadData.load_data_markers(params, 'stance')
U_real_stance = LoadData.load_data_emg(params, 'stance')

# ----------------------------- States & controls ----------------------------------------------------------------------
# CONTROL
u  = MX.sym("u", params.nbU)
activation  = u[:params.nbMus]                                # muscular activation
torque      = u[params.nbMus:]                                # residual joint torque

# PARAMETERS
p  = MX.sym("p", params.nP)                                   # maximal isometric force adjustment

# STATE
x  = MX.sym("x", params.nbX)
q  = x[:params.nbQ]                                           # generalized coordinates
dq = x[params.nbQ: 2 * params.nbQ]                            # velocities

# ----------------------------- Define casadi function -----------------------------------------------------------------
ffcn_contact = Function("ffcn_contact",
                                [x, u, p],
                                [Dynamics.ffcn_contact(x, u, p)],
                                ["states", "controls", "parameters"],
                                ["statesdot"]).expand()

compute_GRF = Function("compute_GRF",
                              [x, u, p],
                              [Dynamics.compute_GRF(x, u, p)],
                              ["states", "controls", "parameters"],
                              ["GRF"]).expand()

ffcn_no_contact = Function("ffcn_no_contact",
                                   [x, u, p],
                                   [Dynamics.ffcn_no_contact(x, u, p)],
                                   ["states", "controls", "parameters"],
                                   ["statesdot"]).expand()

markers = Function("markers",
                   [x],
                   [params.model_stance.markers(x[:params.nbQ])],
                   ["states"],
                   ["markers"]).expand()

def RK_intervall(fun, params, x, u, p):
    nkutta = params.nkutta
    if fun.name() == 'ffcn_contact':
        T = params.T_stance
        nbNoeuds = params.nbNoeuds_stance
    else:
        T = params.T_swing
        nbNoeuds = params.nbNoeuds_swing

    dn  = T / nbNoeuds
    dt1 = dn / 3
    dt = dt1/nkutta
    xj = x
    for i in range(nkutta):
        k1 = fun(xj, u, p)
        x2 = xj + (dt / 2) * k1
        k2 = fun(x2, u, p)
        x3 = xj + (dt / 2) * k2
        k3 = fun(x3, u, p)
        x4 = xj + dt * k3
        k4 = fun(x4, u, p)

        xj += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return xj

# ----------------------------- Dynamics -------------------------------------------------------------------------------
Ja = 0                                                 # objective function for muscle activation
Jm = 0                                                 # objective function for markers
Je = 0                                                 # objective function for EMG
JR = 0                                                 # objective function for ground reactions
Jt = 0
constraints = 0
GRF = DM.zeros(2, params.nbNoeuds_stance + 1)
M = []
x_int_interval = DM.zeros(params.nbX, (params.nbNoeuds_swing + 1) * 3)

# ------------ PHASE 1 : Stance phase
for k in range(params.nbNoeuds_stance):
    # DYNAMIQUE
    Uk = U[:, k]
    Xk = X[:, k]

    # OBJECTIVE FUNCTION
    Mk= markers(Xk)
    M.append(Mk)
    Ja += Fcn_Objective.fcn_objective_activation(params.wL, Uk)
    Jt += Fcn_Objective.fcn_objective_residualtorque(params.wt, Uk[params.nbMus:])

    if file.__contains__('stance'):
        GRF[:, k] = compute_GRF(Xk, Uk, P)
        JR += Fcn_Objective.fcn_objective_GRF_casadi(params.wR, GRF[:, k], GRF_real_stance[:, k])
        Jm += Fcn_Objective.fcn_objective_markers_casadi(params.model_stance, params.wMa, params.wMt, Mk, M_real_stance[:, :, k])
        Je += Fcn_Objective.fcn_objective_emg(params.wU, Uk, U_real_stance[:, k])

        X_int = int_RK4(ffcn_contact, params, Xk, Uk, P)
        constraints += X[:, k + 1] - X_int
    else:
        Jm += Fcn_Objective.fcn_objective_markers_casadi(params.model_swing, params.wMa, params.wMt, Mk, M_real_swing[:, :, k])
        Je += Fcn_Objective.fcn_objective_emg(params.wU, Uk, U_real_swing[:, k])

        X_int = int_RK4(ffcn_no_contact, params, Xk, Uk, P)
        constraints += X[:, k + 1] - X_int
    #
    # # INTEGRATION
    # x_int_interval[:, k * 3] = Xk
    # for i in range(3):
    #     x_int_interval[:, (k * 3) + i + 1] = int_RK4(ffcn_contact, params, x_int_interval[:, (k * 3) + i], Uk, P)

# ----------------------------- Visualize states and controls ----------------------------------------------------------
if file.__contains__('stance'):
    psu.plot_q(np.array(X[:params.nbQ, :]), params.T_stance, params.nbNoeuds_stance)
    psu.plot_dq(np.array(X[params.nbQ:, :]), params.T_stance, params.nbNoeuds_stance)
    psu.plot_activation(params, np.array(U), U_real_stance[:, :-1], params.T_stance, params.nbNoeuds_stance)
    psu.plot_torque(np.array(U[params.nbMus:, :]), params.T_stance, params.nbNoeuds_stance)
    psu.plot_GRF(np.array(GRF), GRF_real_stance, params.T_stance, params.nbNoeuds_stance)
    psu.plot_markers(params.nbNoeuds_stance, M, M_real_stance)

    print('Global                 : ' + str(Jm + Ja + Je + Jt + JR))
    print('activation             : ' + str(Ja))
    print('emg                    : ' + str(Je))
    print('marker                 : ' + str(Jm))
    print('ground reaction forces : ' + str(JR))
    print('residual torques       : ' + str(Jt))

else:
    psu.plot_q(np.array(X[:params.nbQ, :]), params.T_swing, params.nbNoeuds_swing)
    psu.plot_dq(np.array(X[params.nbQ:, :]), params.T_swing, params.nbNoeuds_swing)
    psu.plot_activation(params, np.array(U), U_real_swing[:, :-1], params.T_swing, params.nbNoeuds_swing)
    psu.plot_torque(np.array(U[params.nbMus:, :]), params.T_swing, params.nbNoeuds_swing)
    psu.plot_markers(params.nbNoeuds_swing, M, M_real_swing)

    print('Global                 : ' + str(Jm + Ja + Je + Jt))
    print('activation             : ' + str(Ja))
    print('emg                    : ' + str(Je))
    print('marker                 : ' + str(Jm))
    print('residual torques       : ' + str(Jt))

print('0')

