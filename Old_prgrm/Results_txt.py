import numpy as np
import biorbd
from casadi import DM, Function, MX

import Fcn_plot_states_controls as psu
import LoadData
from Fcn_Objective import Fcn_Objective
from Fcn_forward_dynamic import Dynamics

from Define_parameters import Parameters
from Marche_Fcn_Integration import int_RK4
from Read_Muscod import Muscod

params = Parameters()
muscod = Muscod(params.name_subject)

# ----------------------------- Load Data from c3d file ----------------------------------------------------------------
# Swing
[GRF_real_swing, params.T, params.T_stance, params.T_swing] = LoadData.load_data_GRF(params, 'swing')                   # GROUND REACTION FORCES & SET TIME
M_real_swing = LoadData.load_data_markers(params, 'swing')                                                             # MARKERS POSITION
U_real_swing = LoadData.load_data_emg(params, 'swing')                                                                 # MUSCULAR EXCITATION
# Stance
[GRF_real_stance, params.T, params.T_stance, params.T_swing] = LoadData.load_data_GRF(params, 'stance')
M_real_stance = LoadData.load_data_markers(params, 'stance')
U_real_stance = LoadData.load_data_emg(params, 'stance')

# ----------------------------- Load Results from txt file -------------------------------------------------------------
file = params.save_dir + 'Gait/' + params.name_subject + '_gait_cv_2.txt'
f = open(file, 'r')
content = f.read()
content_divide = content.split('\n')

if file.__contains__('stance'):
    nbNoeuds = params.nbNoeuds_stance
    T = params.T_stance
elif file.__contains__('swing'):
    nbNoeuds = params.nbNoeuds_swing
    T = params.T_swing
elif file.__contains__('impact'):
    nbNoeuds = params.nbNoeuds + 1
    T = params.T
else:
    nbNoeuds = params.nbNoeuds
    T = params.T

# FIND STATE
X = DM.zeros(params.nbX, (nbNoeuds + 1))
idx_init = 2
for n in range(nbNoeuds + 1):
    idx = idx_init + params.nbX * n
    for x in range(params.nbX):
        a = content_divide[idx + x]
        X[x, n] = float(a)

# FIND CONTROL
U = DM.zeros(params.nbU, (nbNoeuds))
idx_u = idx_init + (nbNoeuds + 1) * params.nbX + 4
for n in range(nbNoeuds):
    idx = idx_u + params.nbU * n
    for u_id in range(params.nbU):
        a = content_divide[idx + u_id]
        U[u_id, n] = float(a)

# FIND PARAMETERS
P = DM.zeros(params.nP)
idx_p = idx_init + (nbNoeuds + 1) * params.nbX + 4 + nbNoeuds * params.nbU + 4
for p_id in range(params.nP):
    idx = idx_p + p_id
    P[p_id] = float(content_divide[idx])
f.close()

# ----------------------------- States & controls ----------------------------------------------------------------------
# CONTROL
u = MX.sym("u", params.nbU)
activation = u[:params.nbMus]                                # muscular activation
torque = u[params.nbMus:]                                # residual joint torque

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

ffcn_impact = Function("ffcn_impact",
                                   [x, u, p],
                                   [Dynamics.ffcn_impact(x, u, p)],
                                   ["states", "controls", "parameters"],
                                   ["statesdot"]).expand()

markers = Function("markers",
                   [x],
                   [params.model_stance.markers(x[:params.nbQ])],
                   ["states"],
                   ["markers"]).expand()

CoM = Function("CoM",
               [x],
               [params.model_stance.CoM(x[:params.nbQ]).to_mx()],
               ["states"],
               ["CoM"]).expand()

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
COM = []
x_int_interval = DM.zeros(params.nbX, (params.nbNoeuds_swing + 1) * 3)

for k in range(nbNoeuds):
    # DYNAMIQUE
    Uk = U[:, k]
    Xk = X[:, k]

    # OBJECTIVE FUNCTION
    Mk= markers(Xk)
    CoMk = CoM(Xk)
    M.append(Mk)
    COM.append(CoMk)
    Ja += Fcn_Objective.fcn_objective_activation(params.wL, Uk)
    Jt += Fcn_Objective.fcn_objective_residualtorque(params.wt, Uk[params.nbMus:])

    if file.__contains__('stance'):
        GRF[:, k] = compute_GRF(Xk, Uk, P)
        JR += Fcn_Objective.fcn_objective_GRF_casadi(params.wR, GRF[:, k], GRF_real_stance[:, k])
        Jm += Fcn_Objective.fcn_objective_markers_casadi_maxfoot(params.model_stance, params.wMa, params.wMt, Mk, M_real_stance[:, :, k])
        Je += Fcn_Objective.fcn_objective_emg(params.wU, Uk, U_real_stance[:, k])

        X_int = int_RK4(ffcn_contact, params, Xk, Uk, P)
        constraints += X[:, k + 1] - X_int
    elif file.__contains__('swing'):
        Jm += Fcn_Objective.fcn_objective_markers_casadi(params.model_swing, params.wMa, params.wMt, Mk, M_real_swing[:, :, k])
        Je += Fcn_Objective.fcn_objective_emg(params.wU, Uk, U_real_swing[:, k])

        X_int = int_RK4(ffcn_no_contact, params, Xk, Uk, P)
        constraints += X[:, k + 1] - X_int

    elif file.__contains__('impact'):
        if k < params.nbNoeuds_stance:
            GRF[:, k] = compute_GRF(Xk, Uk, P)
            JR += Fcn_Objective.fcn_objective_GRF_casadi(params.wR, GRF[:, k], GRF_real_stance[:, k])
            Jm += Fcn_Objective.fcn_objective_markers_casadi_maxfoot(params.model_stance, params.wMa, params.wMt, Mk,M_real_stance[:, :, k])
            Je += Fcn_Objective.fcn_objective_emg(params.wU, Uk, U_real_stance[:, k])

            X_int = int_RK4(ffcn_contact, params, Xk, Uk, P)
            constraints += X[:, k + 1] - X_int
        elif (k < (params.nbNoeuds_stance + params.nbNoeuds_swing)):
            Jm += Fcn_Objective.fcn_objective_markers_casadi(params.model_swing, params.wMa, params.wMt, Mk, M_real_swing[:, :, (k - params.nbNoeuds_stance)])
            Je += Fcn_Objective.fcn_objective_emg(params.wU, Uk, U_real_swing[:, (k - params.nbNoeuds_stance)])

            X_int = int_RK4(ffcn_no_contact, params, Xk, Uk, P)
            constraints += X[:, k + 1] - X_int
        else:
            Jm += Fcn_Objective.fcn_objective_markers_casadi(params.model_stance, params.wMa, params.wMt, Mk, M_real_swing[:, :, -1])
            Je += Fcn_Objective.fcn_objective_emg(params.wU, Uk, U_real_swing[:, -1])

            X_int = int_RK4(ffcn_impact, params, Xk, Uk, P)
            constraints += X[:, k + 1] - X_int

    else:
        if k < params.nbNoeuds_stance:
            GRF[:, k] = compute_GRF(Xk, Uk, P)
            JR += Fcn_Objective.fcn_objective_GRF_casadi(params.wR, GRF[:, k], GRF_real_stance[:, k])
            Jm += Fcn_Objective.fcn_objective_markers_casadi_maxfoot(params.model_stance, params.wMa, params.wMt, Mk,M_real_stance[:, :, k])
            Je += Fcn_Objective.fcn_objective_emg(params.wU, Uk, U_real_stance[:, k])

            X_int = int_RK4(ffcn_contact, params, Xk, Uk, P)
            constraints += X[:, k + 1] - X_int
        else:
            Jm += Fcn_Objective.fcn_objective_markers_casadi(params.model_swing, params.wMa, params.wMt, Mk, M_real_swing[:, :, (k - params.nbNoeuds_stance)])
            Je += Fcn_Objective.fcn_objective_emg(params.wU, Uk, U_real_swing[:, (k - params.nbNoeuds_stance)])

            X_int = int_RK4(ffcn_no_contact, params, Xk, Uk, P)
            constraints += X[:, k + 1] - X_int

    # # INTEGRATION
    # x_int_interval[:, k * 3] = Xk
    # for i in range(3):
    #     x_int_interval[:, (k * 3) + i + 1] = int_RK4(ffcn_contact, params, x_int_interval[:, (k * 3) + i], Uk, P)

# ----------------------------- Visualize states and controls ----------------------------------------------------------
print('Global                 : ' + str(Jm + Ja + Je + Jt + JR))
print('activation             : ' + str(Ja))
print('emg                    : ' + str(Je))
print('marker                 : ' + str(Jm))
print('ground reaction forces : ' + str(JR))
print('residual torques       : ' + str(Jt))

if file.__contains__('gait'):
    if file.__contains__('impact'):
        impact = True
        U_real = np.hstack([U_real_stance[:, :-1], U_real_swing[:, :]])
    else:
        impact = False
        U_real = np.hstack([U_real_stance[:, :-1], U_real_swing[:, :-1]])
    # psu.plot_q(np.array(X[:params.nbQ, :]), T, nbNoeuds, gait=True, impact=impact, params=params)
    psu.plot_q_muscod(np.array(X[:params.nbQ, :]), params, muscod, Gaitphase='gait')
    psu.plot_dq(np.array(X[params.nbQ:, :]), T, nbNoeuds, gait=True, impact=impact, params=params)
    psu.plot_torque(np.array(U[params.nbMus:, :]), T, nbNoeuds, gait=True, impact=impact, params=params)
    psu.plot_GRF(np.array(GRF[:, :params.nbNoeuds_stance + 1]), GRF_real_stance, params.T_stance, params.nbNoeuds_stance)
    # psu.plot_activation(params, np.array(U), U_real, T, nbNoeuds, gait=True, impact=impact)
    psu.plot_activation_muscod(params, np.array(U), U_real, muscod, Gaitphase='gait', impact=impact)

    M_real          = np.zeros((3, params.nbMarker, (params.nbNoeuds + 1)))
    M_real[0, :, :] = np.hstack([M_real_stance[0, :, :-1], M_real_swing[0, :, :]])
    M_real[1, :, :] = np.hstack([M_real_stance[1, :, :-1], M_real_swing[1, :, :]])
    M_real[2, :, :] = np.hstack([M_real_stance[2, :, :-1], M_real_swing[2, :, :]])
    psu.plot_markers(nbNoeuds, M, M_real, COM)
else:
    if file.__contains__('stance'):
        psu.plot_q_muscod(np.array(X[:params.nbQ, :]), params, muscod, Gaitphase='stance')
        psu.plot_dq(np.array(X[params.nbQ:, :]), T, nbNoeuds)
        psu.plot_torque(np.array(U[params.nbMus:, :]), T, nbNoeuds)
        psu.plot_activation_muscod(params, np.array(U), U_real_stance[:, :-1], muscod, Gaitphase='stance')
        psu.plot_GRF(np.array(GRF), GRF_real_stance, T, nbNoeuds)
        psu.plot_markers(nbNoeuds, M, M_real_stance, COM)
    else:
        psu.plot_q_muscod(np.array(X[:params.nbQ, :]), params, muscod, Gaitphase='swing')
        psu.plot_dq(np.array(X[params.nbQ:, :]), T, nbNoeuds)
        psu.plot_torque(np.array(U[params.nbMus:, :]), T, nbNoeuds)
        psu.plot_activation_muscod(params, np.array(U), U_real_swing[:, :-1], muscod, Gaitphase='swing')
        psu.plot_markers(nbNoeuds, M, M_real_swing, COM)

print('0')

