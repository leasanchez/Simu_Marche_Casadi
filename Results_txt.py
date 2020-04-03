import numpy as np
import biorbd

import Fcn_plot_states_controls as psu
import LoadData
import Fcn_Objective as fo

from Define_parameters import Parameters

# ----------------------------- Load Results from txt file -------------------------------------------------------------
file = '/home/leasanchez/programmation/Simu_Marche_Casadi/Resultats/equincocont01/RES/Swing/equincocont01_sol_swing.txt'
f = open(file, 'r')
content = f.read()
content_divide = content.split('\n')

params = Parameters()

# FIND STATE
s = np.zeros((params.nbX, (params.nbNoeuds_swing + 1)))
idx_init = 2
for n in range(params.nbNoeuds_swing + 1):
    idx = idx_init + params.nbX * n
    for x in range(params.nbX):
        a = content_divide[idx + x]
        s[x, n] = float(a)

# FIND CONTROL
u = np.zeros((params.nbU, (params.nbNoeuds_swing)))
idx_u = idx_init + (params.nbNoeuds_stance + 1) * params.nbX + 4
for n in range(params.nbNoeuds_stance):
    idx = idx_u + params.nbU * n
    for u_id in range(params.nbU):
        a = content_divide[idx + u_id]
        u[u_id, n] = float(a)

# FIND PARAMETERS
p = np.zeros((params.nP))
idx_p = idx_init + (params.nbNoeuds_stance + 1) * params.nbX + 4 + params.nbNoeuds_stance * 20 + 4
for p_id in range(params.nP):
    idx = idx_p + p_id
    p[p_id] = float(content_divide[idx])

# ----------------------------- Load Data from c3d file ----------------------------------------------------------------
# Swing
[GRF_real_swing, params.T, params.T_stance, params.T_swing] = LoadData.load_data_GRF(params, 'swing')                   # GROUND REACTION FORCES & SET TIME
M_real_swing  = LoadData.load_data_markers(params, 'swing')                                                             # MARKERS POSITION
U_real_swing  = LoadData.load_data_emg(params, 'swing')                                                                 # MUSCULAR EXCITATION
# Stance
[GRF_real, params.T, params.T_stance, params.T_swing] = LoadData.load_data_GRF(params, 'cycle')
M_real_stance = LoadData.load_data_markers(params, 'stance')
U_real_stance = LoadData.load_data_emg(params, 'stance')

# ----------------------------- Dynamics -------------------------------------------------------------------------------
Ja = 0                                                 # objective function for muscle activation
Jm = 0                                                 # objective function for markers
Je = 0                                                 # objective function for EMG
JR = 0                                                 # objective function for ground reactions
constraints = 0
q_int = np.zeros((params.nbQ, 5, params.nbNoeuds_stance))
dq_int = np.zeros((params.nbQ, 5, params.nbNoeuds_stance))
torque = np.zeros((params.nbQ, params.nbNoeuds_stance))
GRF = np.zeros((3, params.nbNoeuds_stance + 1))

# SET ISOMETRIC FORCES
model = params.model_stance
n_muscle = 0
for nGrp in range(model.nbMuscleGroups()):
    for nMus in range(model.muscleGroup(nGrp).nbMuscles()):
        fiso = model.muscleGroup(nGrp).muscle(nMus).characteristics().forceIsoMax()
        model.muscleGroup(nGrp).muscle(nMus).characteristics().setForceIsoMax(p[n_muscle + 1] * fiso)

def fcn_dyn_contact(xk, uk):
    Q  = xk[:params.nbQ]
    dQ = xk[params.nbQ:]
    act = uk[:params.nbMus].squeeze()
    states = biorbd.VecBiorbdMuscleStateDynamics(model.nbMuscleTotal())
    n_muscle = 0
    for state in states:
        state.setActivation(act[n_muscle])
        n_muscle += 1

    joint_torque    = model.muscularJointTorque(states, Q, dQ).to_array()
    joint_torque[0] = uk[params.nbMus + 0]  # ajout des forces au pelvis
    joint_torque[1] = uk[params.nbMus + 1]
    joint_torque[2] = uk[params.nbMus + 2]

    ddQ = model.ForwardDynamicsConstraintsDirect(Q, dQ, joint_torque)
    return np.hstack([dQ, ddQ.to_array()])

def Get_torque(xk, uk):
    Q  = xk[:params.nbQ]
    dQ = xk[params.nbQ:]
    act = uk[:params.nbMus].squeeze()
    states = biorbd.VecBiorbdMuscleStateDynamics(model.nbMuscleTotal())
    n_muscle = 0
    for state in states:
        state.setActivation(act[n_muscle])
        n_muscle += 1

    joint_torque    = model.muscularJointTorque(states, Q, dQ).to_array()
    joint_torque[0] = uk[params.nbMus + 0]  # ajout des forces au pelvis
    joint_torque[1] = uk[params.nbMus + 1]
    joint_torque[2] = uk[params.nbMus + 2]

    return joint_torque

def int_RK4(fcn, x, u):
    dn = params.T_stance / params.nbNoeuds_stance                 # Time step for shooting point
    dn2 = dn / 5
    dt = dn2 / 5                                                   # Time step for iteration
    xj = x
    for i in range(5):
        k1 = fcn(xj, u)
        x2 = xj + (dt/2)*k1
        k2 = fcn(x2, u)
        x3 = xj + (dt/2)*k2
        k3 = fcn(x3, u)
        x4 = xj + dt*k3
        k4 = fcn(x4, u)

        xj += dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return xj

for k in range(params.nbNoeuds_stance):
    # DYNAMIQUE
    Uk = u[:, k]
    Xk = s[:, k]

    # Xk_int = int_RK4(fcn_dyn_contact, Xk, Uk)
    # q_int[:, k] = Xk_int[:params.nbQ]
    # dq_int[:, k] = Xk_int[params.nbQ:]

    # plot integrale
    q_int[:, 0, k] = Xk[:params.nbQ]
    dq_int[:, 0, k] = Xk[params.nbQ:]

    x_int = np.zeros((params.nbX, 5))
    x_int[:, 0] = np.hstack([q_int[:, 0, k], dq_int[:, 0, k]])
    u_int = Uk
    for i in range(4):
        x_int[:, i + 1] = int_RK4(fcn_dyn_contact, x_int[:, i], u_int)
        q_int[:, i + 1, k] = x_int[:params.nbQ, i + 1]
        dq_int[:, i + 1, k] = x_int[params.nbQ:, i + 1]

    # OBJECTIVE FUNCTION
    torque[:, k] = Get_torque(Xk, Uk)
    [grf, Jr] = fo.fcn_objective_GRF(params.wR, Xk, Uk, GRF_real[:, k])                                                    # tracking ground reaction --> stance
    GRF[:, k] = np.array(grf).squeeze()
    JR += Jr
    Jm += fo.fcn_objective_markers(params.wMa, params.wMt, Xk[: params.nbQ], M_real_stance[:, :, k], 'stance')             # tracking marker
    Je += fo.fcn_objective_emg(params.wU, Uk, U_real_stance[:, k])                                                         # tracking emg
    Ja += fo.fcn_objective_activation(params.wL, Uk)                                                                       # min muscle activations (no EMG)


# ----------------------------- Visualize states and controls ----------------------------------------------------------
# psu.plot_q_int(s[:params.nbQ, :], q_int, params.T_stance, params.nbNoeuds_stance)
# psu.plot_q(s[:params.nbQ, :], params.T_stance, params.nbNoeuds_stance)
psu.plot_q(s[:params.nbQ, :], params.T_swing, params.nbNoeuds_swing)
psu.plot_dq(s[params.nbQ:, :], params.T_swing, params.nbNoeuds_swing)
U_real_swing = U_real_swing[:, :-1]
psu.plot_control(u, U_real_swing, params.T_swing, params.nbNoeuds_swing)

psu.plot_pelvis_force(u[params.nbMus:, :], params.T_stance, params.nbNoeuds_stance)
psu.plot_GRF(GRF, GRF_real, params.T_stance, params.nbNoeuds_stance)
psu.plot_torque(torque, u[params.nbMus:, :], params.T_stance, params.nbNoeuds_stance)
psu.plot_markers_result(s[:params.nbQ, :], params.T_swing, params.nbNoeuds_swing, params.nbMarker, M_real_swing)