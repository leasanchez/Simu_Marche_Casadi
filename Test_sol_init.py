import biorbd
from ezc3d import c3d
from casadi import *
from pylab import *
import numpy as np
import time
from LoadData import *
from Fcn_InitialGuess import *
from Marche_Fcn_Integration import *
from Fcn_Objective import *
from Fcn_Affichage import *

# SET MODELS
model_swing  = biorbd.Model('/home/leasanchez/programmation/Simu_Marche_Casadi/ModelesS2M/ANsWER_Rleg_6dof_17muscle_0contact.bioMod')
model_stance = biorbd.Model('/home/leasanchez/programmation/Simu_Marche_Casadi/ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact.bioMod')

# ----------------------------- Probleme -------------------------------------------------------------------------------
nbNoeuds_stance = 25                                   # shooting points for stance phase
nbNoeuds_swing  = 25                                   # shooting points for swing phase
nbNoeuds        = nbNoeuds_stance + nbNoeuds_swing     # total shooting points

nbMus     = model_stance.nbMuscleTotal()               # number of muscles
nbQ       = model_stance.nbDof()                       # number of DoFs
nbMarker  = model_stance.nbMarkers()                   # number of markers
nbBody    = model_stance.nbSegment()                   # number of body segments
nbContact = model_stance.nbContacts()                  # number of contact (2 forces --> plan)

nbU      = nbMus + 3                                   # number of controls : muscle activation + articular torque
nbX      = 2*nbQ                                       # number of states : generalized positions + velocities
nP       = nbMus + 1                                   # number of parameters : 1 global + muscles

# ----------------------------- Load Data ------------------------------------------------------------------------------
# LOAD MEASUREMENT DATA FROM C3D FILE
file        = '/home/leasanchez/programmation/Simu_Marche_Casadi/DonneesMouvement/equincocont01_out.c3d'
kalman_file = '/home/leasanchez/programmation/Simu_Marche_Casadi/DonneesMouvement/equincocont01_out_MOD5000_leftHanded_GenderF_Florent_.Q2'

# ground reaction forces
[GRF_real, T, T_stance] = load_data_GRF(file, nbNoeuds_stance, nbNoeuds_swing, 'cycle')
T_swing                 = T - T_stance                                                                # gait cycle time

# marker
M_real_stance = load_data_markers(file, T_stance, nbNoeuds_stance, nbMarker, 'stance')
M_real_swing  = load_data_markers(file, T_swing, nbNoeuds_swing, nbMarker, 'swing')

# muscular excitation
U_real_swing  = load_data_emg(file, T_swing, nbNoeuds_swing, nbMus, 'swing')
U_real_stance = load_data_emg(file, T_stance, nbNoeuds_stance, nbMus, 'stance')

# ----------------------------- Weighting factors ----------------------------------------------------------------------
wL  = 1                                                # activation
wMa = 30                                               # anatomical marker
wMt = 50                                               # technical marker
wU  = 1                                                # excitation
wR  = 0.05                                             # ground reaction

# ----------------------------- Movement -------------------------------------------------------------------------------
Ja = 0                                                 # objective function for muscle activation
Jm = 0                                                 # objective function for markers
Je = 0                                                 # objective function for EMG
JR = 0                                                 # objective function for ground reactions

# ----------------------------- Initial guess --------------------------------------------------------------------------
init_A = 0.1

# CONTROL
u0               = np.zeros((nbU, nbNoeuds))
a0               = load_initialguess_muscularExcitation(np.hstack([U_real_stance, U_real_swing]))
f0               = [0]*nbNoeuds
f1               = [-500]*nbNoeuds_stance + [0]*nbNoeuds_swing
f2               = [0]*nbNoeuds

u0[: nbMus, :]   = a0
# u0[: nbMus, :]   = np.zeros((nbMus, nbNoeuds)) + init_A
u0[nbMus + 0, :] = f0
u0[nbMus + 1, :] = f1
u0[nbMus + 2, :] = f2
u0               = vertcat(*u0.T)

# STATE
q0_stance = load_initialguess_q(file, kalman_file, T_stance, nbNoeuds_stance, 'stance')
q0_swing  = load_initialguess_q(file, kalman_file, T_swing, nbNoeuds_swing, 'swing')
q0        = np.hstack([q0_stance, q0_swing])
dq0       = np.zeros((nbQ, (nbNoeuds + 1)))

X0                = np.zeros((nbX, (nbNoeuds + 1)))
X0[:nbQ, :]       = q0
X0[nbQ: 2*nbQ, :] = dq0
X0                = vertcat(*X0.T)

GRF = []
# ------------ PHASE 1 : Stance phase
for k in range(nbNoeuds_stance):
    Uk = u0[nbU*k : nbU*(k + 1)]
    Xk = X0[nbX*k: nbX*(k + 1)]

    Q           = np.array(Xk[:nbQ]).squeeze()  # states
    dQ          = np.array(Xk[nbQ:2 * nbQ]).squeeze()
    activations = Uk[: nbMus]  # controls
    F           = np.array(Uk[nbMus :])


    # DYNAMIQUE
    states = biorbd.VecBiorbdMuscleStateDynamics(nbMus)
    n_muscle = 0
    for state in states:
        state.setActivation(double(activations[n_muscle]))  # Set muscles activations
        n_muscle += 1

    torque    = model_stance.muscularJointTorque(states, Q, dQ).to_array()
    torque[0] = F[0]
    torque[1] = F[1]
    torque[2] = F[2]

    C    = model_stance.getConstraints()
    ddQ  = model_stance.ForwardDynamicsConstraintsDirect(Q, dQ, torque, C)
    GRFk = C.getForce().to_array()
    GRF.append(GRFk)

    # OBJECTIVE FUNCTION
    JR += wR * ((GRFk[0] - GRF_real[1, k]) * (GRFk[0] - GRF_real[1, k])) + wR * ((GRFk[2] - GRF_real[2, k]) * (GRFk[2] - GRF_real[2, k]))
    Jm += fcn_objective_markers(wMa, wMt, Q, M_real_stance[:, :, k], 'stance')
    Je += fcn_objective_emg(wU, Uk, U_real_stance[:, k])
    Ja += fcn_objective_activation(wL, Uk)


# ------------ PHASE 2 : Swing phase
for k in range(nbNoeuds_swing):
    # CONTROL AND STATES
    Uk = u0[nbU*nbNoeuds_stance + nbU*k: nbU * nbNoeuds_stance + nbU*(k + 1)]
    Xk = X0[nbX*nbNoeuds_stance + nbX * k: nbX*nbNoeuds_stance + nbX * (k + 1)]

    Q           = Xk[:nbQ]  # states
    dQ          = Xk[nbQ:2 * nbQ]
    activations = Uk[: nbMus]  # controls
    F           = Uk[nbMus :]

    # OBJECTIVE FUNCTION
    Jm += fcn_objective_markers(wMa, wMt, Q, M_real_swing[:, :, k], 'swing')
    Je += fcn_objective_emg(wU, Uk, U_real_swing[:, k])
    Ja += fcn_objective_activation(wL, Uk)

# ----------------------------- Visualization --------------------------------------------------------------------------
# GROUND REACTION FORCES
GRF = np.vstack([GRF])

plt.figure(1)
plt.subplot(211)
plt.title('Ground reactions forces A/P')
t_stance = np.linspace(0, T_stance, nbNoeuds_stance)
t_swing = t_stance[-1] + np.linspace(0, T_swing, nbNoeuds_swing)
t = np.hstack([t_stance, t_swing])

plt.plot(t, GRF_real[1, :-1], 'b-', alpha=0.5, label = 'real')
plt.plot(t_stance, GRF[:, 0], 'r+-', label = 'simu')
plt.legend()

plt.subplot(212)
plt.title('Ground reactions forces vertical')
plt.plot(t, GRF_real[2, :-1], 'b-', alpha=0.5, label = 'real')
plt.plot(t_stance, GRF[:, 2], 'r+-', label = 'simu')
plt.legend()


# JOINT POSITIONS AND VELOCITIES
plt.figure(3)
Labels_X = ['Pelvis_X', 'Pelvis_Y', 'Pelvis_Z', 'Hip', 'Knee', 'Ankle']
tq = np.hstack([t, t[-1] + (t[-1]-t[-2])])

for q in range(nbQ):
    plt.subplot(2, 6, q + 1)
    plt.title('Q ' + Labels_X[q])
    plt.plot(tq, q0[q, :]*180/np.pi)
    plt.xlabel('time [s]')
    if q == 0:
        plt.ylabel('q [°]')

    plt.subplot(2, 6, q + 1 + nbQ)
    plt.title('dQ ' + Labels_X[q])
    plt.plot(tq, dq0[q, :]*180/np.pi)
    plt.xlabel('time [s]')
    if q == 0:
        plt.ylabel('dq [°/s]')

# MUSCULAR ACTIVATIONS
plt.figure(2)
U_real = np.hstack([U_real_stance, U_real_swing])
Labels = ['GLUT_MAX1', 'GLUT_MAX2', 'GLUT_MAX3', 'GLUT_MED1', 'GLUT_MED2', 'GLUT_MED3', 'R_SEMIMEM', 'R_SEMITEN', 'R_BI_FEM_LH', 'R_RECTUS_FEM', 'R_VAS_MED', 'R_VAS_INT', 'R_VAS_LAT', 'R_GAS_MED', 'R_GAS_LAT', 'R_SOLEUS', 'R_TIB_ANT']
nMus_emg = 9

for nMus in range(nbMus):
    plt.subplot(5, 4, nMus + 1)
    plt.title(Labels[nMus])
    plt.plot([0, t[-1]], [0, 0], 'k--')  # lower bound
    plt.plot([0, t[-1]], [1, 1], 'k--')  # upper bound

    if nMus == 1 or nMus == 2 or nMus == 3 or nMus == 5 or nMus == 6 or nMus == 11 or nMus == 12:
        plt.plot(t, a0[nMus, :], 'r+')
    else:
        plt.plot(t, a0[nMus, :], 'r+')
        plt.plot(t, U_real[nMus_emg, :], 'b-', alpha=0.5)
        nMus_emg -= 1


plt.subplot(5, 4, nbMus + 1)
plt.title('Pelvis Tx')
plt.plot(t, f0, 'b+')

plt.subplot(5, 4, nbMus + 2)
plt.title('Pelvis Ty')
plt.plot(t, f1, 'b+')


plt.subplot(5, 4, nbMus + 3)
plt.title('Pelvis Rz')
plt.plot(t, f2, 'b+')


# OBJECTIVE FUNCTION VALUES
J = Ja + Je + Jm + JR
print('Global                 : ' + str(J))
print('activation             : ' + str(Ja))
print('emg                    : ' + str(Je))
print('marker                 : ' + str(Jm))
print('ground reaction forces : ' + str(JR))

plt.show()