import biorbd
from ezc3d import c3d
from casadi import *
from pylab import *
import numpy as np
from Marche_Fcn_Integration import *
from Fcn_Objective import *
from LoadData import *
from Fcn_Affichage import *
from Fcn_InitialGuess import *


# # fonctions
# Forward_Dynamics_SansContact = external('libforward_dynamics_no_contact', 'libforward_dynamics_no_contact.so', {'enable_fd': True})  # sans contact
# Markers                      = external('libmarkers', 'libmarkers.so',{'enable_fd':True})
# SetForceISO                  = external('libforce_iso_max', 'libforce_iso_max.so',{'enable_fd':True})

# Swing
model     = biorbd.Model('/home/leasanchez/programmation/Marche_Florent/ModelesS2M/ANsWER_Rleg_6dof_17muscle_0contact.bioMod')
GaitPhase = 'swing'

# ----------------------------- Probleme -------------------------------------------------------------------------------
nbNoeuds  = 25
nbMuscle  = model.nbMuscleTotal()
nbQ       = model.nbDof()
nbMarker  = model.nbMarkers()
nbBody    = model.nbSegment()
nbContact = model.nbContacts()
nbU       = nbMuscle + 3
nbX       = 2*nbQ + nbMuscle
nP        = nbMuscle + 1


nkutta   = 5

# ----------------------------- Scaling term ---------------------------------------------------------------------------
wL  = 1                                               # activation
wMa = 30                                              # anatomical marker
wMt = 50                                              # technical marker
wU  = 1                                               # excitation
wR  = 1                                               # ground reaction

# ----------------------------- Load Data ------------------------------------------------------------------------------
# LOAD MEASUREMENT DATA FROM C3D FILE
file        = '/home/leasanchez/programmation/Marche_Florent/DonneesMouvement/normal04_out.c3d'
kalman_file = '/home/leasanchez/programmation/Marche_Florent/DonneesMouvement/equinus15_out_MOD5000_leftHanded_GenderF_Florent_.Q2'

[GRF_real, T_stance, T_swing, idx_TO, idx_HS] = load_data_GRF(file, 0, nbNoeuds, GaitPhase)                                      # ground reaction forces
T                                             = round(T_stance + T_swing, 2)                                                     # gait cycle time

M_real                                        = load_data_markers(file, T_swing, idx_TO, idx_HS, nbNoeuds, nbMarker, GaitPhase)  # markers position
U_real                                        = load_data_emg(file, T_swing, idx_TO, idx_HS, nbNoeuds, nbMuscle, GaitPhase)      # muscular excitation

# VISUALIZATION INPUT
affichage_emg_real(U_real, T_swing)
affichage_markers_real(M_real)
affichage_GRF_real(GRF_real, T_stance, T_swing)

# EXTRACT INITIAL MAXIMAL ISOMETRIC FORCES FROM THE MODEL
FISO0 = np.zeros(nbMuscle)
n_muscle = 0
for nGrp in range(model.nbMuscleGroups()):
    for nMus in range(model.muscleGroup(nGrp).nbMuscles()):
        FISO0[n_muscle] = model.muscleGroup(nGrp).muscle(nMus).characteristics().forceIsoMax()
        n_muscle += 1


# ----------------------------- Etats et control -----------------------------------------------------------------------
# CONTROL
u = MX.sym("u", nbU)
e = u[:nbMuscle]                                      # excitations musculaires
F = u[nbMuscle:]                                      # 3 Forces pelvis

# PARAMETERS
p = MX.sym("p", nP)                                   # force isometrique adjustment

# STATE
x  = MX.sym("x", nbX)
q  = x[:nbQ]                                          # position généralisée
dq = x[nbQ: 2*nbQ]                                    # velocities
a  = x[2*nbQ:]                                        # activation musculaire



# ----------------------------- Movement -------------------------------------------------------------------------------
U = MX.sym("U", nbU*nbNoeuds)                         # control
X = MX.sym("X", nbX*(nbNoeuds+1))                     # state
G = []                                                # liste des contraintes 'égalité'
Ja = 0                                                # Objective function for muscle activation
Jm = 0                                                # Objective function for markers
Je = 0                                                # Objective function for EMG
JR = 0                                                # Objective function for ground reactions

# FIND THE PARAMETERS P OPTIMISING THE MAXIMUM ISOMETRIC FORCES
n_muscle = 0
for nGrp in range(model.nbMuscleGroups()):
    for nMus in range(model.muscleGroup(nGrp).nbMuscles()):
        SetForceISO(nGrp, nMus, p[0] * p[nMus + 1] * FISO0[n_muscle])
        n_muscle += 1


for k in range(nbNoeuds):
    # DYNAMIQUE
    Uk = U[nbU*k: nbU*(k + 1)]
    G.append(X[nbX*(k + 1): nbX*(k + 2)] - int_RK4_swing(T_swing, nbNoeuds, nkutta,  X[nbX*k: nbX*(k + 1)], Uk))

    # OBJECTIVE FUNCTION
    # Tracking
    # JR +=  fcn_objective_GRF(wR, X[nbX*k: nbX*(k + 1)], Uk, k, GRF_real[:, k])         # Ground Reaction
    Jm += fcn_objective_markers(wMa, wMt, X[nbX*k: nbX*k + nbQ], M_real[:, :, k])        # Markers
    Je += fcn_objective_emg(wU, Uk, U_real[:, k])                                        # Emg

    # Muscle activations (no EMG)
    u_no_emg = [Uk[1], Uk[2], Uk[3], Uk[5], Uk[6], Uk[11], Uk[12]]
    Ja      += fcn_objective_activation(wL, u_no_emg)


# ----------------------------- Contraintes ----------------------------------------------------------------------------
lbg = [0]*nbX*nbNoeuds               # uniquement les contraintes egalites
ubg = [0]*nbX*nbNoeuds

# ----------------------------- Bounds on w ----------------------------------------------------------------------------
# activation - excitation musculaire
min_A = 1e-3                        # 0 exclusif
max_A = 1

# parameters
min_pg = 0.5
min_p  = 0.8
max_p  = 2

# forces
min_F = -1
max_F = 1

lbu = ([min_A]*nbMuscle + [min_F]*3)*nbNoeuds
ubu = ([max_A]*nbMuscle + [max_F]*3)*nbNoeuds

# q et dq
min_Q = -50
max_Q = 50
lbX = ([min_Q]*2*nbQ + [min_A]*nbMuscle)*(nbNoeuds + 1)
ubX = ([max_Q]*2*nbQ + [max_A]*nbMuscle)*(nbNoeuds + 1)

lbx = vertcat(lbu, lbX)
ubx = vertcat(ubu, ubX)

# ----------------------------- Initial guess --------------------------------------------------------------------------
init_A = 0.1
u0 = ([init_A]*nbMuscle + [0] + [0.3] + [-0.1])*nbNoeuds                           # ou u0 = load_initialguess_u(U_real, nbNoeuds, nbMus)
X0 = ([0]*nbQ + [0]*nbQ + [init_A]*nbMuscle)*(nbNoeuds + 1)

x0 = vertcat(u0, X0)

# ----------------------------- Solver ---------------------------------------------------------------------------------
w = vertcat(U, X)
J = Ja + Je + Jm + JR
# J = 0

nlp = {'x': w, 'p': p, 'f': J, 'g': vertcat(*G)}
opts = {"ipopt.tol": 1e-4, "ipopt.linear_solver": "ma57", "ipopt.hessian_approximation":"limited-memory"}
solver = nlpsol("solver", "ipopt", nlp, opts)
res = solver(lbg = lbg,
             ubg = ubg,
             lbx = lbx,
             ubx = ubx,
             x0  = x0)


# RESULTS
sol_U  = res["x"][:nbU * nbNoeuds]
sol_X  = res["x"][nbU * nbNoeuds:]
sol_q  = 0
sol_dq = 0
sol_a  = 0

for k in range(nbNoeuds +1):
    xk = sol_X[nbX*k: nbX*(k + 1)]
    sol_q  += xk[:nbQ]
    sol_dq += xk[nbQ:2*nbQ]
    sol_a  += xk[2*nbQ:]

# # visualize optim
# qs = np.array(sol_q)
# b = BiorbdViz(loaded_model=model)
# b.load_movement(qs.T)
# b.exec()