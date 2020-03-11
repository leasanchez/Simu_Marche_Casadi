import biorbd
from ezc3d import c3d
from casadi import *
import matplotlib
import matplotlib.pyplot as plt
from pylab import *
import numpy as np
import time
import _thread

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
nbNoeuds_phase  = [nbNoeuds_stance, nbNoeuds_swing]

nbMus     = model_stance.nbMuscleTotal()               # number of muscles
nbQ       = model_stance.nbDof()                       # number of DoFs
nbMarker  = model_stance.nbMarkers()                   # number of markers
nbBody    = model_stance.nbSegment()                   # number of body segments
nbContact = model_stance.nbContacts()                  # number of contact (2 forces --> plan)

nbU      = nbMus + 3                                   # number of controls : muscle activation + articular torque
nbX      = 2*nbQ                                       # number of states : generalized positions + velocities
nP       = nbMus + 1                                   # number of parameters : 1 global + muscles


# ----------------------------- States & controls ----------------------------------------------------------------------
# CONTROL
u  = MX.sym("u", nbU)
a  = u[:nbMus]                                         # muscular activation
F  = u[nbMus:]                                         # articular torque

# PARAMETERS
p  = MX.sym("p", nP)                                   # maximal isometric force adjustment

# STATE
x  = MX.sym("x", nbX)
q  = x[:nbQ]                                           # generalized coordinates
dq = x[nbQ: 2*nbQ]                                     # velocities


# ----------------------------- Load Data ------------------------------------------------------------------------------
# LOAD MEASUREMENT DATA FROM C3D FILE
name_subject = 'equincocont01'
file         = '/home/leasanchez/programmation/Simu_Marche_Casadi/DonneesMouvement/' + name_subject + '_out.c3d'
kalman_file  = '/home/leasanchez/programmation/Simu_Marche_Casadi/DonneesMouvement/' + name_subject + '_out_MOD5000_leftHanded_GenderF_Florent_.Q2'

save_dir     = '/home/leasanchez/programmation/Simu_Marche_Casadi/Resultats/' + name_subject + '/'

# ground reaction forces
[GRF_real, T, T_stance] = load_data_GRF(file, nbNoeuds_stance, nbNoeuds_swing, 'cycle')
T_swing                 = T - T_stance                                                                # gait cycle time
T_phase                 = [T_stance, T_swing]

# SAVE GRF FROM PLATFORM
filename_GRF = name_subject + '_GRF.txt'
if filename_GRF not in os.listdir(save_dir):
    f = open(save_dir + filename_GRF, 'a')
    f.write("Ground Reaction Forces from force plateform \n\n")
    for n in range(nbNoeuds):
        np.savetxt(f, GRF_real[:, n], delimiter=' , ')
        f.write("\n")
    f.close()


# marker
M_real_stance = load_data_markers(file, T_stance, nbNoeuds_stance, nbMarker, 'stance')
M_real_swing  = load_data_markers(file, T_swing, nbNoeuds_swing, nbMarker, 'swing')

M_real          = np.zeros((3, nbMarker, (nbNoeuds + 1)))
M_real[0, :, :] = np.hstack([M_real_stance[0, :, :], M_real_swing[0, :, :]])
M_real[1, :, :] = np.hstack([M_real_stance[1, :, :], M_real_swing[1, :, :]])
M_real[2, :, :] = np.hstack([M_real_stance[2, :, :], M_real_swing[2, :, :]])

# SAVE MARKERS POSITIONS FROM MOTION CAPTURE
filename_M = name_subject + '_Markers.txt'
if filename_M not in os.listdir(save_dir):
    f = open(save_dir + filename_M, 'a')
    f.write("Markers position from motion capture \n\n")
    for n in range(nbNoeuds):
        for m in range(nbMarker):
            f.write(str(m + 1) + "  :  ")
            np.savetxt(f, M_real[:, m, n], delimiter=',')
            f.write("\n")
        f.write("\n")
    f.close()

# muscular excitation
U_real_swing  = load_data_emg(file, T_swing, nbNoeuds_swing, nbMus, 'swing')
U_real_stance = load_data_emg(file, T_stance, nbNoeuds_stance, nbMus, 'stance')
U_real        = np.hstack([U_real_stance, U_real_swing])

# SAVE EMG
filename_EMG = name_subject + '_EMG.txt'
if filename_EMG not in os.listdir(save_dir):
    f = open(save_dir + filename_EMG, 'a')
    f.write("Muscular excitations from emg \n\n")
    for n in range(nbNoeuds):
        for m in range(nbMus - 7):
            f.write(str(m + 1) + "  :  " + str(U_real[m, n]) + " \n")
        f.write("\n")
    f.close()

# EXTRACT INITIAL MAXIMAL ISOMETRIC FORCES FROM THE MODEL
FISO0 = np.zeros(nbMus)
n_muscle = 0
for nGrp in range(model_stance.nbMuscleGroups()):
    for nMus in range(model_stance.muscleGroup(nGrp).nbMuscles()):
        FISO0[n_muscle] = model_stance.muscleGroup(nGrp).muscle(nMus).characteristics().forceIsoMax()
        n_muscle += 1


nkutta = 4                                             # number of iteration for integration
# ----------------------------- Weighting factors ----------------------------------------------------------------------
wL  = 1                                                # activation
wMa = 30                                               # anatomical marker
wMt = 50                                               # technical marker
wU  = 1                                                # excitation
wR  = 0.05 # 30                                        # ground reaction

# SAVE PARAMETERS USED FOR SIMULATIONS
filename_param = name_subject + '_params.txt'
if filename_param in os.listdir(save_dir):
    os.remove(save_dir + filename_param)

f = open(save_dir + filename_param, 'a')
f.write("Parameters for the simulation \n\n\n")
f.write('MODEL\n')
f.write('model stance : ' + str(model_stance))
f.write('\nmodel swing  : ' + str(model_swing))

f.write('\n\nMODEL PARAMETERS\n')
f.write('\nnbNoeuds stance : ' + str(nbNoeuds_stance))
f.write('\nnbNoeuds swing  : ' + str(nbNoeuds_swing))
f.write('\nnbNoeuds        : ' + str(nbNoeuds))
f.write('\nnbMus           : ' + str(nbMus))
f.write('\nnbQ             : ' + str(nbQ))
f.write('\nnbMarker        : ' + str(nbMarker))
f.write('\nnbBody          : ' + str(nbBody))
f.write('\nnbContact       : ' + str(nbContact))
f.write('\nnbU             : ' + str(nbU))
f.write('\nnbX             : ' + str(nbX))
f.write('\nnP              : ' + str(nP))
f.write('\nnkutta          : ' + str(nkutta))

f.write('\n\nWEIGHTING FACTORS\n')
f.write('wL   : ' + str(wL))
f.write('\nwMa : ' + str(wMa))
f.write('\nwMt : ' + str(wMt))
f.write('\nwU  : ' + str(wU))
f.write('\nwR  : ' + str(wR))
f.close()

# ----------------------------- Movement -------------------------------------------------------------------------------
U = MX.sym("U", nbU*nbNoeuds)                          # controls
X = MX.sym("X", nbX*(nbNoeuds+1))                      # states
G = []                                                 # equality constraints
Ja = 0                                                 # objective function for muscle activation
Jm = 0                                                 # objective function for markers
Je = 0                                                 # objective function for EMG
JR = 0                                                 # objective function for ground reactions

# ------------ PHASE 1 : Stance phase
# FIND THE PARAMETERS P OPTIMISING THE MAXIMUM ISOMETRIC FORCES -- MODEL STANCE
Set_forceISO_max = external('libforce_iso_max_stance', 'libforce_iso_max_stance.so',{'enable_fd':True})
forceISO         = p[0] * p[1:] * FISO0
Set_forceISO_max(forceISO)

for k in range(nbNoeuds_stance):
    # DYNAMIQUE
    Uk = U[nbU*k: nbU*(k + 1)]
    G.append(X[nbX*(k + 1): nbX*(k + 2)] - int_RK4_stance(T_stance, nbNoeuds_stance, nkutta,  X[nbX*k: nbX*(k + 1)], Uk))

    # OBJECTIVE FUNCTION
    # Tracking
    [grf, Jr] = fcn_objective_GRF(wR, X[nbX*k: nbX*(k + 1)], Uk, GRF_real[:, k])                        # Ground Reaction --> stance  # Ground Reaction --> stance
    JR += Jr
    Jm += fcn_objective_markers(wMa, wMt, X[nbX*k: nbX*k + nbQ], M_real_stance[:, :, k], 'stance')      # Marker
    Je += fcn_objective_emg(wU, Uk, U_real_stance[:, k])                                                # EMG

    # Activations
    Ja += fcn_objective_activation(wL, Uk)                                                              # Muscle activations (no EMG)


# ------------ PHASE 2 : Swing phase
# FIND THE PARAMETERS P OPTIMISING THE MAXIMUM ISOMETRIC FORCES -- MODEL SWING
Set_forceISO_max_swing = external('libforce_iso_max', 'libforce_iso_max.so',{'enable_fd':True})
Set_forceISO_max_swing(forceISO)

for k in range(nbNoeuds_swing):
    # DYNAMIQUE
    Uk = U[nbU*nbNoeuds_stance + nbU*k: nbU * nbNoeuds_stance + nbU*(k + 1)]
    G.append(X[nbX * nbNoeuds_stance + nbX*(k + 1): nbX*nbNoeuds_stance + nbX*(k + 2)] - int_RK4_swing(T_swing, nbNoeuds_swing, nkutta,  X[nbX*nbNoeuds_stance + nbX*k: nbX*nbNoeuds_stance + nbX*(k + 1)], Uk))

    # OBJECTIVE FUNCTION
    # Tracking
    Jm += fcn_objective_markers(wMa, wMt, X[nbX*nbNoeuds_stance + nbX*k: nbX*nbNoeuds_stance + nbX*k + nbQ], M_real_swing[:, :, k], 'swing')   # marker
    Je += fcn_objective_emg(wU, Uk, U_real_swing[:, k])                                                                                        # emg
    # Activations
    Ja += fcn_objective_activation(wL, Uk)

# ----------------------------- Contraintes ----------------------------------------------------------------------------
# égalité
lbg = [0]*nbX*nbNoeuds
ubg = [0]*nbX*nbNoeuds

# ----------------------------- Bounds on w ----------------------------------------------------------------------------
# activation - excitation musculaire
min_A = 1e-3                        # 0 exclusif
max_A = 1
lowerbound_u = [min_A]*nbMus + [-1000] + [-2000] + [-200]
upperbound_u = [max_A]*nbMus + [1000]  + [2000]  + [200]
lbu = (lowerbound_u)*nbNoeuds
ubu = (upperbound_u)*nbNoeuds

# q et dq
lowerbound_x = [-10, -0.5, -np.pi/4, -np.pi/9, -np.pi/2, -np.pi/3, 0.5, -0.5, -1.7453, -5.2360, -5.2360, -5.2360]
upperbound_x = [10, 1.5, np.pi/4, np.pi/3, 0.0873, np.pi/9, 1.5, 0.5, 1.7453, 5.2360, 5.2360, 5.2360]
lbX   = (lowerbound_x)*(nbNoeuds + 1)
ubX   = (upperbound_x)*(nbNoeuds + 1)

# parameters
min_pg = 1
min_p  = 0.2
max_p  = 5
max_pg = 1
lbp = [min_pg] + [min_p]*nbMus
ubp = [max_pg] + [max_p]*nbMus


lbx = vertcat(lbu, lbX, lbp)
ubx = vertcat(ubu, ubX, ubp)

# SAVE BOUNDS
f = open(save_dir + filename_param, 'a')
f.write('\n\nBOUNDS\n')
f.write('Control max\n')
np.savetxt(f, upperbound_u, delimiter='\n')
f.write('\nControl min\n')
np.savetxt(f, lowerbound_u, delimiter='\n')
f.write('\n\nState max \n')
np.savetxt(f, upperbound_x, delimiter='\n')
f.write('\nState min\n')
np.savetxt(f, lowerbound_x, delimiter='\n')
f.write('\n\nParameter max \n')
np.savetxt(f, ubp, delimiter='\n')
f.write('\nParameter min\n')
np.savetxt(f, lbp, delimiter='\n')
f.close()

# ----------------------------- Initial guess --------------------------------------------------------------------------
init_A = 0.1

# CONTROL
u0               = np.zeros((nbU, nbNoeuds))
u0[: nbMus, :]   = load_initialguess_muscularExcitation(np.hstack([U_real_stance, U_real_swing]))
# u0[: nbMus, :]   = np.zeros((nbMus, nbNoeuds)) + init_A
u0[nbMus + 0, :] = [0]*nbNoeuds                                  # pelvis forces
u0[nbMus + 1, :] = [-500]*nbNoeuds_stance + [0]*nbNoeuds_swing
u0[nbMus + 2, :] = [0]*nbNoeuds
# u0               = vertcat(*u0.T)

# STATE
q0_stance = load_initialguess_q(file, kalman_file, T_stance, nbNoeuds_stance, 'stance')
q0_swing  = load_initialguess_q(file, kalman_file, T_swing, nbNoeuds_swing, 'swing')
q0        = np.hstack([q0_stance, q0_swing])
dq0       = np.zeros((nbQ, (nbNoeuds + 1)))

X0                = np.zeros((nbX, (nbNoeuds + 1)))
X0[:nbQ, :]       = q0
X0[nbQ: 2*nbQ, :] = dq0
# X0                = vertcat(*X0.T)

# PARAMETERS
p0 = [1] + [1]*nbMus

# SAVE INITIAL GUESS
filename_init = name_subject + '_initialguess.txt'
if filename_init not in os.listdir(save_dir):
    f = open(save_dir + filename_init, 'a')
    f.write('Initial guess\n\n\n')
    f.write('CONTROL\n')
    for n in range(nbNoeuds):
        f.write('\nu0   ' + str(n) + '\n')
        np.savetxt(f, u0[:, n], delimiter='\n')
    f.write('\n\nSTATE\n')
    for n in range(nbNoeuds + 1):
        f.write('\nx0   ' + str(n) + '\n')
        np.savetxt(f, X0[:, n], delimiter='\n')
    f.write('\n\nPARAMETER\n')
    np.savetxt(f, p0, delimiter='\n')
f.close()


w0 = vertcat(vertcat(*u0.T), vertcat(*X0.T), p0)

# ----------------------------- Callback -------------------------------------------------------------------------------
class AnimateCallback(Callback):
    def __init__(self, name, nx, ng, np, opts={}):
        Callback.__init__(self)
        self.nx     = nx                   # optimized value number
        self.ng     = ng                   # constraints number
        self.nP     = np

        self.sol_data   = None             # optimized variables
        self.update_sol = True             # first iteration

        self.construct(name, opts)

    def get_n_in(self): return nlpsol_n_out()
    def get_n_out(self): return 1
    def get_name_in(self, i): return nlpsol_out(i)
    def get_name_out(self, i): return "ret"

    def get_sparsity_in(self, i):
        n = nlpsol_out(i)
        if n == 'f': return Sparsity.scalar()
        elif n in ('x', 'lam_x'): return Sparsity.dense(self.nx)
        elif n in ('g', 'lam_g'): return Sparsity.dense(self.ng)
        else: return Sparsity(0, 0)

    def eval(self, arg):
        darg = {}
        for (i, s) in enumerate(nlpsol_out()):
            darg[s] = arg[i]

        self.sol_data   = darg["x"]
        self.update_sol = True
        # plot_callback(self)
        print('coucou')
        return [0]

mycallback = AnimateCallback('mycallback', (nbU*nbNoeuds + nbX*(nbNoeuds + 1) + nP), nbX*nbNoeuds, 0)


# ----------------------------- Plot results optimization --------------------------------------------------------------
def plot_callback(callback):
    # INITIALISATION GRAPH WITH INITIAL GUESS
    # TIME
    t_stance = np.linspace(0, T_stance, nbNoeuds_stance)
    t_swing = t_stance[-1] + np.linspace(0, T_swing, nbNoeuds_swing)
    t = np.hstack([t_stance, t_swing])

    def plot_control(ax, t, x):
        nbPoints = len(np.array(x))
        for n in range(nbPoints - 1):
            ax.plot([t[n], t[n + 1], t[n + 1]], [x[n], x[n], x[n + 1]], 'b')

    # CONTROL
    fig1, axes1 = plt.subplots(5, 4, sharex=True, figsize=(10, 10))                                                     # Create figure
    Labels = ['GLUT_MAX1', 'GLUT_MAX2', 'GLUT_MAX3', 'GLUT_MED1', 'GLUT_MED2', 'GLUT_MED3',
              'R_SEMIMEM', 'R_SEMITEN', 'R_BI_FEM_LH', 'R_RECTUS_FEM', 'R_VAS_MED', 'R_VAS_INT',
              'R_VAS_LAT', 'R_GAS_MED', 'R_GAS_LAT', 'R_SOLEUS', 'R_TIB_ANT', 'Pelvis Tx', 'Pelvis Ty', 'Pelvis Rz']    # Control labels
    axes1 = axes1.flatten()                                                                                             # Get axes figure (vector)
    u_emg = 9                                                                                                           # init variable for muscle with emg
    for i in range(nbU):
        ax = axes1[i]                                                            # get the correct subplot
        ax.set_title(Labels[i])                                                  # put control label
        ax.plot([0, T], [lowerbound_u[i], lowerbound_u[i]], 'k--')               # lower bound
        ax.plot([0, T], [upperbound_u[i], upperbound_u[i]], 'k--')               # upper bound
        ax.plot([T_stance, T_stance], [lowerbound_u[i], upperbound_u[i]], 'k:')  # end of the stance phase
        ax.grid(True)
        if (i != 1) and (i != 2) and (i != 3) and (i != 5) and (i != 6) and (i != 11) and (i != 12) and (i < (nbMus - 1)):
            ax.plot(t, U_real[u_emg, :], 'r')                                    # plot emg if available
            u_emg -= 1
        if (i > nbU - 5):
            ax.set_xlabel('time (s)')
        if (i < (nbMus - 1)):
            ax.yaxis.set_ticks(np.arange(0, 1, 0.5))
        plot_control(ax, t, u0[i, :])                                            # plot initial guess
    plt.savefig(save_dir + '/RES/plot_control_initialguess')                     # save figure of initial guess


    # STATES
    ts = np.hstack([t_stance, t_swing, t_swing[-1] + (t_swing[-1] - t_swing[-2])])                                      # Adjusted time (T + dt)

    fig2, axes2 = plt.subplots(2, 6, sharex=True, figsize=(20, 10))                                                     # Create figure
    axes2       = axes2.flatten()                                                                                       # Get axes figure (vector)
    Labels_X    = ['Pelvis_TX', 'Pelvis_TY', 'Pelvis_RZ', 'Hip', 'Knee', 'Ankle']                                       # Control labels

    for q in range(nbQ):
        ax1 = axes2[q]
        ax1.set_title('Q ' + Labels_X[q])
        if q != 0 and q != 1:
            ax1.plot([0, T], [lowerbound_x[q] * (180 / np.pi), lowerbound_x[q] * (180 / np.pi)], 'k--')                 # lower bound
            ax1.plot([0, T], [upperbound_x[q] * (180 / np.pi), upperbound_x[q] * (180 / np.pi)], 'k--')                 # upper bound
            ax1.plot([T_stance, T_stance], [lowerbound_x[q] * (180 / np.pi), upperbound_x[q] * (180 / np.pi)], 'k:')    # end of the stance phase
            ax1.plot(ts, X0[q, :] * (180 / np.pi), 'b')                                                                 # plot initial guess q (degre)
        else:
            ax1.plot([0, T], [lowerbound_x[q], lowerbound_x[q]], 'k--')
            ax1.plot([0, T], [upperbound_x[q], upperbound_x[q]], 'k--')
            ax1.plot([T_stance, T_stance], [lowerbound_x[q], upperbound_x[q]], 'k:')                                    # end of the stance phase
            ax1.plot(ts, X0[q, :], 'b')                                                                                 # plot initial guess q (trans)
        ax1.grid(True)

        ax2 = axes2[q + nbQ]
        ax2.set_title('dQ ' + Labels_X[q])
        ax2.plot([0, T], [lowerbound_x[q], lowerbound_x[q]], 'k--')                                                     # lower bound
        ax2.plot([0, T], [upperbound_x[q], upperbound_x[q]], 'k--')                                                     # upper bound
        ax2.plot([T_stance, T_stance], [lowerbound_x[q], upperbound_x[q]], 'k:')                                        # end of the stance phase
        ax2.plot(ts, X0[(q + nbQ), :], 'b')                                                                             # plot initial guess dq
        ax2.set_xlabel('time (s)')
        ax2.grid(True)
    plt.savefig(save_dir + '/RES/plot_state_initialguess')                                                              # save initial guess plot

    # CONVERGENCE
    JR = Je = Jm = Ja = constraint = 0

    GRF = np.zeros((3, nbNoeuds))
    for k in range(nbNoeuds_stance):
        # GROUND REACTION FORCES
        [F, Jr] = fcn_objective_GRF(wR, X0[:, k], u0[:, k], GRF_real[:, k])  # Ground Reaction --> stance
        GRF[:, k] = np.array(F).squeeze()

        # OBJECTIVE FUNCTION
        JR += Jr  # Ground Reaction --> stance
        Jm += fcn_objective_markers(wMa, wMt, q0[: , k], M_real_stance[:, :, k], 'stance')  # Marker
        Je += fcn_objective_emg(wU, u0[:, k], U_real_stance[:, k])                          # EMG
        Ja += fcn_objective_activation(wL, u0[:, k])                                        # Muscle activations (no EMG)

        # CONSTRAINT
        constraint += X0[:, k + 1] - int_RK4_stance(T_stance, nbNoeuds_stance, nkutta, X0[:, k], u0[:, k])

    for k in range(nbNoeuds_swing):
        # OBJECTIVE FUNCTION
        Jm += fcn_objective_markers(wMa, wMt, q0[: , nbNoeuds_stance + k], M_real_swing[:, :, k], 'swing')
        Je += fcn_objective_emg(wU, u0[:, nbNoeuds_stance + k], U_real_swing[:, k])
        Ja += fcn_objective_activation(wL, u0[:, nbNoeuds_stance + k])

        # CONSTRAINT
        constraint += X0[:, nbNoeuds_stance + k + 1] - int_RK4_swing(T_swing, nbNoeuds_swing, nkutta, X0[:, nbNoeuds_stance + k], u0[:, nbNoeuds_stance + k])

    J = Ja + Je + Jm + JR

    print('\n \nGlobal                 : ' + str(J))
    print('activation             : ' + str(Ja))
    print('emg                    : ' + str(Je))
    print('marker                 : ' + str(Jm))
    print('ground reaction forces : ' + str(JR))
    print('constraints            : ' + str(sum(constraint)) + '\n')

    # GROUND REACTION FORCES
    fig3, axes3 = plt.subplots(2, 1, sharex=True)
    axes3.flatten()

    ax_ap = axes3[0]
    ax_ap.set_title('GRF A/P  during the gait')
    ax_ap.plot(t, GRF_real[1, :], 'r')
    ax_ap.plot([T_stance, T_stance], [min(GRF_real[1, :]), max(GRF_real[1, :])], 'k:')  # end of the stance phase
    ax_ap.plot(t, GRF[0, :], 'b')
    ax_ap.grid(True)

    ax_v = axes3[1]
    ax_v.set_title('GRF vertical')
    ax_v.plot(t, GRF_real[2, :], 'r')
    ax_v.plot([T_stance, T_stance], [min(GRF_real[2, :]), max(GRF_real[2, :])], 'k:')
    ax_v.plot(t, GRF[2, :], 'b')
    ax_v.set_xlabel('time (s)')
    ax_v.grid(True)
    fig3.tight_layout()
    plt.savefig(save_dir + '/RES/plot_GRF_initialguess')

    plt.show(block=False)


    while plt.get_fignums():            # figures opened
        if callback.update_sol:         # optimized values modified (new iteration)
            # STRUCTURE DATA
            sol_U = callback.sol_data[:nbU * nbNoeuds]
            sol_X = callback.sol_data[nbU * nbNoeuds: -nP]

            sol_q  = np.hstack([sol_X[0::nbX], sol_X[1::nbX], sol_X[2::nbX], sol_X[3::nbX], sol_X[4::nbX], sol_X[5::nbX]])
            sol_dq = np.hstack([sol_X[6::nbX], sol_X[7::nbX], sol_X[8::nbX], sol_X[9::nbX], sol_X[10::nbX], sol_X[11::nbX]])
            sol_a  = np.hstack([sol_U[0::nbU], sol_U[1::nbU], sol_U[2::nbU], sol_U[3::nbU], sol_U[4::nbU], sol_U[5::nbU],
                               sol_U[6::nbU], sol_U[7::nbU], sol_U[8::nbU], sol_U[9::nbU], sol_U[10::nbU],
                               sol_U[11::nbU], sol_U[12::nbU], sol_U[13::nbU], sol_U[14::nbU], sol_U[15::nbU],
                               sol_U[16::nbU]])
            sol_F  = np.hstack([sol_U[17::nbU], sol_U[18::nbU], sol_U[19::nbU]])

            # CONVERGENCE
            JR = Je = Jm = Ja = constraint = 0
            GRF = np.zeros((3, nbNoeuds))

            for k in range(nbNoeuds_stance):
                # GROUND REACTION FORCES
                [F, Jr] = fcn_objective_GRF(wR, sol_X[nbX * k: nbX * (k + 1)], sol_U[nbU * k: nbU * (k + 1)], GRF_real[:, k])  # Ground Reaction --> stance
                GRF[:, k] = np.array(F).squeeze()

                # OBJECTIVE FUNCTION
                JR += Jr                                                                                                # Ground Reaction --> stance
                Jm += fcn_objective_markers(wMa, wMt, sol_X[nbX * k: nbX * k + nbQ], M_real_stance[:, :, k], 'stance')  # Marker
                Je += fcn_objective_emg(wU, sol_U[nbU * k: nbU * (k + 1)], U_real_stance[:, k])                         # EMG
                Ja += fcn_objective_activation(wL, sol_U[nbU * k: nbU * (k + 1)])                                       # Muscle activations (no EMG)

                # CONSTRAINT
                constraint += sol_X[nbX * (k + 1): nbX * (k + 2)] - int_RK4_stance(T_stance, nbNoeuds_stance, nkutta, sol_X[nbX * k: nbX * (k + 1)], sol_U[nbU * k: nbU * (k + 1)])

            for k in range(nbNoeuds_swing):
                # OBJECTIVE FUNCTION
                Jm += fcn_objective_markers(wMa, wMt, sol_X[nbX * nbNoeuds_stance + nbX * k: nbX * nbNoeuds_stance + nbX * k + nbQ], M_real_swing[:, :, k], 'swing')
                Je += fcn_objective_emg(wU, sol_U[nbU * nbNoeuds_stance + nbU * k: nbU * nbNoeuds_stance + nbU * (k + 1)], U_real_swing[:, k])
                Ja += fcn_objective_activation(wL, sol_U[nbU * nbNoeuds_stance + nbU * k: nbU * nbNoeuds_stance + nbU * (k + 1)])

                # CONSTRAINT
                constraint += sol_X[nbX * nbNoeuds_stance + nbX * (k + 1): nbX * nbNoeuds_stance + nbX * (k + 2)] - int_RK4_swing(T_swing, nbNoeuds_swing, nkutta,
                                                                                                                                  sol_X[nbX * nbNoeuds_stance + nbX * k: nbX * nbNoeuds_stance + nbX * (k + 1)],
                                                                                                                                  sol_U[nbU * nbNoeuds_stance + nbU * k: nbU * nbNoeuds_stance + nbU * (k + 1)])

            J = Ja + Je + Jm + JR

            # PRINT VALUES
            print('\n \nGlobal                 : ' + str(J))
            print('activation             : ' + str(Ja))
            print('emg                    : ' + str(Je))
            print('marker                 : ' + str(Jm))
            print('ground reaction forces : ' + str(JR))
            print('constraints            : ' + str(sum(constraint)) + '\n')

            # SAVE OBJECTIVE FUNCTION AND CONSTRAINTS VALUE FOR EACH ITERATION IN TXT
            filename_J = name_subject + '_objvalue.txt'
            f = open(save_dir + '/RES/' + filename_J, 'a')
            f.write('Global                 : ' + str(J) + '\n')
            f.write('activation             : ' + str(Ja) + '\n')
            f.write('emg                    : ' + str(Je) + '\n')
            f.write('marker                 : ' + str(Jm) + '\n')
            f.write('ground reaction forces : ' + str(JR) + '\n')
            f.write('constraints            : ' + str(sum(constraint)) + '\n\n')
            f.close()

            def plot_control_update(ax, t, x):
                nbPoints = len(np.array(x))
                for n in range(nbPoints - 1):
                    lines = ax.get_lines()
                    if size(lines) > 52:
                        lines[4 + n].set_xdata([t[n], t[n + 1], t[n + 1]])
                        lines[4 + n].set_ydata([x[n], x[n], x[n + 1]])
                    else:
                        lines[3 + n].set_xdata([t[n], t[n + 1], t[n + 1]])
                        lines[3 + n].set_ydata([x[n], x[n], x[n + 1]])

            # CONTROL
            axes1 = plt.figure(1).axes
            for i in range(nbMus):
                ax = axes1[i]
                plot_control_update(ax, t, sol_a[:, i])
            for j in range(3):
                ax = axes1[i + 1 + j]
                plot_control_update(ax, t, sol_F[:, j])
            plt.savefig(save_dir + '/RES/plot_control')

            # STATE
            axes2 = plt.figure(2).axes
            for q in range(nbQ):
                ax1 = axes2[q]
                lines = ax1.get_lines()
                if q != 0 and q != 1:
                    lines[3].set_ydata(sol_q[:, q] * (180 / np.pi))
                else:
                    lines[3].set_ydata(sol_q[:, q])
            for dq in range(nbQ):
                ax1 = axes2[q + 1 + dq]
                lines = ax1.get_lines()
                lines[3].set_ydata(sol_dq[:, dq])
            plt.savefig(save_dir + '/RES/plot_state')

            # GRF
            axes3 = plt.figure(3).axes
            ax_ap = axes3[0]
            lines = ax_ap.get_lines()
            lines[2].set_ydata(GRF[0, :])

            ax_v  = axes3[1]
            lines = ax_v.get_lines()
            lines[2].set_ydata(GRF[2, :])
            plt.savefig(save_dir + '/RES/plot_GRF')

            callback.update_sol = False         # can get new iteration
        plt.draw()                              # update plots
        plt.pause(.001)


# _thread.start_new_thread(plot_callback, (mycallback,))          # nouveau thread

# ----------------------------- Solver ---------------------------------------------------------------------------------
w = vertcat(U, X, p)
J = Ja + Je + Jm + JR

nlp    = {'x': w, 'f': J, 'g': vertcat(*G)}
opts   = {"ipopt.tol": 1e-1, "ipopt.linear_solver": "ma57", "ipopt.hessian_approximation":"limited-memory", "iteration_callback": mycallback}
solver = nlpsol("solver", "ipopt", nlp, opts)

# start_opti = time.time()

res = solver(lbg = lbg,
             ubg = ubg,
             lbx = lbx,
             ubx = ubx,
             x0  = w0)


# RESULTS
# stop_opti = time.time() - start_opti
# print('Time to solve : ' + str(stop_opti))

sol_U  = res["x"][:nbU * nbNoeuds]
sol_X  = res["x"][nbU * nbNoeuds: -nP]
sol_p  = res["x"][-nP:]

sol_q  = [sol_X[0::nbX], sol_X[1::nbX], sol_X[2::nbX], sol_X[3::nbX], sol_X[4::nbX], sol_X[5::nbX]]
sol_dq = [sol_X[6::nbX], sol_X[7::nbX], sol_X[8::nbX], sol_X[9::nbX], sol_X[10::nbX], sol_X[11::nbX]]
sol_a  = [sol_U[0::nbU], sol_U[1::nbU], sol_U[2::nbU], sol_U[3::nbU], sol_U[4::nbU], sol_U[5::nbU], sol_U[6::nbU],
         sol_U[7::nbU], sol_U[8::nbU], sol_U[9::nbU], sol_U[10::nbU], sol_U[11::nbU], sol_U[12::nbU], sol_U[13::nbU],
         sol_U[14::nbU], sol_U[15::nbU], sol_U[16::nbU]]
sol_F  = [sol_U[17::nbU], sol_U[18::nbU], sol_U[19::nbU]]

nbNoeuds_phase = [nbNoeuds_stance, nbNoeuds_swing]
T_phase        = [T_stance, T_swing]