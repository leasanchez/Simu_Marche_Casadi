import biorbd
from ezc3d import c3d
from casadi import *
from pylab import *
import numpy as np
import time
from Marche_Fcn_Integration import *
from Fcn_Objective import *
from LoadData import *
from Fcn_Affichage import *
from Fcn_InitialGuess import *

# SET MODELS
model_swing  = biorbd.Model('/home/leasanchez/programmation/Marche_Florent/ModelesS2M/ANsWER_Rleg_6dof_17muscle_0contact.bioMod')
model_stance = biorbd.Model('/home/leasanchez/programmation/Marche_Florent/ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact.bioMod')

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


# ----------------------------- States & controls ----------------------------------------------------------------------
# CONTROL
u  = MX.sym("u", nbU)
e  = u[:nbMus]                                         # muscular excitation
F  = u[nbMus:]                                         # articular torque

# PARAMETERS
p  = MX.sym("p", nP)                                   # maximal isometric force adjustment

# STATE
x  = MX.sym("x", nbX)
q  = x[:nbQ]                                           # generalized coordinates
dq = x[nbQ: 2*nbQ]                                     # velocities


# ----------------------------- Load Data ------------------------------------------------------------------------------
# LOAD MEASUREMENT DATA FROM C3D FILE
file        = '/home/leasanchez/programmation/Marche_Florent/DonneesMouvement/equinus01_out.c3d'
kalman_file = '/home/leasanchez/programmation/Marche_Florent/DonneesMouvement/equinus01_out_MOD5000_leftHanded_GenderF_Florent_.Q2'

# ground reaction forces
[GRF_real, T, T_stance] = load_data_GRF(file, nbNoeuds_stance, nbNoeuds_swing, 'cycle')
T_swing                 = T - T_stance                                                                # gait cycle time

# marker
M_real_stance = load_data_markers(file, T_stance, nbNoeuds_stance, nbMarker, 'stance')
M_real_swing  = load_data_markers(file, T_swing, nbNoeuds_swing, nbMarker, 'swing')

# muscular excitation
U_real_swing  = load_data_emg(file, T_swing, nbNoeuds_swing, nbMus, 'swing')
U_real_stance = load_data_emg(file, T_stance, nbNoeuds_stance, nbMus, 'stance')


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
wR  = 0.05                                             # ground reaction




# ----------------------------- Movement -------------------------------------------------------------------------------
U = MX.sym("U", nbU*nbNoeuds)                          # controls
X = MX.sym("X", nbX*(nbNoeuds+1))                      # states
G = []                                                 # equality constraints
Ja = 0                                                 # objective function for muscle activation
Jm = 0                                                 # objective function for markers
Je = 0                                                 # objective function for EMG
JR = 0                                                 # objective function for ground reactions

M_simu = np.zeros((nbMarker, nbNoeuds))
# ------------ PHASE 1 : Stance phase
# FIND THE PARAMETERS P OPTIMISING THE MAXIMUM ISOMETRIC FORCES -- MODEL STANCE
Set_forceISO_max = external('libforce_iso_max_stance', 'libforce_iso_max_stance.so',{'enable_fd':True})
forceISO         = p[0]*p[1:]*FISO0
Set_forceISO_max(forceISO)

for k in range(nbNoeuds_stance):
    # DYNAMIQUE
    Uk = U[nbU*k: nbU*(k + 1)]
    G.append(X[nbX*(k + 1): nbX*(k + 2)] - int_RK4_stance(T_stance, nbNoeuds_stance, nkutta,  X[nbX*k: nbX*(k + 1)], Uk))

    # OBJECTIVE FUNCTION
    # Tracking
    JR += fcn_objective_GRF(wR, X[nbX*k: nbX*(k + 1)], Uk, GRF_real[:, k])                              # Ground Reaction --> stance
    Jm += fcn_objective_markers(wMa, wMt, X[nbX*k: nbX*k + nbQ], M_real_stance[:, :, k], 'stance')      # Marker
    Je += fcn_objective_emg(wU, Uk, U_real_stance[:, k])                                                # EMG

    # Activations
    Ja += fcn_objective_activation(wL, Uk)                                                                 # Muscle activations (no EMG)


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
# contrainte cyclique???

# ----------------------------- Bounds on w ----------------------------------------------------------------------------
# activation - excitation musculaire
min_A = 1e-3                        # 0 exclusif
max_A = 1

lbu = ([min_A]*nbMus + [-1000] + [-2000] + [-200])*nbNoeuds
ubu = ([max_A]*nbMus + [1000]  + [2000]  + [200])*nbNoeuds

# q et dq
min_Q = -50
max_Q = 50
lbX   = ([min_Q]*2*nbQ )*(nbNoeuds + 1)
ubX   = ([max_Q]*2*nbQ)*(nbNoeuds + 1)

# parameters
min_pg = 1
min_p  = 0.2
max_p  = 5
max_pg = 1
lbp = [min_pg] + [min_p]*nbMus
ubp = [max_pg] + [max_p]*nbMus


lbx = vertcat(lbu, lbX, lbp)
ubx = vertcat(ubu, ubX, ubp)

# ----------------------------- Initial guess --------------------------------------------------------------------------
init_A = 0.1

# CONTROL
u0               = np.zeros((nbU, nbNoeuds))
# u0[: nbMus, :]   = load_initialguess_muscularExcitation(np.hstack([U_real_stance, U_real_swing]))
u0[: nbMus, :]   = np.zeros((nbMus, nbNoeuds)) + init_A
u0[nbMus + 0, :] = [0]*nbNoeuds               # pelvis forces
u0[nbMus + 1, :] = [-500]*nbNoeuds_stance + [0]*nbNoeuds_swing
u0[nbMus + 2, :] = [0]*nbNoeuds
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

# PARAMETERS
p0 = [1] + [1]*nbMus


x0 = vertcat(u0, X0, p0)

# ----------------------------- Solver ---------------------------------------------------------------------------------
w = vertcat(U, X, p)
J = Ja + Je + Jm + JR

class AnimateCallback(casadi.Callback):
    def __init__(self, name):
        Callback.__init__(self)
        self.name = name               # callback name

        self.J  = []                    # objective function
        self.Ja = []
        self.Je = []
        self.Jm = []
        self.JR = []

        # CONVERGENCE
        plt.figure(1)
        title('Convergence objective functions')

        # CONTROL
        plt.figure(2)
        Labels = ['GLUT_MAX1', 'GLUT_MAX2', 'GLUT_MAX3', 'GLUT_MED1', 'GLUT_MED2', 'GLUT_MED3',
                  'R_SEMIMEM', 'R_SEMITEN', 'R_BI_FEM_LH', 'R_RECTUS_FEM', 'R_VAS_MED', 'R_VAS_INT',
                  'R_VAS_LAT', 'R_GAS_MED', 'R_GAS_LAT', 'R_SOLEUS', 'R_TIB_ANT']
        for nMus in range(nbMus):
            plt.subplot(5, 4, nMus + 1)
            title(Labels[nMus])
            plt.plot([0, nbNoeuds], [min_A, min_A], 'k--')  # lower bound
            plt.plot([0, nbNoeuds], [max_A, max_A], 'k--')  # upper bound

        plt.subplot(5, 4, nbMus + 1)
        plt.title('Pelvis Tx')
        plt.plot([0, nbNoeuds], [-1000, -1000], 'k--')
        plt.plot([0, nbNoeuds], [1000, 1000], 'k--')

        plt.subplot(5, 4, nbMus + 2)
        plt.title('Pelvis Ty')
        plt.plot([0, nbNoeuds], [-2000, -2000], 'k--')
        plt.plot([0, nbNoeuds], [2000, 2000], 'k--')

        plt.subplot(5, 4, nbMus + 3)
        plt.title('Pelvis Rz')
        plt.plot([0, nbNoeuds], [-200, -200], 'k--')
        plt.plot([0, nbNoeuds], [200, 200], 'k--')


        # STATES
        plt.figure(3)
        Labels_X = ['Pelvis_TX', 'Pelvis_TY', 'Pelvis_RZ', 'Hip', 'Knee', 'Ankle']
        for q in range(nbQ):
            plt.subplot(2, 6, q + 1)
            plt.title('Q ' + Labels_X[q])
            plt.plot([0, nbNoeuds], [min_Q, min_Q], 'k--')
            plt.plot([0, nbNoeuds], [max_Q, max_Q], 'k--')

            plt.subplot(2, 6, q + 1 + nbQ)
            plt.title('dQ ' + Labels_X[q])
            plt.plot([0, nbNoeuds], [min_Q, min_Q], 'k--')
            plt.plot([0, nbNoeuds], [max_Q, max_Q], 'k--')

        plt.show()

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
            # GET CONTROL AND STATES
            for (i, s) in enumerate(nlpsol_out()): darg[s] = arg[i]
            sol_U = darg["x"][:nbU * nbNoeuds]
            sol_X = darg["x"][nbU * nbNoeuds: -nP]

            sol_q  = [np.array(sol_X[0::nbX]).squeeze(), np.array(sol_X[1::nbX]).squeeze(), np.array(sol_X[2::nbX]).squeeze(), np.array(sol_X[3::nbX]).squeeze(), np.array(sol_X[4::nbX]).squeeze(), np.array(sol_X[5::nbX]).squeeze()]
            sol_dq = [np.array(sol_X[6::nbX]).squeeze(), np.array(sol_X[7::nbX]).squeeze(), np.array(sol_X[8::nbX]).squeeze(), np.array(sol_X[9::nbX]).squeeze(), np.array(sol_X[10::nbX]).squeeze(), np.array(sol_X[11::nbX]).squeeze()]
            sol_a  = [np.array(sol_U[0::nbU]).squeeze(), np.array(sol_U[1::nbU]).squeeze(), np.array(sol_U[2::nbU]).squeeze(), np.array(sol_U[3::nbU]).squeeze(), np.array(sol_U[4::nbU]).squeeze(), np.array(sol_U[5::nbU]).squeeze(), np.array(sol_U[6::nbU]).squeeze(), np.array(sol_U[7::nbU]).squeeze(), np.array(sol_U[8::nbU]).squeeze(), np.array(sol_U[9::nbU]).squeeze(), np.array(sol_U[10::nbU]).squeeze(), np.array(sol_U[11::nbU]).squeeze(), np.array(sol_U[12::nbU]).squeeze(), np.array(sol_U[13::nbU]).squeeze(), np.array(sol_U[14::nbU]).squeeze(), np.array(sol_U[15::nbU]).squeeze(), np.array(sol_U[16::nbU]).squeeze()]
            sol_F  = [np.array(sol_U[17::nbU]).squeeze(), np.array(sol_U[18::nbU]).squeeze(), np.array(sol_U[19::nbU]).squeeze()]

            # CONVERGENCE
            JR = 0
            Je = 0
            Jm = 0
            Ja = 0

            # OBJECTIVE FUNCTION
            for k in range(nbNoeuds_stance):
                JR += fcn_objective_GRF(wR, X[nbX * k: nbX * (k + 1)], U[nbU * k: nbU * (k + 1)], GRF_real[:, k])  # Ground Reaction --> stance
                Jm += fcn_objective_markers(wMa, wMt, X[nbX * k: nbX * k + nbQ], M_real_stance[:, :, k],'stance')  # Marker
                Je += fcn_objective_emg(wU, U[nbU * k: nbU * (k + 1)], U_real_stance[:, k])                        # EMG
                Ja += fcn_objective_activation(wL, U[nbU * k: nbU * (k + 1)])                                      # Muscle activations (no EMG)
            for k in range(nbNoeuds_swing):
                Jm += fcn_objective_markers(wMa, wMt, X[nbX * nbNoeuds_stance + nbX * k: nbX * nbNoeuds_stance + nbX * k + nbQ], M_real_swing[:, :, k], 'swing')  # marker
                Je += fcn_objective_emg(wU, U[nbU * nbNoeuds_stance + nbU * k: nbU * nbNoeuds_stance + nbU * (k + 1)], U_real_swing[:, k])  # emg
                Ja += fcn_objective_activation(wL, U[nbU * nbNoeuds_stance + nbU * k: nbU * nbNoeuds_stance + nbU * (k + 1)])

            J = Ja + Je + Jm + JR

            self.J.append(J)
            self.Ja.append(Ja)
            self.Je.append(Je)
            self.Jm.append(Jm)
            self.JR.append(JR)

            plt.figure(1)
            plt.plot(J,  'r', Labels = 'global')
            plt.plot(Jm, 'g', Labels = 'markers')
            plt.plot(Je, 'b', Labels = 'emg')
            plt.plot(JR, 'o', Labels = 'ground reaction forces')
            plt.plot(Ja, 'k', Labels = 'activations')
            plt.legend()

            # CONTROL
            plt.figure(2)
            # muscular activation
            for nMus in range(nbMus):
                plt.subplot(5, 4, nMus + 1)
                for n in range(nbNoeuds - 1):
                    plt.plot([n, n + 1, n + 1], [sol_a[nMus, n], sol_a[nMus, n], sol_a[nMus, n + 1]], 'b')
            # pelvis forces
            for nF in range(3):
                plt.subplot(5, 4, nbMus + nF)
                for n in range(nbNoeuds - 1):
                    plt.plot([n, n + 1, n + 1], [sol_F[nF, n], sol_F[nF, n], sol_F[nF, n + 1]], 'b')

            # STATE
            plt.figure(3)
            for q in range(nbQ):
                # joint position
                plt.subplot(2, 6, q)
                plt.plot(sol_q[q, :])

                # velocities
                plt.subplot(2, 6, q + nbQ)
                plt.plot(sol_dq[q, :])

        plt.show()

nlp = {'x': w, 'f': J, 'g': vertcat(*G)}
callback = AnimateCallback('callback')
opts = {"ipopt.tol": 1e-1, "ipopt.linear_solver": "ma57", "ipopt.hessian_approximation":"limited-memory", "iteration_callback": callback}
solver = nlpsol("solver", "ipopt", nlp, opts)

start_opti = time.time()
print('Start optimisation : ' + str(start_opti))
res = solver(lbg = lbg,
             ubg = ubg,
             lbx = lbx,
             ubx = ubx,
             x0  = x0)


# RESULTS
stop_opti = time.time() - start_opti
print('Time to solve : ' + str(stop_opti))
save()

sol_U  = res["x"][:nbU * nbNoeuds]
sol_X  = res["x"][nbU * nbNoeuds: -nP]
sol_p  = res["x"][-nP:]

sol_q = [np.array(sol_X[0::nbX]).squeeze(), np.array(sol_X[1::nbX]).squeeze(), np.array(sol_X[2::nbX]).squeeze(),
         np.array(sol_X[3::nbX]).squeeze(), np.array(sol_X[4::nbX]).squeeze(), np.array(sol_X[5::nbX]).squeeze()]

sol_dq = [np.array(sol_X[6::nbX]).squeeze(), np.array(sol_X[7::nbX]).squeeze(), np.array(sol_X[8::nbX]).squeeze(),
          np.array(sol_X[9::nbX]).squeeze(), np.array(sol_X[10::nbX]).squeeze(), np.array(sol_X[11::nbX]).squeeze()]

sol_a = [np.array(sol_U[0::nbU]).squeeze(), np.array(sol_U[1::nbU]).squeeze(), np.array(sol_U[2::nbU]).squeeze(),
         np.array(sol_U[3::nbU]).squeeze(), np.array(sol_U[4::nbU]).squeeze(), np.array(sol_U[5::nbU]).squeeze(),
         np.array(sol_U[6::nbU]).squeeze(), np.array(sol_U[7::nbU]).squeeze(), np.array(sol_U[8::nbU]).squeeze(),
         np.array(sol_U[9::nbU]).squeeze(), np.array(sol_U[10::nbU]).squeeze(), np.array(sol_U[11::nbU]).squeeze(),
         np.array(sol_U[12::nbU]).squeeze(), np.array(sol_U[13::nbU]).squeeze(), np.array(sol_U[14::nbU]).squeeze(),
         np.array(sol_U[15::nbU]).squeeze(), np.array(sol_U[16::nbU]).squeeze()]

sol_F = [np.array(sol_U[17::nbU]).squeeze(), np.array(sol_U[18::nbU]).squeeze(), np.array(sol_U[19::nbU]).squeeze()]

nbNoeuds_phase = [nbNoeuds_stance, nbNoeuds_swing]
T_phase        = [T_stance, T_swing]