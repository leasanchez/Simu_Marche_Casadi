import _thread
import biorbd
from casadi import *
from matplotlib import pyplot as plt
import numpy as np

from LoadData import load_data_markers, load_data_emg, load_data_GRF
from Fcn_InitialGuess import load_initialguess_muscularExcitation, load_initialguess_q
from Marche_Fcn_Integration import int_RK4_swing, int_RK4_stance
from Fcn_Objective import fcn_objective_activation, fcn_objective_emg, fcn_objective_markers, fcn_objective_GRF
# from Fcn_print_data import save_GRF_real, save_Markers_real, save_EMG_real, save_params, save_bounds, save_initialguess

# SET MODELS
model_stance_dir = '/home/leasanchez/programmation/Simu_Marche_Casadi/ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact.bioMod'
model_stance      = biorbd.Model(model_stance_dir)

model_swing_dir = '/home/leasanchez/programmation/Simu_Marche_Casadi/ModelesS2M/ANsWER_Rleg_6dof_17muscle_0contact.bioMod'
model_swing     = biorbd.Model(model_swing_dir)

# ----------------------------- Probleme -------------------------------------------------------------------------------
nbNoeuds_stance = 10                                   # shooting points for stance phase
nbNoeuds_swing  = 10                                   # shooting points for swing phase
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
# save_GRF_real(name_subject, save_dir, GRF_real)


# marker
M_real_stance = load_data_markers(file, T_stance, nbNoeuds_stance, nbMarker, 'stance')
M_real_swing  = load_data_markers(file, T_swing, nbNoeuds_swing, nbMarker, 'swing')

M_real          = np.zeros((3, nbMarker, (nbNoeuds + 1)))
M_real[0, :, :] = np.hstack([M_real_stance[0, :, :], M_real_swing[0, :, :]])
M_real[1, :, :] = np.hstack([M_real_stance[1, :, :], M_real_swing[1, :, :]])
M_real[2, :, :] = np.hstack([M_real_stance[2, :, :], M_real_swing[2, :, :]])
# save_Markers_real(name_subject, save_dir, M_real)

# muscular excitation
U_real_swing  = load_data_emg(file, T_swing, nbNoeuds_swing, nbMus, 'swing')
U_real_stance = load_data_emg(file, T_stance, nbNoeuds_stance, nbMus, 'stance')
U_real        = np.hstack([U_real_stance, U_real_swing])
# save_EMG_real(name_subject, save_dir, U_real)

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

class Parameters():
    def __init__(self):
        # MODEL
        self.model_stance_dir = model_stance_dir
        self.model_swing_dir = model_swing_dir

        self.model_stance = model_stance
        self.model_swing  = model_swing

        # PROBLEME
        self.nbNoeuds_stance = 10                                   # shooting points for stance phase
        self.nbNoeuds_swing  = 10                                   # shooting points for swing phase
        self.nbNoeuds        = nbNoeuds_stance + nbNoeuds_swing     # total shooting points

        self.T_stance = T_stance                                    # stance phase duration
        self.T_swing  = T_swing                                     # swing phase duration
        self.T        = T                                           # gait cycle duration

        self.nbMus     = model_stance.nbMuscleTotal()               # number of muscles
        self.nbQ       = model_stance.nbDof()                       # number of DoFs
        self.nbMarker  = model_stance.nbMarkers()                   # number of markers
        self.nbBody    = model_stance.nbSegment()                   # number of body segments
        self.nbContact = model_stance.nbContacts()                  # number of contact (2 forces --> plan)

        self.nbU      = nbMus + 3                                   # number of controls : muscle activation + articular torque
        self.nbX      = 2*nbQ                                       # number of states : generalized positions + velocities
        self.nP       = nbMus + 1                                   # number of parameters : 1 global + muscles

        # WEIGHTING FACTORS
        self.wL  = 1                                                # activation
        self.wMa = 30                                               # anatomical marker
        self.wMt = 50                                               # technical marker
        self.wU  = 1                                                # excitation
        self.wR  = 0.05 # 30                                        # ground reaction

params = Parameters()
# save_params(name_subject, save_dir, params)

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

# save_bounds(name_subject, save_dir, lbx, ubx, params)

# ----------------------------- Initial guess --------------------------------------------------------------------------
init_A = 0.1

# CONTROL
u0               = np.zeros((nbU, nbNoeuds))
u0[: nbMus, :]   = load_initialguess_muscularExcitation(np.hstack([U_real_stance, U_real_swing]))
# u0[: nbMus, :]   = np.zeros((nbMus, nbNoeuds)) + init_A
u0[nbMus + 0, :] = [0]*nbNoeuds                                  # pelvis forces
u0[nbMus + 1, :] = [-500]*nbNoeuds_stance + [0]*nbNoeuds_swing
u0[nbMus + 2, :] = [0]*nbNoeuds

# STATE
q0_stance = load_initialguess_q(file, kalman_file, T_stance, nbNoeuds_stance, 'stance')
q0_swing  = load_initialguess_q(file, kalman_file, T_swing, nbNoeuds_swing, 'swing')
q0        = np.hstack([q0_stance, q0_swing])
dq0       = np.zeros((nbQ, (nbNoeuds + 1)))

X0                = np.zeros((nbX, (nbNoeuds + 1)))
X0[:nbQ, :]       = q0
X0[nbQ: 2*nbQ, :] = dq0

# PARAMETERS
p0 = [1] + [1]*nbMus
# save_initialguess(name_subject, save_dir, u0, X0, p0)

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
        return [0]

mycallback = AnimateCallback('mycallback', (nbU*nbNoeuds + nbX*(nbNoeuds + 1) + nP), nbX*nbNoeuds, 0)



def print_callback(callback_data):
    print('NEW THREAD')
    fig = plt.figure()
    plot_handler = plt.plot(np.random.rand(), np.random.rand(), 'x')
    plt.show(block=False)
    print('FIGURE')

    while True:
        if callback_data.update_sol:
            print('NEW DATA \n')
            # fig = plt.figure()
            # plot_handler = plt.plot(np.random.rand(), np.random.rand(), 'x')
            # plt.show(block=False)
            # plt.pause(0.01)

            callback_data.update_sol = False

        # plt.draw()
        plt.pause(0.01)

_thread.start_new_thread(print_callback, (mycallback,))          # nouveau thread

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