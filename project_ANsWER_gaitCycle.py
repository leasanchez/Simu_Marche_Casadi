import biorbd
from ezc3d import c3d
from casadi import *
import matplotlib
import matplotlib.pyplot as plt
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
file        = '/home/leasanchez/programmation/Simu_Marche_Casadi/DonneesMouvement/equincocont01_out.c3d'
kalman_file = '/home/leasanchez/programmation/Simu_Marche_Casadi/DonneesMouvement/equincocont01_out_MOD5000_leftHanded_GenderF_Florent_.Q2'

# ground reaction forces
[GRF_real, T, T_stance] = load_data_GRF(file, nbNoeuds_stance, nbNoeuds_swing, 'cycle')
T_swing                 = T - T_stance                                                                # gait cycle time

# marker
M_real_stance = load_data_markers(file, T_stance, nbNoeuds_stance, nbMarker, 'stance')
M_real_swing  = load_data_markers(file, T_swing, nbNoeuds_swing, nbMarker, 'swing')
M_real        = [np.hstack([M_real_stance[0, :, :], M_real_swing[0, :, :]]), np.hstack([M_real_stance[1, :, :], M_real_swing[1, :, :]]), np.hstack([M_real_stance[2, :, :], M_real_swing[2, :, :]])]

# muscular excitation
U_real_swing  = load_data_emg(file, T_swing, nbNoeuds_swing, nbMus, 'swing')
U_real_stance = load_data_emg(file, T_stance, nbNoeuds_stance, nbMus, 'stance')
U_real        = np.hstack([U_real_stance, U_real_swing])

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
u0[: nbMus, :]   = load_initialguess_muscularExcitation(np.hstack([U_real_stance, U_real_swing]))
#u0[: nbMus, :]   = np.zeros((nbMus, nbNoeuds)) + init_A
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


w0 = vertcat(u0, X0, p0)

# ----------------------------- Solver ---------------------------------------------------------------------------------
w = vertcat(U, X, p)
J = Ja + Je + Jm + JR

class AnimateCallback(casadi.Callback):
    def __init__(self, name, nx, ng, np, opts={}):
        Callback.__init__(self)
        self.name = name               # callback name

        self.nx = nx                   # optimized value number
        self.ng = ng                   # constraints number
        self.np = np                   # parameters number

        # CONTROL
        fig1 = plt.figure(1)
        Labels = ['GLUT_MAX1', 'GLUT_MAX2', 'GLUT_MAX3', 'GLUT_MED1', 'GLUT_MED2', 'GLUT_MED3',
                  'R_SEMIMEM', 'R_SEMITEN', 'R_BI_FEM_LH', 'R_RECTUS_FEM', 'R_VAS_MED', 'R_VAS_INT',
                  'R_VAS_LAT', 'R_GAS_MED', 'R_GAS_LAT', 'R_SOLEUS', 'R_TIB_ANT']
        ax = []
        for nMus in range(nbMus):
            ax.append(fig1.add_subplot(5, 4, nMus + 1))
            title(Labels[nMus])
            plt.plot([0, nbNoeuds], [min_A, min_A], 'k--')  # lower bound
            plt.plot([0, nbNoeuds], [max_A, max_A], 'k--')  # upper bound

        ax.append(fig1.add_subplot(5, 4, nbMus + 1))
        plt.title('Pelvis Tx')
        plt.plot([0, nbNoeuds], [-1000, -1000], 'k--')
        plt.plot([0, nbNoeuds], [1000, 1000], 'k--')

        ax.append(fig1.add_subplot(5, 4, nbMus + 2))
        plt.title('Pelvis Ty')
        plt.plot([0, nbNoeuds], [-2000, -2000], 'k--')
        plt.plot([0, nbNoeuds], [2000, 2000], 'k--')

        ax.append(fig1.add_subplot(5, 4, nbMus + 3))
        plt.title('Pelvis Rz')
        plt.plot([0, nbNoeuds], [-200, -200], 'k--')
        plt.plot([0, nbNoeuds], [200, 200], 'k--')


        # STATES
        fig2 = plt.figure(2)
        Labels_X = ['Pelvis_TX', 'Pelvis_TY', 'Pelvis_RZ', 'Hip', 'Knee', 'Ankle']
        ax2 = []
        for q in range(nbQ):
            ax2.append(fig2.add_subplot(2, 6, q + 1))
            plt.title('Q ' + Labels_X[q])
            plt.plot([0, nbNoeuds], [min_Q, min_Q], 'k--')
            plt.plot([0, nbNoeuds], [max_Q, max_Q], 'k--')

            ax2.append(fig2.add_subplot(2, 6, q + 1 + nbQ))
            plt.title('dQ ' + Labels_X[q])
            plt.plot([0, nbNoeuds], [min_Q, min_Q], 'k--')
            plt.plot([0, nbNoeuds], [max_Q, max_Q], 'k--')

        draw()

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
        # GET CONTROL AND STATES
        for (i, s) in enumerate(nlpsol_out()): darg[s] = arg[i]
        sol_U = darg["x"][:nbU * nbNoeuds]
        sol_X = darg["x"][nbU * nbNoeuds: -nP]

        sol_q  = [sol_X[0::nbX], sol_X[1::nbX], sol_X[2::nbX], sol_X[3::nbX], sol_X[4::nbX], sol_X[5::nbX]]
        sol_dq = [sol_X[6::nbX], sol_X[7::nbX], sol_X[8::nbX], sol_X[9::nbX], sol_X[10::nbX], sol_X[11::nbX]]
        sol_a  = [sol_U[0::nbU], sol_U[1::nbU], sol_U[2::nbU], sol_U[3::nbU], sol_U[4::nbU], sol_U[5::nbU], sol_U[6::nbU], sol_U[7::nbU], sol_U[8::nbU], sol_U[9::nbU], sol_U[10::nbU], sol_U[11::nbU], sol_U[12::nbU], sol_U[13::nbU], sol_U[14::nbU], sol_U[15::nbU],sol_U[16::nbU]]
        sol_F  = [sol_U[17::nbU], sol_U[18::nbU], sol_U[19::nbU]]

        # CONVERGENCE
        JR = 0
        Je = 0
        Jm = 0
        Ja = 0

        # OBJECTIVE FUNCTION
        for k in range(nbNoeuds_stance):
            JR += fcn_objective_GRF(wR, sol_X[nbX * k: nbX * (k + 1)], sol_U[nbU * k: nbU * (k + 1)], GRF_real[:, k])  # Ground Reaction --> stance
            Jm += fcn_objective_markers(wMa, wMt, sol_X[nbX * k: nbX * k + nbQ], M_real_stance[:, :, k],'stance')  # Marker
            Je += fcn_objective_emg(wU, sol_U[nbU * k: nbU * (k + 1)], U_real_stance[:, k])                        # EMG
            Ja += fcn_objective_activation(wL, sol_U[nbU * k: nbU * (k + 1)])                                      # Muscle activations (no EMG)
        for k in range(nbNoeuds_swing):
            Jm += fcn_objective_markers(wMa, wMt, sol_X[nbX * nbNoeuds_stance + nbX * k: nbX * nbNoeuds_stance + nbX * k + nbQ], M_real_swing[:, :, k], 'swing')  # marker
            Je += fcn_objective_emg(wU, sol_U[nbU * nbNoeuds_stance + nbU * k: nbU * nbNoeuds_stance + nbU * (k + 1)], U_real_swing[:, k])  # emg
            Ja += fcn_objective_activation(wL, sol_U[nbU * nbNoeuds_stance + nbU * k: nbU * nbNoeuds_stance + nbU * (k + 1)])
        J = Ja + Je + Jm + JR

        print('Global                 : ' + str(J))
        print('activation             : ' + str(Ja))
        print('emg                    : ' + str(Je))
        print('marker                 : ' + str(Jm))
        print('ground reaction forces : ' + str(JR))

        def plot_control(t, x):
            nbPoints = len(np.array(x))
            for n in range(nbPoints - 1):
                plt.plot([t[n], t[n + 1], t[n + 1]], [x[n], x[n], x[n + 1]], 'b')

        # CONTROL
        # TIME
        t_stance = np.linspace(0, T_stance, nbNoeuds_stance)
        t_swing = t_stance[-1] + np.linspace(0, T_swing, nbNoeuds_swing)
        t = np.hstack([t_stance, t_swing])

        if hasattr(self, 'mus0'):
            # MUSCULAR ACTIVATION
            self.mus0[0].set_xdata(t)
            self.mus0[0].set_ydata(sol_a[0])
            self.mus1[0].set_xdata(t)
            self.mus1[0].set_ydata(sol_a[1])
            self.mus2[0].set_xdata(t)
            self.mus2[0].set_ydata(sol_a[2])
            self.mus3[0].set_xdata(t)
            self.mus3[0].set_ydata(sol_a[3])
            self.mus4[0].set_xdata(t)
            self.mus4[0].set_ydata(sol_a[4])
            self.mus5[0].set_xdata(t)
            self.mus5[0].set_ydata(sol_a[5])
            self.mus6[0].set_xdata(t)
            self.mus6[0].set_ydata(sol_a[6])
            self.mus7[0].set_xdata(t)
            self.mus7[0].set_ydata(sol_a[7])
            self.mus8[0].set_xdata(t)
            self.mus8[0].set_ydata(sol_a[8])
            self.mus9[0].set_xdata(t)
            self.mus9[0].set_ydata(sol_a[9])
            self.mus10[0].set_xdata(t)
            self.mus10[0].set_ydata(sol_a[10])
            self.mus11[0].set_xdata(t)
            self.mus11[0].set_ydata(sol_a[11])
            self.mus12[0].set_xdata(t)
            self.mus12[0].set_ydata(sol_a[12])
            self.mus13[0].set_xdata(t)
            self.mus13[0].set_ydata(sol_a[13])
            self.mus14[0].set_xdata(t)
            self.mus14[0].set_ydata(sol_q[14])
            self.mus15[0].set_xdata(t)
            self.mus15[0].set_ydata(sol_q[15])
            self.mus16[0].set_xdata(t)
            self.mus16[0].set_ydata(sol_q[16])

            # PELVIS FORCES
            self.f0[0].set_xdata(t)
            self.f0[0].set_ydata(sol_F[0])
            self.f1[0].set_xdata(t)
            self.f1[0].set_ydata(sol_F[1])
            self.f2[0].set_xdata(t)
            self.f2[0].set_ydata(sol_F[2])

        else:
            self.mus0 = ax[0].plot_control(t, sol_a[0])
            plt.subplot(5, 4, 2)
            self.mus1 = plot_control(t, sol_a[1])
            plt.subplot(5, 4, 3)
            self.mus2 = plot_control(t, sol_a[2])
            plt.subplot(5, 4, 4)
            self.mus3 = plot_control(t, sol_a[3])
            plt.subplot(5, 4, 5)
            self.mus4 = plot_control(t, sol_a[4])
            plt.subplot(5, 4, 6)
            self.mus5 = plot_control(t, sol_a[5])
            plt.subplot(5, 4, 7)
            self.mus6 = plot_control(t, sol_a[6])
            plt.subplot(5, 4, 8)
            self.mus7 = plot_control(t, sol_a[7])
            plt.subplot(5, 4, 9)
            self.mus8 = plot_control(t, sol_a[8])
            plt.subplot(5, 4, 10)
            self.mus9 = plot_control(t, sol_a[9])
            plt.subplot(5, 4, 11)
            self.mus10 = plot_control(t, sol_a[10])
            plt.subplot(5, 4, 12)
            self.mus11 = plot_control(t, sol_a[11])
            plt.subplot(5, 4, 13)
            self.mus12 = plot_control(t, sol_a[12])
            plt.subplot(5, 4, 14)
            self.mus13 = plot_control(t, sol_a[13])
            plt.subplot(5, 4, 15)
            self.mus14 = plot_control(t, sol_a[14])
            plt.subplot(5, 4, 16)
            self.mus15 = plot_control(t, sol_a[15])
            plt.subplot(5, 4, 17)
            self.mus16 = plot_control(t, sol_a[16])

            plt.subplot(5, 4, 18)
            self.f0 = plot_control(t, sol_F[0])
            plt.subplot(5, 4, 19)
            self.f0 = plot_control(t, sol_F[1])
            plt.subplot(5, 4, 20)
            self.f0 = plot_control(t, sol_F[2])

        # STATE
        plt.figure(3)
        # TIME
        t = np.hstack([t_stance, t_swing, t_swing[-1] + (t_swing[-1] - t_swing[-2])])

        if hasattr(self, 'q0'):
            # JOINT POSITION
            self.q0[0].set_xdata(t)
            self.q0[0].set_ydata(sol_q[0])
            self.q1[0].set_xdata(t)
            self.q1[0].set_ydata(sol_q[1])
            self.q2[0].set_xdata(t)
            self.q2[0].set_ydata(sol_q[2])
            self.q3[0].set_xdata(t)
            self.q3[0].set_ydata(sol_q[3])
            self.q4[0].set_xdata(t)
            self.q4[0].set_ydata(sol_q[4])
            self.q5[0].set_xdata(t)
            self.q5[0].set_ydata(sol_q[5])

            # JOINT SPEED
            self.dq0[0].set_xdata(t)
            self.dq0[0].set_ydata(sol_dq[0])
            self.dq1[0].set_xdata(t)
            self.dq1[0].set_ydata(sol_dq[1])
            self.dq2[0].set_xdata(t)
            self.dq2[0].set_ydata(sol_dq[2])
            self.dq3[0].set_xdata(t)
            self.dq3[0].set_ydata(sol_dq[3])
            self.dq4[0].set_xdata(t)
            self.dq4[0].set_ydata(sol_dq[4])
            self.dq5[0].set_xdata(t)
            self.dq5[0].set_ydata(sol_dq[5])

        else:
            # JOINT POSITION
            plt.subplot(2, 6, 1)
            self.q0 = plt.plot(t, sol_q[0], 'r')
            plt.subplot(2, 6, 2)
            self.q1 = plt.plot(t, sol_q[1], 'r')
            plt.subplot(2, 6, 3)
            self.q2 = plt.plot(t, sol_q[2], 'r')
            plt.subplot(2, 6, 4)
            self.q3 = plt.plot(t, sol_q[3], 'r')
            plt.subplot(2, 6, 5)
            self.q4 = plt.plot(t, sol_q[4], 'r')
            plt.subplot(2, 6, 6)
            self.q5 = plt.plot(t, sol_q[5], 'r')

            # JOINT SPEED
            plt.subplot(2, 6, 7)
            self.dq0 = plt.plot(t, sol_dq[0], 'r')
            plt.subplot(2, 6, 8)
            self.dq1 = plt.plot(t, sol_dq[1], 'r')
            plt.subplot(2, 6, 9)
            self.dq2 = plt.plot(t, sol_dq[2], 'r')
            plt.subplot(2, 6, 10)
            self.dq3 = plt.plot(t, sol_dq[3], 'r')
            plt.subplot(2, 6, 11)
            self.dq4 = plt.plot(t, sol_dq[4], 'r')
            plt.subplot(2, 6, 12)
            self.dq5 = plt.plot(t, sol_dq[5], 'r')

        plt.draw()
        time.sleep(0.25)
        return [0]

nlp = {'x': w, 'f': J, 'g': vertcat(*G)}
callback = AnimateCallback('callback', (nbU*nbNoeuds + nbX*(nbNoeuds + 1) + nP), nbX*nbNoeuds, 0)
opts = {"ipopt.tol": 1e-1, "ipopt.linear_solver": "ma57", "ipopt.hessian_approximation":"limited-memory", "iteration_callback": callback}
solver = nlpsol("solver", "ipopt", nlp, opts)

start_opti = time.time()
print('Start optimisation : ' + str(start_opti))
res = solver(lbg = lbg,
             ubg = ubg,
             lbx = lbx,
             ubx = ubx,
             x0  = w0)

plt.interactive(False)
plt.show()

# RESULTS
stop_opti = time.time() - start_opti
print('Time to solve : ' + str(stop_opti))


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