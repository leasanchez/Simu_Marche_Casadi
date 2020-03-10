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
name_subject = 'equincocont01'
file         = '/home/leasanchez/programmation/Simu_Marche_Casadi/DonneesMouvement/' + name_subject + '_out.c3d'
kalman_file  = '/home/leasanchez/programmation/Simu_Marche_Casadi/DonneesMouvement/' + name_subject + '_out_MOD5000_leftHanded_GenderF_Florent_.Q2'

save_dir = '/home/leasanchez/programmation/Simu_Marche_Casadi/Resultats/' + name_subject + '/'


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

# sauvegarde valeurs comparées mesurées

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

# sauvegarde weighting factor


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
    [grf, Jr] = fcn_objective_GRF(wR, X[nbX*k: nbX*(k + 1)], Uk, GRF_real[:, k])                        # Ground Reaction --> stance  # Ground Reaction --> stance
    JR += Jr
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

# sauvegarde contraintes

# ----------------------------- Initial guess --------------------------------------------------------------------------
init_A = 0.1

# CONTROL
u0               = np.zeros((nbU, nbNoeuds))
u0[: nbMus, :]   = load_initialguess_muscularExcitation(np.hstack([U_real_stance, U_real_swing]))
# u0[: nbMus, :]   = np.zeros((nbMus, nbNoeuds)) + init_A
u0[nbMus + 0, :] = [0]*nbNoeuds                                  # pelvis forces
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
    def __init__(self, name, nx, ng, nP, opts={}):
        Callback.__init__(self)
        self.name = name               # callback name

        self.nx = nx                   # optimized value number
        self.ng = ng                   # constraints number
        self.nP = nP                   # parameters number
        self.update = 0                # first iteration?

        # CONTROL
        fig1, axes1 = plt.subplots(5, 4, sharex=True, figsize=(10,10))
        Labels = ['GLUT_MAX1', 'GLUT_MAX2', 'GLUT_MAX3', 'GLUT_MED1', 'GLUT_MED2', 'GLUT_MED3',
                  'R_SEMIMEM', 'R_SEMITEN', 'R_BI_FEM_LH', 'R_RECTUS_FEM', 'R_VAS_MED', 'R_VAS_INT',
                  'R_VAS_LAT', 'R_GAS_MED', 'R_GAS_LAT', 'R_SOLEUS', 'R_TIB_ANT', 'Pelvis Tx', 'Pelvis Ty', 'Pelvis Rz']
        lower_bound = [min_A] * nbMus + [-1000] + [-2000] + [-200]
        upper_bound = [max_A] * nbMus + [1000]  + [2000]  + [200]

        # TIME
        t_stance = np.linspace(0, T_stance, nbNoeuds_stance)
        t_swing = t_stance[-1] + np.linspace(0, T_swing, nbNoeuds_swing)
        t = np.hstack([t_stance, t_swing])

        axes1 = axes1.flatten()
        u_emg = 9
        for i in range(nbU):
            ax = axes1[i]
            ax.set_title(Labels[i])
            ax.plot([0, T], [lowerbound_u[i], lowerbound_u[i]], 'k--')               # lower bound
            ax.plot([0, T], [upperbound_u[i], upperbound_u[i]], 'k--')               # upper bound
            ax.plot([T_stance, T_stance], [lowerbound_u[i], upperbound_u[i]], 'k:')  # end of the stance phase
            ax.grid(True)
            if (i != 1) and (i != 2) and (i != 3) and (i != 5) and (i != 6) and (i != 11) and (i != 12) and (i < (nbMus - 1)):
                ax.plot(t, U_real[u_emg, :], 'r')
                u_emg -= 1
            if (i < (nbMus - 1)):
                ax.yaxis.set_ticks(np.arange(0, 1, 0.5))
            if (i > nbU - 5):
                ax.set_xlabel('time (s)')


        # STATES
        fig2, axes2 = plt.subplots(2,6, sharex=True, figsize=(20,10))
        axes2 = axes2.flatten()

        Labels_X = ['Pelvis_TX', 'Pelvis_TY', 'Pelvis_RZ', 'Hip', 'Knee', 'Ankle']

        for q in range(nbQ):
            ax1 = axes2[q]
            ax1.set_title('Q ' + Labels_X[q])
            if q!= 0 and q!= 1:
                ax1.plot([0, T], [lowerbound_x[q] * (180/np.pi), lowerbound_x[q] * (180/np.pi)], 'k--')
                ax1.plot([0, T], [upperbound_x[q] * (180/np.pi), upperbound_x[q] * (180/np.pi)], 'k--')
                ax1.plot([T_stance, T_stance], [lowerbound_x[q] * (180/np.pi), upperbound_x[q] * (180/np.pi)], 'k:')  # end of the stance phase
            else:
                ax1.plot([0, T], [lowerbound_x[q], lowerbound_x[q]], 'k--')
                ax1.plot([0, T], [upperbound_x[q], upperbound_x[q]], 'k--')
                ax1.plot([T_stance, T_stance], [lowerbound_x[q], upperbound_x[q]], 'k:')  # end of the stance phase

            ax1.grid(True)

            ax2 = axes2[q + nbQ]
            ax2.set_title('dQ ' + Labels_X[q])
            ax2.plot([0, T], [lowerbound_x[q], lowerbound_x[q]], 'k--')
            ax2.plot([0, T], [upperbound_x[q], upperbound_x[q]], 'k--')
            ax2.plot([T_stance, T_stance], [lowerbound_x[q], upperbound_x[q]], 'k:')  # end of the stance phase
            ax2.set_xlabel('time (s)')
            ax2.grid(True)

        # GROUND REACTION FORCES
        fig3, axes3 = plt.subplots(2, 1, sharex=True)
        axes3.flatten()

        ax_ap = axes3[0]
        ax_ap.set_title('GRF A/P  during the gait')
        ax_ap.plot(t, GRF_real[1, :], 'r')
        ax_ap.plot([T_stance, T_stance], [min(GRF_real[1, :]), max(GRF_real[1, :])], 'k:')  # end of the stance phase
        ax_ap.grid(True)

        ax_v  = axes3[1]
        ax_v.set_title('GRF vertical')
        ax_v.plot(t, GRF_real[2, :], 'r')
        ax_v.plot([T_stance, T_stance], [min(GRF_real[2, :]), max(GRF_real[2, :])], 'k:')
        ax_v.set_xlabel('time (s)')
        ax_v.grid(True)
        fig3.tight_layout()

        plt.draw()
        plt.interactive(True)
        plt.show()
        time.sleep(0.25)

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

        sol_q  = np.hstack([sol_X[0::nbX], sol_X[1::nbX], sol_X[2::nbX], sol_X[3::nbX], sol_X[4::nbX], sol_X[5::nbX]])
        sol_dq = np.hstack([sol_X[6::nbX], sol_X[7::nbX], sol_X[8::nbX], sol_X[9::nbX], sol_X[10::nbX], sol_X[11::nbX]])
        sol_a  = np.hstack([sol_U[0::nbU], sol_U[1::nbU], sol_U[2::nbU], sol_U[3::nbU], sol_U[4::nbU], sol_U[5::nbU], sol_U[6::nbU], sol_U[7::nbU], sol_U[8::nbU], sol_U[9::nbU], sol_U[10::nbU], sol_U[11::nbU], sol_U[12::nbU], sol_U[13::nbU], sol_U[14::nbU], sol_U[15::nbU],sol_U[16::nbU]])
        sol_F  = np.hstack([sol_U[17::nbU], sol_U[18::nbU], sol_U[19::nbU]])

        # CONVERGENCE
        JR = Je = Jm = Ja = constraint = 0

        # OBJECTIVE FUNCTION
        GRF = np.zeros((3, nbNoeuds))
        for k in range(nbNoeuds_stance):
            [F, Jr] = fcn_objective_GRF(wR, sol_X[nbX * k: nbX * (k + 1)], sol_U[nbU * k: nbU * (k + 1)], GRF_real[:, k])  # Ground Reaction --> stance
            GRF[:, k] = np.array(F).squeeze()
            JR += Jr
            Jm += fcn_objective_markers(wMa, wMt, sol_X[nbX * k: nbX * k + nbQ], M_real_stance[:, :, k],'stance')      # Marker
            Je += fcn_objective_emg(wU, sol_U[nbU * k: nbU * (k + 1)], U_real_stance[:, k])                            # EMG
            Ja += fcn_objective_activation(wL, sol_U[nbU * k: nbU * (k + 1)])                                          # Muscle activations (no EMG)

            constraint += sol_X[nbX * (k + 1): nbX * (k + 2)] - int_RK4_stance(T_stance, nbNoeuds_stance, nkutta, sol_X[nbX * k: nbX * (k + 1)], sol_U[nbU * k: nbU * (k + 1)])

        for k in range(nbNoeuds_swing):
            Jm += fcn_objective_markers(wMa, wMt, sol_X[nbX * nbNoeuds_stance + nbX * k: nbX * nbNoeuds_stance + nbX * k + nbQ], M_real_swing[:, :, k], 'swing')
            Je += fcn_objective_emg(wU, sol_U[nbU * nbNoeuds_stance + nbU * k: nbU * nbNoeuds_stance + nbU * (k + 1)], U_real_swing[:, k])
            Ja += fcn_objective_activation(wL, sol_U[nbU * nbNoeuds_stance + nbU * k: nbU * nbNoeuds_stance + nbU * (k + 1)])

            constraint += sol_X[nbX * nbNoeuds_stance + nbX*(k + 1): nbX*nbNoeuds_stance + nbX*(k + 2)] - int_RK4_swing(T_swing, nbNoeuds_swing, nkutta,  sol_X[nbX*nbNoeuds_stance + nbX*k: nbX*nbNoeuds_stance + nbX*(k + 1)], sol_U[nbU * nbNoeuds_stance + nbU * k: nbU * nbNoeuds_stance + nbU * (k + 1)])

        J = Ja + Je + Jm + JR

        print('\n \nGlobal                 : ' + str(J))
        print('activation             : ' + str(Ja))
        print('emg                    : ' + str(Je))
        print('marker                 : ' + str(Jm))
        print('ground reaction forces : ' + str(JR))
        print('constraints            : ' + str(sum(constraint)) + '\n')

        def plot_control(ax, t, x):
            nbPoints = len(np.array(x))
            for n in range(nbPoints - 1):
                ax.plot([t[n], t[n + 1], t[n + 1]], [x[n], x[n], x[n + 1]], 'b')

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


        if (self.update == 1):
            # CONTROL
            # TIME
            t_stance = np.linspace(0, T_stance, nbNoeuds_stance)
            t_swing = t_stance[-1] + np.linspace(0, T_swing, nbNoeuds_swing)
            t = np.hstack([t_stance, t_swing])

            axes1 = plt.figure(1).axes
            for i in range(nbMus):
                ax = axes1[i]
                plot_control_update(ax, t, sol_a[:, i])
            for j in range(3):
                ax = axes1[i + 1 + j]
                plot_control_update(ax, t, sol_F[:, j])

            plt.savefig(save_dir + 'plot_control')

            # STATE
            axes2 = plt.figure(2).axes
            # ADJUSTED TIME
            ts = np.hstack([t_stance, t_swing, t_swing[-1] + (t_swing[-1] - t_swing[-2])])
            for q in range(nbQ):
                ax1 = axes2[q]
                lines = ax1.get_lines()
                if q != 0 and q != 1:
                    lines[3].set_ydata(sol_q[:, q] * (180/np.pi))
                else:
                    lines[3].set_ydata(sol_q[:, q])

            for dq in range(nbQ):
                ax1 = axes2[q + 1 + dq]
                lines = ax1.get_lines()
                lines[3].set_ydata(sol_dq[:, dq])

            plt.savefig(save_dir + 'plot_state')

            # GRF
            axes3 = plt.figure(3).axes
            ax_ap = axes3[0]
            lines = ax_ap.get_lines()
            lines[2].set_ydata(GRF[0, :])

            ax_v = axes3[1]
            lines = ax_v.get_lines()
            lines[2].set_ydata(GRF[2, :])
            plt.savefig(save_dir + 'plot_GRF')

        else:
            # CONTROL
            # TIME
            t_stance = np.linspace(0, T_stance, nbNoeuds_stance)
            t_swing = t_stance[-1] + np.linspace(0, T_swing, nbNoeuds_swing)
            t = np.hstack([t_stance, t_swing])

            axes1 = plt.figure(1).axes
            for i in range(nbMus):
                ax = axes1[i]
                plot_control(ax, t, sol_a[:, i])
            for j in range(3):
                ax = axes1[i + 1 + j]
                plot_control(ax, t, sol_F[:, j])

            plt.savefig(save_dir + 'plot_control')

            # STATE
            axes2 = plt.figure(2).axes
            # ADJUSTED TIME
            ts = np.hstack([t_stance, t_swing, t_swing[-1] + (t_swing[-1] - t_swing[-2])])
            for q in range(nbQ):
                ax1 = axes2[q]
                if q!=0 and q!= 1:
                    ax1.plot(ts, sol_q[:, q] * (180/np.pi), 'b')
                else:
                    ax1.plot(ts, sol_q[:, q], 'b')
            for dq in range(nbQ):
                ax1 = axes2[q + 1 + dq]
                ax1.plot(ts, sol_dq[:, dq], 'b')
            plt.savefig(save_dir + 'plot_state')

            # GRF
            axes3 = plt.figure(3).axes
            ax_ap = axes3[0]
            ax_ap.plot(t, GRF[0, :], 'b')

            ax_v = axes3[1]
            ax_v.plot(t, GRF[2, :], 'b')
            plt.savefig(save_dir + 'plot_GRF')
            self.update = 1

        plt.interactive(True)
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