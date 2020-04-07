from casadi import *
from matplotlib import pyplot as plt
import numpy as np

# add classes
from Define_parameters import Parameters
from Fcn_Objective import Fcn_Objective
from Fcn_forward_dynamic import Dynamics
from Fcn_plot_Results import Plot_results

# add fcn
import LoadData
from Fcn_InitialGuess import load_initialguess_q
from Marche_Fcn_Integration import int_RK4


# SET PARAMETERS
params = Parameters()

# ----------------------------- States & controls ----------------------------------------------------------------------
# CONTROL
u           = MX.sym("u", params.nbU)
activation  = u[:params.nbMus]                                # muscular activation
torque      = u[params.nbMus:]                                # articular torque

# PARAMETERS
p  = MX.sym("p", params.nP)                                   # maximal isometric force adjustment

# STATE
x  = MX.sym("x", params.nbX)
q  = x[:params.nbQ]                                           # generalized coordinates
dq = x[params.nbQ: 2 * params.nbQ]                            # velocities

# ----------------------------- Define casadi function -----------------------------------------------------------------
ffcn_no_contact = casadi.Function("ffcn_no_contact",
                                   [x, u, p],
                                   [Dynamics.ffcn_no_contact(x, u, p)],
                                   ["states", "controls", "parameters"],
                                   ["statesdot"]).expand()

# ----------------------------- Load Data from c3d file ----------------------------------------------------------------
[GRF_real_swing, params.T, params.T_stance, params.T_swing] = LoadData.load_data_GRF(params, 'swing')                   # GROUND REACTION FORCES & SET TIME
M_real_swing  = LoadData.load_data_markers(params, 'swing')                                                             # MARKERS POSITION
U_real_swing  = LoadData.load_data_emg(params, 'swing')                                                                 # MUSCULAR EXCITATION

# ----------------------------- Movement -------------------------------------------------------------------------------
U = MX.sym("U", params.nbU * params.nbNoeuds_swing)          # controls
X = MX.sym("X", params.nbX * (params.nbNoeuds_swing + 1))    # states
P = MX.sym("P", params.nP)                                   # parameters
G = []                                                 # equality constraints
Ja = 0                                                 # objective function for muscle activation
Jm = 0                                                 # objective function for markers
Je = 0                                                 # objective function for EMG
Jt = 0                                                 # min residual torque


# ------------ PHASE 2 : Swing phase
for k in range(params.nbNoeuds_swing):
    # DYNAMIQUE
    Uk = U[params.nbU*k: params.nbU*(k + 1)]
    Xk = X[params.nbX*k: params.nbX*(k + 1)]
    G.append(X[params.nbX*(k + 1): params.nbX*(k + 2)] - int_RK4(ffcn_no_contact, params, Xk, Uk, P))

    # OBJECTIVE FUNCTION
    Jm += Fcn_Objective.fcn_objective_markers(params.wMa, params.wMt, Xk[: params.nbQ], M_real_swing[:, :, k], 'swing') # tracking marker
    Je += Fcn_Objective.fcn_objective_emg(params.wU, Uk[:params.nbMus], U_real_swing[:, k])                             # tracking emg
    Ja += Fcn_Objective.fcn_objective_activation(params.wL, Uk[:params.nbMus])                                          # min muscular activation
    Jt += Fcn_Objective.fcn_objective_residualtorque(params.wt, Uk[params.nbMus:])                                      # min residual torques

# ----------------------------- Contraintes ----------------------------------------------------------------------------
# égalité
lbg = [0] * params.nbX * params.nbNoeuds_swing
ubg = [0] * params.nbX * params.nbNoeuds_swing

# ----------------------------- Bounds on w ----------------------------------------------------------------------------
# activation - excitation musculaire
min_A = 1e-3                        # 0 exclusif
max_A = 1
lowerbound_u = [min_A] * params.nbMus + [-1000]*params.nbQ
upperbound_u = [max_A] * params.nbMus + [1000]*params.nbQ
lbu = (lowerbound_u) * params.nbNoeuds_swing
ubu = (upperbound_u) * params.nbNoeuds_swing

# q et dq
lowerbound_x = [-10, -0.5, -np.pi/4, -np.pi/9, -np.pi/2, -np.pi/3, 0.5, -0.5, -1.7453, -5.2360, -5.2360, -5.2360]
upperbound_x = [10, 1.5, np.pi/4, np.pi/3, 0.0873, np.pi/9, 1.5, 0.5, 1.7453, 5.2360, 5.2360, 5.2360]
lbX   = (lowerbound_x) * (params.nbNoeuds_swing + 1)
ubX   = (upperbound_x) * (params.nbNoeuds_swing + 1)

# parameters
min_p  = 0.2
max_p  = 5
lbp = [min_p] * params.nbMus
ubp = [max_p] * params.nbMus


lbx = vertcat(lbu, lbX, lbp)
ubx = vertcat(ubu, ubX, ubp)

# ----------------------------- Initial guess --------------------------------------------------------------------------
# CONTROL
u0                    = np.zeros((params.nbU, params.nbNoeuds_swing))
u0[: params.nbMus, :] = np.zeros((params.nbMus, params.nbNoeuds_swing)) + 0.1       # muscular activations
u0[params.nbMus:]     = np.zeros((params.nbQ, params.nbNoeuds_swing))               # residual torques

# STATE
q0  = load_initialguess_q(params, 'swing')
dq0 = np.gradient(q0)
dq0 = dq0[0]

X0                 = np.zeros((params.nbX, (params.nbNoeuds_swing + 1)))
X0[:params.nbQ, :] = q0
X0[params.nbQ: 2 * params.nbQ, :] = dq0

# PARAMETERS
p0 = [1] * params.nbMus

w0 = vertcat(vertcat(*u0.T), vertcat(*X0.T), p0)

# ----------------------------- Solver ---------------------------------------------------------------------------------
w = vertcat(U, X, P)
J = Ja + Je + Jm + Jt

nlp    = {'x': w, 'f': J, 'g': vertcat(*G)}
opts   = {"ipopt.tol": 1e-2, "ipopt.linear_solver": "ma57"}
solver = nlpsol("solver", "ipopt", nlp, opts)

res = solver(lbg = lbg,
             ubg = ubg,
             lbx = lbx,
             ubx = ubx,
             x0  = w0)


# RESULTS
sol_U  = res["x"][:params.nbU * params.nbNoeuds_swing]
sol_X  = res["x"][params.nbU * params.nbNoeuds_swing: -params.nP]
sol_p  = res["x"][-params.nP:]

# save txt file
file = '/home/leasanchez/programmation/Simu_Marche_Casadi/Resultats/equincocont01/RES/equincocont01_sol_swing.txt'
f = open(file, 'a')
f.write('STATE\n\n')
np.savetxt(f, sol_X, delimiter = '\n')
f.write('\n\nCONTROL\n\n')
np.savetxt(f, sol_U, delimiter = '\n')
f.write('\n\nPARAMETER\n\n')
np.savetxt(f, sol_p, delimiter = '\n')
f.close()