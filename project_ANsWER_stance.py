from casadi import *
from matplotlib import pyplot as plt
import numpy as np

# add classes
from Define_parameters import Parameters
from Define_casadi_callback import AnimateCallback

# add fcn
from LoadData import load_data_markers, load_data_emg, load_data_GRF
from Fcn_InitialGuess import load_initialguess_muscularExcitation, load_initialguess_q
from Marche_Fcn_Integration import int_RK4_swing, int_RK4_stance, int_RK4
from Fcn_forward_dynamic import ffcn_contact, ffcn_no_contact
from Fcn_Objective import fcn_objective_activation, fcn_objective_emg, fcn_objective_markers, fcn_objective_GRF
from Fcn_print_data import save_GRF_real, save_Markers_real, save_EMG_real, save_params, save_bounds, save_initialguess

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


# ----------------------------- Load Data from c3d file ----------------------------------------------------------------
# GROUND REACTION FORCES & SET TIME
[GRF_real, params.T, params.T_stance, params.T_swing] = load_data_GRF(params, 'cycle')

# MARKERS POSITION
M_real_stance = load_data_markers(params, 'stance')

# MUSCULAR EXCITATION
U_real_stance = load_data_emg(params, 'stance')


# ----------------------------- Movement -------------------------------------------------------------------------------
U = MX.sym("U", params.nbU * params.nbNoeuds_stance)          # controls
X = MX.sym("X", params.nbX * (params.nbNoeuds_stance + 1))    # states
P = MX.sym("P", params.nP)                             # parameters
G = []                                                 # equality constraints
Ja = 0                                                 # objective function for muscle activation
Jm = 0                                                 # objective function for markers
Je = 0                                                 # objective function for EMG
JR = 0                                                 # objective function for ground reactions

# ------------ PHASE 1 : Stance phase
for k in range(params.nbNoeuds_stance):
    # DYNAMIQUE
    Uk = U[params.nbU*k: params.nbU*(k + 1)]
    Xk = X[params.nbX*k: params.nbX*(k + 1)]
    G.append(X[params.nbX * (k + 1): params.nbX * (k + 2)] - int_RK4(ffcn_contact, params, Xk, Uk, P))

    # OBJECTIVE FUNCTION
    [grf, Jr] = fcn_objective_GRF(params.wR, Xk, Uk, GRF_real[:, k])                                                    # tracking ground reaction --> stance
    JR += Jr
    Jm += fcn_objective_markers(params.wMa, params.wMt, Xk[: params.nbQ], M_real_stance[:, :, k], 'stance')             # tracking marker
    Je += fcn_objective_emg(params.wU, Uk, U_real_stance[:, k])                                                         # tracking emg
    Ja += fcn_objective_activation(params.wL, Uk)                                                                       # min muscle activations (no EMG)


# ----------------------------- Contraintes ----------------------------------------------------------------------------
# égalité
lbg = [0] * params.nbX * params.nbNoeuds_stance
ubg = [0] * params.nbX * params.nbNoeuds_stance

# ----------------------------- Bounds on w ----------------------------------------------------------------------------
# activation - excitation musculaire
min_A = 1e-3                        # 0 exclusif
max_A = 1
lowerbound_u = [min_A] * params.nbMus + [-1000] + [-2000] + [-200]
upperbound_u = [max_A] * params.nbMus + [1000]  + [2000]  + [200]
lbu = (lowerbound_u) * params.nbNoeuds_stance
ubu = (upperbound_u) * params.nbNoeuds_stance

# q et dq
lowerbound_x = [-10, -0.5, -np.pi/4, -np.pi/9, -np.pi/2, -np.pi/3, 0.5, -0.5, -1.7453, -5.2360, -5.2360, -5.2360]
upperbound_x = [10, 1.5, np.pi/4, np.pi/3, 0.0873, np.pi/9, 1.5, 0.5, 1.7453, 5.2360, 5.2360, 5.2360]
lbX   = (lowerbound_x) * (params.nbNoeuds_stance + 1)
ubX   = (upperbound_x) * (params.nbNoeuds_stance + 1)

# parameters
min_pg = 1
min_p  = 0.2
max_p  = 5
max_pg = 1
lbp = [min_pg] + [min_p] * params.nbMus
ubp = [max_pg] + [max_p] * params.nbMus


lbx = vertcat(lbu, lbX, lbp)
ubx = vertcat(ubu, ubX, ubp)

# ----------------------------- Initial guess --------------------------------------------------------------------------
# CONTROL
u0                    = np.zeros((params.nbU, params.nbNoeuds_stance))
u0[: params.nbMus, :]   = np.zeros((params.nbMus, params.nbNoeuds_stance)) + 0.1
u0[params.nbMus + 0, :] = [0] * params.nbNoeuds_stance                                  # pelvis forces
u0[params.nbMus + 1, :] = [-500] * params.nbNoeuds_stance
u0[params.nbMus + 2, :] = [0] * params.nbNoeuds_stance

# STATE
q0  = load_initialguess_q(params, 'stance')
dq0 = np.gradient(q0)
dq0 = dq0[0]
# dq0       = np.zeros((nbQ, (nbNoeuds + 1)))

X0                 = np.zeros((params.nbX, (params.nbNoeuds_stance + 1)))
X0[:params.nbQ, :] = q0
X0[params.nbQ: 2 * params.nbQ, :] = dq0

# PARAMETERS
p0 = [1] + [1] * params.nbMus

w0 = vertcat(vertcat(*u0.T), vertcat(*X0.T), p0)


# ----------------------------- Solver ---------------------------------------------------------------------------------
w = vertcat(U, X, p)
J = Ja + Je + Jm + JR

nlp    = {'x': w, 'f': J, 'g': vertcat(*G)}
opts   = {"ipopt.tol": 1e-2, "ipopt.linear_solver": "ma57", "ipopt.hessian_approximation":"limited-memory"}
solver = nlpsol("solver", "ipopt", nlp, opts)

res = solver(lbg = lbg,
             ubg = ubg,
             lbx = lbx,
             ubx = ubx,
             x0  = w0)


# RESULTS
sol_U  = res["x"][:params.nbU * params.nbNoeuds_stance]
sol_X  = res["x"][params.nbU * params.nbNoeuds_stance: -params.nP]
sol_p  = res["x"][-params.nP:]

sol_q  = [sol_X[0::params.nbX], sol_X[1::params.nbX], sol_X[2::params.nbX], sol_X[3::params.nbX], sol_X[4::params.nbX], sol_X[5::params.nbX]]
sol_dq = [sol_X[6::params.nbX], sol_X[7::params.nbX], sol_X[8::params.nbX], sol_X[9::params.nbX], sol_X[10::params.nbX], sol_X[11::params.nbX]]
sol_a  = [sol_U[0::params.nbU], sol_U[1::params.nbU], sol_U[2::params.nbU], sol_U[3::params.nbU], sol_U[4::params.nbU], sol_U[5::params.nbU], sol_U[6::params.nbU],
         sol_U[7::params.nbU], sol_U[8::params.nbU], sol_U[9::params.nbU], sol_U[10::params.nbU], sol_U[11::params.nbU], sol_U[12::params.nbU], sol_U[13::params.nbU],
         sol_U[14::params.nbU], sol_U[15::params.nbU], sol_U[16::params.nbU]]
sol_F  = [sol_U[17::params.nbU], sol_U[18::params.nbU], sol_U[19::params.nbU]]

nbNoeuds_phase = [params.nbNoeuds_stance, params.nbNoeuds_swing]
T_phase        = [params.T_stance, params.T_swing]