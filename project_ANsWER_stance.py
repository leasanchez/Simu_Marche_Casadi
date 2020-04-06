from casadi import *
import numpy as np

# add classes
from Define_parameters import Parameters
from Fcn_Objective import Fcn_Objective
from Fcn_forward_dynamic import Dynamics

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
# p = [1, 0.2,0.21, 0.524, 0.223, 0.2, 0.2, 1.68, 0.28, 0.2, 2.84, 0.2, 0.2, 0.38, 4.97, 5, 1.18, 5] # MUSCOD results
# STATE
x  = MX.sym("x", params.nbX)
q  = x[:params.nbQ]                                           # generalized coordinates
dq = x[params.nbQ: 2 * params.nbQ]                            # velocities

# ----------------------------- Define casadi function -----------------------------------------------------------------
ffcn_contact = casadi.Function("ffcn_contact",
                                [x, u, p],
                                [Dynamics.ffcn_contact(x, u, p)],
                                ["states", "controls", "parameters"],
                                ["statesdot"]).expand()

compute_GRF = casadi.Function("compute_GRF",
                              [x, u, p],
                              [Dynamics.compute_GRF(x, u, p)],
                              ["states", "controls", "parameters"],
                              ["GRF"]).expand()

# ----------------------------- Load Data from c3d file ----------------------------------------------------------------
[GRF_real, params.T, params.T_stance, params.T_swing] = LoadData.load_data_GRF(params, 'cycle')                         # GROUND REACTION FORCES & SET TIME
M_real_stance = LoadData.load_data_markers(params, 'stance')                                                            # MARKERS POSITION
U_real_stance = LoadData.load_data_emg(params, 'stance')                                                                # MUSCULAR EXCITATION


# ----------------------------- Movement -------------------------------------------------------------------------------
U = MX.sym("U", params.nbU * params.nbNoeuds_stance)          # controls
X = MX.sym("X", params.nbX * (params.nbNoeuds_stance + 1))    # states
# P = MX.sym("P", params.nP)                                    # parameters
P = p
G = []                                                 # equality constraints
Ja = 0                                                 # objective function for muscle activation
Jm = 0                                                 # objective function for markers
Je = 0                                                 # objective function for EMG
JR = 0                                                 # objective function for ground reactions
Jt = 0                                                 # objective function for residual torque

# ------------ PHASE 1 : Stance phase
for k in range(params.nbNoeuds_stance):
    # DYNAMIQUE
    Uk = U[params.nbU*k: params.nbU*(k + 1)]
    Xk = X[params.nbX*k: params.nbX*(k + 1)]
    G.append(X[params.nbX * (k + 1): params.nbX * (k + 2)] - int_RK4(ffcn_contact, params, Xk, Uk, P))

    # OBJECTIVE FUNCTION
    # [grf, Jr] = Fcn_Objective.fcn_objective_GRF(params.wR, Xk, Uk, GRF_real[:, k])
    GRF = compute_GRF(Xk, Uk, P)
    JR += Fcn_Objective.fcn_objective_GRF_casadi(params.wR, GRF, GRF_real[:, k])                                                      # tracking ground reaction --> stance
    Jm += Fcn_Objective.fcn_objective_markers(params.wMa, params.wMt, Xk[: params.nbQ], M_real_stance[:, :, k], 'stance')             # tracking marker
    Je += Fcn_Objective.fcn_objective_emg(params.wU, Uk, U_real_stance[:, k])                                                         # tracking emg
    Ja += Fcn_Objective.fcn_objective_activation(params.wL, Uk)                                                                       # min muscle activations (no EMG)
    Jt += Fcn_Objective.fcn_objective_residualtorque(params.wt, Uk[params.nbMus:])                                                    # min residual torques


# ----------------------------- Contraintes ----------------------------------------------------------------------------
# égalité
lbg = [0] * params.nbX * params.nbNoeuds_stance
ubg = [0] * params.nbX * params.nbNoeuds_stance

# ----------------------------- Bounds on w ----------------------------------------------------------------------------
# activation - excitation musculaire
min_A = 1e-3                        # 0 exclusif
max_A = 1
lowerbound_u = [min_A] * params.nbMus + [-1000]*params.nbQ
upperbound_u = [max_A] * params.nbMus + [1000]*params.nbQ
lbu = (lowerbound_u) * params.nbNoeuds_stance
ubu = (upperbound_u) * params.nbNoeuds_stance

# q et dq
lowerbound_x = [-10, -0.5, -np.pi/4, -np.pi/9, -np.pi/2, -np.pi/3, 0.5, -0.5, -1.7453, -5.2360, -5.2360, -5.2360]
upperbound_x = [10, 1.5, np.pi/4, np.pi/3, 0.0873, np.pi/9, 1.5, 0.5, 1.7453, 5.2360, 5.2360, 5.2360]
lbX   = (lowerbound_x) * (params.nbNoeuds_stance + 1)
ubX   = (upperbound_x) * (params.nbNoeuds_stance + 1)

# parameters
min_p  = 0.2
max_p  = 5
lbp = [min_p] * params.nbMus
ubp = [max_p] * params.nbMus


lbx = vertcat(lbu, lbX, lbp)
ubx = vertcat(ubu, ubX, ubp)

# ----------------------------- Initial guess --------------------------------------------------------------------------
# CONTROL
u0                      = np.zeros((params.nbU, params.nbNoeuds_stance))
u0[: params.nbMus, :]   = np.zeros((params.nbMus, params.nbNoeuds_stance)) + 0.1                                 # pelvis forces
u0[params.nbMus + 1, :] = [-500] * params.nbNoeuds_stance

# STATE
q0  = load_initialguess_q(params, 'stance')
dq0 = np.gradient(q0)
dq0 = dq0[0]

X0                 = np.zeros((params.nbX, (params.nbNoeuds_stance + 1)))
X0[:params.nbQ, :] = q0
X0[params.nbQ: 2 * params.nbQ, :] = dq0

# PARAMETERS
p0 = [1] * params.nbMus

w0 = vertcat(vertcat(*u0.T), vertcat(*X0.T), p0)


# ----------------------------- Solver ---------------------------------------------------------------------------------
w = vertcat(U, X, P)
J = Ja + Je + Jm + JR + Jt

nlp    = {'x': w, 'f': J, 'g': vertcat(*G)}
opts   = {"ipopt.tol": 1e-2, "ipopt.linear_solver": "ma57"}
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

# save txt file
file = '/home/leasanchez/programmation/Simu_Marche_Casadi/Resultats/equincocont01/RES/Stance/equincocont01_stance.txt'
f = open(file, 'a')
f.write('STATE\n\n')
np.savetxt(f, sol_X, delimiter = '\n')
f.write('\n\nCONTROL\n\n')
np.savetxt(f, sol_U, delimiter = '\n')
f.write('\n\nPARAMETER\n\n')
np.savetxt(f, sol_p, delimiter = '\n')
f.close()

sol_q  = [sol_X[0::params.nbX], sol_X[1::params.nbX], sol_X[2::params.nbX], sol_X[3::params.nbX], sol_X[4::params.nbX], sol_X[5::params.nbX]]
sol_dq = [sol_X[6::params.nbX], sol_X[7::params.nbX], sol_X[8::params.nbX], sol_X[9::params.nbX], sol_X[10::params.nbX], sol_X[11::params.nbX]]
sol_a  = [sol_U[0::params.nbU], sol_U[1::params.nbU], sol_U[2::params.nbU], sol_U[3::params.nbU], sol_U[4::params.nbU], sol_U[5::params.nbU], sol_U[6::params.nbU],
         sol_U[7::params.nbU], sol_U[8::params.nbU], sol_U[9::params.nbU], sol_U[10::params.nbU], sol_U[11::params.nbU], sol_U[12::params.nbU], sol_U[13::params.nbU],
         sol_U[14::params.nbU], sol_U[15::params.nbU], sol_U[16::params.nbU]]
sol_F  = [sol_U[17::params.nbU], sol_U[18::params.nbU], sol_U[19::params.nbU]]

nbNoeuds_phase = [params.nbNoeuds_stance, params.nbNoeuds_swing]
T_phase        = [params.T_stance, params.T_swing]