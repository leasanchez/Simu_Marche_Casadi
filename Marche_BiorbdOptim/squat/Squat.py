import numpy as np
from casadi import dot, Function, vertcat, MX, mtimes, nlpsol, mmax
import biorbd
import bioviz
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    BoundsList,
    Bounds,
    QAndQDotBounds,
    InitialGuessList,
    InitialGuess,
    ObjectiveList,
    ObjectiveFcn,
    InterpolationType,
    Node,
    ConstraintList,
    ConstraintFcn,
    Solver,
    PenaltyNodes,
)

from ocp.objective_functions import objective
from ocp.constraint_functions import constraint
from ocp.bounds_functions import bounds
from ocp.initial_guess_functions import initial_guess


def custom_CoM_low(pn: PenaltyNodes) -> MX:
    nq = pn.nlp.shape["q"]
    compute_CoM = biorbd.to_casadi_func("CoM", pn.nlp.model.CoM, pn.nlp.q)
    com = compute_CoM(pn.x[15][:nq])
    return com[2] + 0.30

def custom_CoM_variation(pn: PenaltyNodes) -> MX:
    nq = pn.nlp.shape["q"]
    compute_CoM = biorbd.to_casadi_func("CoM", pn.nlp.model.CoM, pn.nlp.q)
    com_init = compute_CoM(pn.x[0][:nq])
    val = []
    for n in range(1, pn.nlp.ns):
        val = vertcat(val, com_init[0] - compute_CoM(pn.x[pn.t[n]][:nq])[0])
        val = vertcat(val, com_init[1] - compute_CoM(pn.x[pn.t[n]][:nq])[1])
    return val

def custom_CoM_high(pn: PenaltyNodes) -> MX:
    nq = pn.nlp.shape["q"]
    compute_CoM = biorbd.to_casadi_func("CoM", pn.nlp.model.CoM, pn.nlp.q)
    com = compute_CoM(pn.x[0][:nq])
    return com[2]

# OPTIMAL CONTROL PROBLEM
model = biorbd.Model("Modeles_S2M/2legs_18dof_flatfootR.bioMod")
# c = model.contactNames()
# for (i, name) in enumerate(c):
#     print(f"{i} : {name.to_string()}")

# q_name = []
# for s in range(model.nbSegment()):
#     seg_name = model.segment(s).name().to_string()
#     for d in range(model.segment(s).nbDof()):
#         dof_name = model.segment(s).nameDof(d).to_string()
#         q_name.append(seg_name + "_" + dof_name)
# for (i, q) in enumerate(q_name):
#     print(f"{i} : {q}")

# --- Problem parameters --- #
nb_q = model.nbQ()
nb_qdot = model.nbQdot()
nb_tau = model.nbGeneralizedTorque()
nb_mus = model.nbMuscleTotal()
nb_shooting = 30
final_time=1.0
min_bound, max_bound = 0, np.inf
torque_min, torque_max, torque_init = -1000, 1000, 0
activation_min, activation_max, activation_init = 1e-3, 1.0, 0.1

# --- Subject positions and initial trajectories --- #
position_zeros = [0]*nb_q
position_high = [[0], [-0.07], [0], [0], [0], [-0.4],
                [0], [0], [0.37], [-0.13], [0], [0.11],
                [0], [0], [0.37], [-0.13], [0], [0.11]]
position_low = [-0.06, -0.36, 0, 0, 0, -0.8,
                0, 0, 1.53, -1.55, 0, 0.68,
                0, 0, 1.53, -1.55, 0, 0.68]

# position_high = [[0], [-0.07], [0], [0], [0], [-0.4],
#                 [0], [0], [0],
#                 [0], [0], [0.37], [-0.13], [0], [0.11],
#                 [0], [0], [0.37], [-0.13], [0], [0.11]]
# position_low = [-0.06, -0.36, 0, 0, 0, -0.8,
#                 0, 0, 0,
#                 0, 0, 1.53, -1.55, 0, 0.68,
#                 0, 0, 1.53, -1.55, 0, 0.68]
q_init = np.zeros((nb_q, nb_shooting + 1))
for i in range(nb_q):
    q_init[i, :int(nb_shooting/2)] = np.linspace(position_high[i], position_low[i], int(nb_shooting/2)).squeeze()
    q_init[i, int(nb_shooting/2):] = np.linspace(position_low[i], position_high[i], int(nb_shooting/2) + 1).squeeze()
    # q_init[i, :] = np.linspace(position_high[i], position_low[i], nb_shooting + 1).squeeze()
qdot_init = np.gradient(q_init)[1]
qddot_init = np.gradient(qdot_init)[1]

# --- Compute CoM position --- #
symbolic_q = MX.sym("q", nb_q, 1)
compute_CoM = Function(
    "ComputeCoM",
    [symbolic_q],
    [model.CoM(symbolic_q).to_mx()],
    ["q"],
    ["CoM"],
).expand()

CoM_high = compute_CoM(np.array(position_high))
CoM_low = compute_CoM(np.array(position_low))

# # --- Animate model --- #
# b = bioviz.Viz(loaded_model=model, show_muscles=False, show_segments_center_of_mass=False, show_local_ref_frame=False)
# b.set_q(np.array(position_high).squeeze())
# # b.set_q(np.array(position_high))
# b.load_movement(q_init)
# b.exec()


# --- Objective function --- #
objective_functions = ObjectiveList()
objective_functions = objective.set_objectif_function_torque_driven(objective_functions)

# --- Dynamics --- #
dynamics = DynamicsList()
dynamics.add(DynamicsFcn.TORQUE_DRIVEN_WITH_CONTACT)

# --- Constraints --- #
constraints = ConstraintList()
constraints = constraint.set_constraints(constraints)

# --- Path constraints --- #
x_bounds = BoundsList()
u_bounds = BoundsList()
x_bounds, u_bounds = bounds.set_bounds_torque_driven(model, x_bounds, u_bounds, position_high)

# --- Initial guess --- #
x_init = InitialGuessList()
init_x = np.zeros((nb_q + nb_qdot, nb_shooting + 1))
init_x[:nb_q, :] = q_init
init_x[nb_q:, :] = qdot_init
x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)

u_init = InitialGuessList()
# u_init.add([torque_init]*nb_tau + [activation_init]*nb_mus)
u_init.add([torque_init] * nb_tau)

# # Load previous solution
# save_path = './RES/torque_driven/'
# x_init = InitialGuessList()
# init_x = np.zeros((nb_q + nb_qdot, nb_shooting + 1))
# init_x[:nb_q, :] = np.load(save_path + "q.npy")
# init_x[nb_q:, :] = np.load(save_path + "qdot.npy")
# x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)
#
# u_init = InitialGuessList()
# init_u = np.load(save_path + "tau.npy")[:, :-1]
# u_init.add(init_u, interpolation=InterpolationType.EACH_FRAME)

# ------------- #

ocp = OptimalControlProgram(
    model,
    dynamics,
    nb_shooting,
    final_time,
    x_init,
    u_init,
    x_bounds,
    u_bounds,
    objective_functions,
    constraints,
    n_threads=4,
)

# # --- Get Previous Results --- #
# path_previous = './RES/torque_driven/cycle.bo'
# ocp_previous, sol_previous = ocp.load(path_previous)
# states_previous, controls_previous = Data.get_data(ocp_previous, sol_previous["x"])
# q_previous = states_previous["q"]
#
# # --- Show results --- #
# ShowResult(ocp_previous, sol_previous).animate(show_muscles=False)
# ShowResult(ocp_previous, sol_previous).animate(show_muscles=False, show_segments_center_of_mass=False, show_local_ref_frame=False)
# ShowResult(ocp_previous, sol_previous).graphs()
#
# # --- Plot CoM --- #
# CoM = np.zeros((3, q_previous.shape[1]))
# for n in range(nb_shooting + 1):
#     CoM[:, n:n+1] = compute_CoM(q_previous[:, n])
# plt.figure()
# plt.plot(CoM[2, :])

# --- Solve the program --- #
sol = ocp.solve(
    solver=Solver.IPOPT,
    solver_options={
        "ipopt.tol": 1e-6,
        "ipopt.max_iter": 5000,
        "ipopt.hessian_approximation": "exact",
        "ipopt.limited_memory_max_history": 50,
        "ipopt.linear_solver": "ma57",
    },
    show_online_optim=False,
)

# --- Get Results --- #
states_sol, controls_sol = Data.get_data(ocp, sol["x"])
q = states_sol["q"]
q_dot = states_sol["qdot"]
tau = controls_sol["tau"]

# --- Save results ---
save_path = './RES/torque_driven/'
ocp.save(sol, save_path + 'cycle.bo')
np.save(save_path + 'qdot', q_dot)
np.save(save_path + 'q', q)
np.save(save_path + 'tau', tau)

# --- Plot CoM --- #
CoM = np.zeros((3, q.shape[1]))
for n in range(nb_shooting + 1):
    CoM[:, n:n+1] = compute_CoM(q[:, n])
plt.figure()
plt.plot(CoM[2, :])

# --- Show results --- #
ShowResult(ocp, sol).animate(show_muscles=False)
ShowResult(ocp, sol).graphs()


