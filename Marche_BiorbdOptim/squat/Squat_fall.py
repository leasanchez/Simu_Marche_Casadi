import biorbd_casadi as biorbd
# import bioviz
import numpy as np
from casadi import MX
from matplotlib import pyplot as plt
from time import time
from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    BoundsList,
    InitialGuessList,
    ObjectiveList,
    ConstraintList,
    Solver,
)

from ocp.objective_functions import objective
from ocp.constraint_functions import constraint
from ocp.bounds_functions import bounds
from ocp.initial_guess_functions import initial_guess
# from Compute_Results.Plot_results import Affichage
# from Compute_Results.Contact_Forces import contact

def get_results(sol):
    q = sol.states["q"]
    qdot = sol.states["qdot"]
    tau = sol.controls["tau"]
    return q, qdot, tau

def save_results(ocp, sol, save_path):
    ocp.save(sol, save_path + 'cycle.bo')
    q, qdot, tau = get_results(sol.merge_phases())
    np.save(save_path + 'qdot', qdot)
    np.save(save_path + 'q', q)
    np.save(save_path + 'tau', tau)


model = biorbd.Model("models/2legs_18dof_flatfootR.bioMod")

# --- Problem parameters --- #
nb_q = model.nbQ()
nb_qdot = model.nbQdot()
nb_tau = model.nbGeneralizedTorque()
nb_mus = model.nbMuscleTotal()
nb_mark = model.nbMarkers()
nb_shooting = 20
time_min, time_max, final_time = 0.2, 1.0, 0.77
tau_min, tau_max, tau_init = -500, 500, 0
activation_min, activation_max, activation_init = 0, 1, 0.5

# --- Subject positions and initial trajectories --- #
# position_high = [[0], [-0.054], [0], [0], [0], [-0.4],
#                 [0], [0], [0.37], [-0.13], [0], [0.11],
#                 [0], [0], [0.37], [-0.13], [0], [0.11]]
position_high = [0, -0.054, 0, 0, 0, -0.4,
                 0, 0, 0.37, -0.13, 0, 0.11,
                 0, 0, 0.37, -0.13, 0, 0.11]
position_low = [-0.12, -0.43, 0.0, 0.0, 0.0, -0.74,
                0.0, 0.0, 1.82, -1.48, 0.0, 0.36,
                0.0, 0.0, 1.82, -1.48, 0.0, 0.36]
position_low_2 = [-0.06, -0.36, 0, 0, 0, -0.8,
                   0, 0, 1.53, -1.55, 0, 0.68,
                   0, 0, 1.53, -1.55, 0, 0.68]

# --- Compute CoM position --- #
compute_CoM = biorbd.to_casadi_func("CoM", model.CoM, MX.sym("q", nb_q, 1))
CoM_high = compute_CoM(np.array(position_high))
CoM_low = compute_CoM(np.array(position_low))

# --- Define Optimal Control Problem --- #
# Objective function
objective_functions = ObjectiveList()
objective.set_objectif_function_fall(objective_functions, muscles=True)

# Dynamics
dynamics = DynamicsList()
dynamics.add(DynamicsFcn.MUSCLE_DRIVEN, with_residual_torque=True, with_contact=True, expand=False)

# Constraints
constraints = ConstraintList()
# constraint.set_constraints_fall(constraints, inequality_value=0.0)

# Path constraints
x_bounds = BoundsList()
u_bounds = BoundsList()
bounds.set_bounds(model, x_bounds, u_bounds, muscles=True)

# Initial guess
q_ref = np.linspace(position_high, position_low, nb_shooting + 1)
x_init = InitialGuessList()
u_init = InitialGuessList()
initial_guess.set_initial_guess(model, x_init, u_init, q_ref.T, muscles=True, mapping=False)

# ------------- #
ocp = OptimalControlProgram(
    biorbd_model=model,
    dynamics=dynamics,
    n_shooting=nb_shooting,
    phase_time=final_time,
    x_init=x_init,
    u_init=u_init,
    x_bounds=x_bounds,
    u_bounds=u_bounds,
    objective_functions=objective_functions,
    constraints=constraints,
    n_threads=8,
)

# --- Solve the program --- #
tic = time()
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
toc = time() - tic

# --- Save results --- #
save_path = './RES/torque_driven/position_basse/'
save_results(ocp, sol, save_path)

# --- Plot CoP --- #
q = sol.states["q"]
com = np.zeros((3, q.shape[1]))
for n in range(q.shape[1]):
    com[:, n:n+1] = compute_CoM(q[:, n:n+1])
plt.plot(com[2, :])

# --- Plot markers --- #
markers = biorbd.to_casadi_func("markers", model.markers, MX.sym("q", nb_q, 1))
markers_velocity = biorbd.to_casadi_func("markers_velocity", model.markersVelocity, MX.sym("q", nb_q, 1), MX.sym("qdot", nb_q, 1))
contact_idx_right = (31, 32, 33)
contact_idx_left = (55, 56, 57)
q = sol.states["q"]
qdot = sol.states["qdot"]
markers_pos = np.zeros((3, nb_mark, q.shape[1]))
markers_vel = np.zeros((3, nb_mark, q.shape[1]))
for n in range(q.shape[1]):
    markers_pos[:, :, n] = markers(q[:, n:n+1])
    markers_vel[:, :, n] = markers_velocity(q[:, n:n + 1], qdot[:, n:n+1])

# # --- Plot Symetry --- #
# plot_result = Affichage(ocp, sol, muscles=False)
# plot_result.plot_q_symetry()
# plot_result.plot_tau_symetry()
# plot_result.plot_qdot_symetry()
# plot_result.plot_individual_forces()
#
# contact_results = contact(ocp, sol, muscles=False)

# --- Show results --- #
sol.animate()
sol.print()
sol.graphs()