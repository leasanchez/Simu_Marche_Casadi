import biorbd
import bioviz
import numpy as np
from casadi import MX, Function
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
from Compute_Results.Plot_results import Affichage
from Compute_Results.Muscles import muscle


def get_results(sol):
    q = sol.states["q"]
    qdot = sol.states["qdot"]
    tau = sol.controls["tau"]
    muscle = sol.controls["muscles"]
    return q, qdot, tau, muscle

def save_results(ocp, sol, save_path):
    ocp.save(sol, save_path + 'cycle.bo')
    q, qdot, tau, muscle = get_results(sol.merge_phases())
    np.save(save_path + 'qdot', qdot)
    np.save(save_path + 'q', q)
    np.save(save_path + 'tau', tau)
    np.save(save_path + 'muscle', muscle)


model = (biorbd.Model("models/2legs_18dof_flatfootR.bioMod"),
         biorbd.Model("models/2legs_18dof_flatfootR.bioMod"),)

# --- Problem parameters --- #
nb_q = model[0].nbQ()
nb_qdot = model[0].nbQdot()
nb_tau = model[0].nbGeneralizedTorque()
nb_mus = model[0].nbMuscleTotal()
nb_mark = model[0].nbMarkers()
nb_shooting = (20, 20)
final_time = (0.6, 0.6)

# --- Subject positions and initial trajectories --- #
position_high = [[0], [-0.054], [0], [0], [0], [-0.4],
                [0], [0], [0.37], [-0.13], [0], [0.11],
                [0], [0], [0.37], [-0.13], [0], [0.11]]
position_high_3 = [[0], [-0.14], [0], [-0.03], [0.16], [-0.37],
                [-0.15], [-0.06], [0.35], [-0.09], [0], [0.10],
                [-0.15], [-0.06], [0.41], [-0.20], [0], [0.16]]
position_high_4 = [[0], [-0.14], [0], [-0.04], [0.21], [-0.36],
                [-0.19], [-0.07], [0.34], [-0.09], [0], [0.10],
                [-0.19], [-0.07], [0.43], [-0.26], [0], [0.19]]
position_high_5 = [[0], [-0.14], [0], [-0.05], [0.24], [-0.35],
                [-0.23], [-0.08], [0.33], [-0.08], [0], [0.10],
                [-0.22], [-0.08], [0.45], [-0.33], [0], [0.22]]
position_low = [-0.06, -0.36, 0, 0, 0, -0.8,
                0, 0, 1.53, -1.55, 0, 0.68,
                0, 0, 1.53, -1.55, 0, 0.68]

# --- Compute CoM position --- #
compute_CoM = biorbd.to_casadi_func("CoM", model[0].CoM, MX.sym("q", nb_q, 1))
CoM_high = compute_CoM(np.array(position_high))
CoM_low = compute_CoM(np.array(position_low))

# # --- Inequality position --- #
# q = np.load('./RES/muscle_driven/inequality/q.npy')
#
#
# # # --- bioviz --- #
# b = bioviz.Viz(loaded_model=model[0])
# b.load_movement(np.array(position_high))

# --- Define Optimal Control Problem --- #
# Objective function
objective_functions = ObjectiveList()
objective_functions = objective.set_objectif_function_multiphase(objective_functions, position_high_3)

# Dynamics
dynamics = DynamicsList()
dynamics.add(DynamicsFcn.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN_WITH_CONTACT, phase=0)
dynamics.add(DynamicsFcn.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN_WITH_CONTACT, phase=1)

# Constraints
constraints = ConstraintList()
constraints = constraint.set_constraints_multiphase(constraints, inequality_value=0.03)

# Path constraints
x_bounds = BoundsList()
u_bounds = BoundsList()
x_bounds, u_bounds = bounds.set_bounds(model[0], x_bounds, u_bounds, mapping=False)
x_bounds, u_bounds = bounds.set_bounds(model[0], x_bounds, u_bounds, mapping=False)
x_bounds[0][nb_q:, 0]=0

# Initial guess
x_init = InitialGuessList()
u_init = InitialGuessList()
x_init, u_init = initial_guess.set_initial_guess_multiphase(model, x_init, u_init, position_high_3, position_low, nb_shooting,mapping=False)

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

# --- Load previous solution --- #
# ocp_prev, sol = ocp.load('./RES/muscle_driven/CoM_obj/cycle.bo')
# plot_result = Affichage(ocp_prev, sol_prev, muscles=True)
# muscle_prev = muscle(ocp_prev, sol_prev)
# plot_result.plot_momentarm(idx_muscle=11)
# plot_result.plot_muscles_activation_symetry()
# plot_result.plot_muscles_force_symetry()
# plot_result.plot_tau_symetry()

# plot_result.plot_q_symetry()
# plot_result.plot_tau_symetry()
# plot_result.plot_qdot_symetry()
# plot_result.plot_individual_forces()

# # --- Plot CoP --- #
# q = sol_prev.states["q"]
# cop = np.zeros((3, q.shape[1]))
# for n in range(q.shape[1]):
#     cop[:, n:n+1] = compute_CoM(q[:, n:n+1])
# plt.figure()
# plt.plot(cop[2, :])
# sol_prev.animate()

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
save_path = './RES/muscle_driven/multiphase/3cm/'
save_results(ocp, sol, save_path)

# # --- Plot markers --- #
# markers = biorbd.to_casadi_func("markers", model[0].markers, MX.sym("q", nb_q, 1))
# contact_idx_right = (31, 32, 33)
# contact_idx_left = (55, 56, 57)
# markers_pos = np.zeros((3, nb_mark, q.shape[1]))
# for n in range(q.shape[1]):
#     markers_pos[:, :, n] = markers(q[:, n:n+1])

# --- Plot Symetry --- #
sol_merged=sol.merge_phases()
plot_result = Affichage(ocp, sol_merged, muscles=True)
plot_result.plot_q_symetry()
plot_result.plot_tau_symetry()
plot_result.plot_qdot_symetry()
plot_result.plot_individual_forces()
plot_result.plot_sum_forces()
plot_result.plot_muscles_activation_symetry()
plot_result.plot_CoM_displacement(target=0.3)

# --- Show results --- #
sol.animate()
sol.print()
sol.graphs()