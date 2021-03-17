import biorbd
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


model = biorbd.Model("models/2legs_18dof_flatfootR.bioMod")

# --- Problem parameters --- #
nb_q = model.nbQ()
nb_qdot = model.nbQdot()
nb_tau = model.nbGeneralizedTorque() - model.nbRoot()
nb_mus = model.nbMuscleTotal()
nb_shooting = 30
final_time = 1.0

# --- Subject positions and initial trajectories --- #
position_high = [[0], [-0.07], [0], [0], [0], [-0.4],
                [0], [0], [0.37], [-0.13], [0], [0.11],
                [0], [0], [0.37], [-0.13], [0], [0.11]]
position_low = [-0.06, -0.36, 0, 0, 0, -0.8,
                0, 0, 1.53, -1.55, 0, 0.68,
                0, 0, 1.53, -1.55, 0, 0.68]

# # --- Subject positions and initial trajectories avec tronc --- #
# position_high = [[0], [-0.07], [0], [0], [0], [-0.4],
#                  [0], [0], [0.21],
#                 [0], [0], [0.37], [-0.13], [0], [0.11],
#                 [0], [0], [0.37], [-0.13], [0], [0.11]]
# position_low = [-0.06, -0.36, 0, 0, 0, -0.8,
#                 0, 0, 0.21,
#                 0, 0, 1.53, -1.55, 0, 0.68,
#                 0, 0, 1.53, -1.55, 0, 0.68]

# --- Compute CoM position --- #
compute_CoM = biorbd.to_casadi_func("CoM", model.CoM, MX.sym("q", nb_q, 1))
CoM_high = compute_CoM(np.array(position_high))
CoM_low = compute_CoM(np.array(position_low))

# --- Define Optimal Control Problem --- #
# Objective function
objective_functions = ObjectiveList()
objective_functions = objective.set_objectif_function(objective_functions, position_high)

# Dynamics
dynamics = DynamicsList()
dynamics.add(DynamicsFcn.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN_WITH_CONTACT)

# Constraints
constraints = ConstraintList()
constraints = constraint.set_constraints(constraints)

# Path constraints
x_bounds = BoundsList()
u_bounds = BoundsList()
x_bounds, u_bounds = bounds.set_bounds(model, x_bounds, u_bounds, mapping=False)

# Initial guess
x_init = InitialGuessList()
u_init = InitialGuessList()
x_init, u_init = initial_guess.set_initial_guess(model, x_init, u_init, position_high, position_low, nb_shooting,mapping=False)
# x_init, u_init = initial_guess.set_initial_guess_from_previous_solution(model,
#                                                                         x_init,
#                                                                         u_init,
#                                                                         save_path='./RES/muscle_driven/symetry_by_grf/',
#                                                                         nb_shooting=nb_shooting,
#                                                                         mapping=False)
# Remove pelvis torque
u_mapping = bounds.set_mapping()

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
    n_threads=6,
    # tau_mapping=u_mapping,
)

# --- Load previous solution --- #
ocp_prev, sol_prev = ocp.load('./RES/muscle_driven/CoM_obj/cycle.bo')
plot_result = Affichage(ocp_prev, sol_prev, muscles=True)
plot_result.plot_q_symetry()
plot_result.plot_tau_symetry()
plot_result.plot_qdot_symetry()
plot_result.plot_individual_forces()
plot_result.plot_muscles_symetry()
# --- Plot CoP --- #
q = sol_prev.states["q"]
cop = np.zeros((3, q.shape[1]))
for n in range(q.shape[1]):
    cop[:, n:n+1] = compute_CoM(q[:, n:n+1])
plt.figure()
plt.plot(cop[2, :])
sol_prev.animate()

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
save_path = './RES/muscle_driven/CoM_obj/'
save_results(ocp, sol, save_path)

# --- Plot CoP --- #
q = sol.states["q"]
cop = np.zeros((3, q.shape[1]))
for n in range(q.shape[1]):
    cop[:, n:n+1] = compute_CoM(q[:, n:n+1])

# --- Show results --- #
sol.animate()
sol.print()
sol.graphs()