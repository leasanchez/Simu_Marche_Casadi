import biorbd_casadi as biorbd
import numpy as np
import bioviz
import xarray
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
    OdeSolver,
)

from ocp.load_data import data
from ocp.objective_functions import objective
from ocp.bounds_functions import bounds
from ocp.initial_guess_functions import initial_guess
from ocp.constraint_functions import constraint

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


# OPTIMAL CONTROL PROBLEM
# experimental data
name = "EriHou"
model_path = "/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/squat/Data_test/" + name + "/" + name + "_3D.bioMod"
model = biorbd.Model(model_path)

# --- Subject positions and initial trajectories --- #
# idx = [0, 1, 5, # pelvis
#        8, # hip R
#        11, # knee R
#        14, # ankle R
#        17, # hip L
#        20, # knee L
#        23] # ankle L
idx = range(model.nbQ())
q_kalman = data.get_q(name=name, title='squat_controle')
position_high = q_kalman[idx, 0]
position_low = q_kalman[idx, 100]

# --- compute markers position --- #
compute_markers_position = biorbd.to_casadi_func("Markers", model.markers, MX.sym("q", model.nbQ(), 1))
mark = np.zeros((3, model.nbMarkers(), q_kalman.shape[1]))
for i in range(q_kalman.shape[1]):
    mark[:, :, i] = compute_markers_position(q_kalman[idx, i])
data_marker = xarray.DataArray(data=mark)

# --- Compute CoM position --- #
compute_CoM = biorbd.to_casadi_func("CoM", model.CoM, MX.sym("q", model.nbQ(), 1))
CoM_trajectory = np.zeros((3, q_kalman.shape[1]))
CoM_high = compute_CoM(np.array(position_high))
CoM_low = compute_CoM(np.array(position_low))
for i in range(q_kalman.shape[1]):
    CoM_trajectory[:, i:i+1] = compute_CoM(np.array(q_kalman[idx, i]))

# --- affichage biomod --- #
# b = bioviz.Viz(model_path=model_path)
# b.load_movement(q_kalman[idx, :])
# # b.load_experimental_markers(mark)
# b.exec()

# --- Problem parameters --- #
nb_shooting = 38
final_time = 1.8

# # load previous solution
# save_path = "/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/squat/Data_test/" + name + "/Simulation/torque_driven/" + name + "_coloc_2D.bo"
# save_path2 = "/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/squat/Data_test/" + name + "/Simulation/torque_driven/" + name + "_5cm_coloc_2D.bo"
# ocp_load, sol_load = OptimalControlProgram.load(save_path2)
# sol_load.graphs()
# q, q_dot, tau = get_results(sol_load)
# b = bioviz.Viz(model_path=model_path)
# b.load_movement(q)
# b.exec()


# # model init
# model = biorbd.Model("models/2legs_18dof_flatfootR.bioMod")
# # --- Subject positions and initial trajectories --- #
# # position_high = [0, -0.054, 0, 0, 0, -0.4,
# #                  0, 0, 0.37, -0.13, 0, 0.11,
# #                  0, 0, 0.37, -0.13, 0, 0.11]
# # position_low = [-0.12, -0.43, 0.0, 0.0, 0.0, -0.74,
# #                 0.0, 0.0, 1.82, -1.48, 0.0, 0.36,
# #                 0.0, 0.0, 1.82, -1.48, 0.0, 0.36]
# position_high = [0.0, 0.246, 0.0, 0.0, 0.0, -0.4,
#                  0.0, 0.0, 0.37, -0.13, 0.0, 0.0, 0.11,
#                  0.0, 0.0, 0.37, -0.13, 0.0, 0.0, 0.11]
# position_low = [-0.12, -0.13, 0.0, 0.0, 0.0, -0.74,
#                 0.0, 0.0, 1.82, -1.48, 0.0, 0.0, 0.36,
#                 0.0, 0.0, 1.82, -1.48, 0.0, 0.0, 0.36]
# q_ref = np.concatenate([np.linspace(position_high, position_low,  int(nb_shooting/2)),
#                         np.linspace(position_low, position_high,  int(nb_shooting/2) + 1)])

# --- Dynamics --- #
dynamics = DynamicsList()
dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True, expand=False)
# dynamics.add(DynamicsFcn.MUSCLE_DRIVEN, with_residual_torque=True, with_contact=True, expand=False)

# --- Objective function --- #
com_ref = CoM_trajectory[:, 0::5]
q_ref = q_kalman[idx, 0::5]
objective_functions = ObjectiveList()
objective.set_objectif_function_exp(objective_functions, q_ref, mark[:, :, 0::5])

# --- Constraints --- #
constraints = ConstraintList()
constraint.set_constraints_exp(constraints)

# --- Path constraints --- #
x_bounds = BoundsList()
u_bounds = BoundsList()
bounds.set_bounds(model, x_bounds, u_bounds, muscles=False)
x_bounds[0][:model.nbQ(), 0] = position_high

# --- Initial guess --- #
x_init = InitialGuessList()
u_init = InitialGuessList()
initial_guess.set_initial_guess(model, x_init, u_init, q_ref[:, :-1], muscles=False, mapping=False)
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
    n_threads=8,
    # ode_solver=OdeSolver.COLLOCATION(method="radau", polynomial_degree=4)
)

# --- Solve the program --- #
sol = ocp.solve(
    solver=Solver.IPOPT,
    solver_options={
        "ipopt.tol": 1e-6,
        "ipopt.max_iter": 2000,
        "ipopt.hessian_approximation": "exact",
        "ipopt.limited_memory_max_history": 50,
        "ipopt.linear_solver": "ma57",
    },
    show_online_optim=False,
)

# # --- Get and Animate Results --- #
# q, q_dot, tau = get_results(sol)
# b = bioviz.Viz(model_path=model_path)
# b.load_movement(q)
# b.exec()

# # --- Save results --- #
# save_path = "/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/squat/Data_test/" + name + "/Simulation/torque_driven/" + name + "_controle_3D.bo"
# ocp.save(sol, save_path)

# --- Show results --- #
sol.print()
sol.animate()
sol.graphs()


