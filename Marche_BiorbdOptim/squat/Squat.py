import biorbd_casadi as biorbd
import numpy as np
import bioviz
from casadi import MX
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

from Marche_BiorbdOptim.squat.load_data import data
from ocp.objective_functions import objective
from ocp.bounds_functions import bounds
from ocp.initial_guess_functions import initial_guess
from ocp.constraint_functions import constraint


def get_generic_model(nb_shooting):
    # model init
    model_path = "models/2legs_18dof_flatfootR.bioMod"
    position_high = [0.0, 0.9, 0.0, 0.0, 0.0, -0.4,
                     0.0, 0.0, 0.37, -0.13, 0.0, 0.0, 0.11,
                     0.0, 0.0, 0.37, -0.13, 0.0, 0.0, 0.11]
    position_low = [-0.12, 0.5, 0.0, 0.0, 0.0, -0.74,
                    0.0, 0.0, 1.82, -1.48, 0.0, 0.0, 0.36,
                    0.0, 0.0, 1.82, -1.48, 0.0, 0.0, 0.36]
    q_ref = np.concatenate([np.linspace(position_high, position_low,  int(nb_shooting/2)),
                            np.linspace(position_low, position_high,  int(nb_shooting/2) + 1)])
    return model_path, position_high, position_low, q_ref.T


# OPTIMAL CONTROL PROBLEM
nb_shooting = 40
final_time = 1.4

# generic model
# model_path, position_high, position_low, q_ref = get_generic_model(nb_shooting)
# # --- visualize model and trajectories --- #
# b = bioviz.Viz(model_path=model_path)
# b.load_movement(q_ref)
# b.exec()

# experimental subject -- check the path !!!
name = "EriHou"
higher_foot = 'R'
condition = "squat_controle"
model_path = "./Data_test/" + name + "/" + name + ".bioMod"
model = biorbd.Model(model_path)

# --- load joint trajectories, markers position and muscle activation --- #
q_ref = data.data_one_phase(name, higher_foot, condition, final_time, nb_shooting, "q")
marker_ref = data.data_one_phase(name, higher_foot, condition, final_time, nb_shooting, "marker")
activation_ref = data.data_one_phase(name, higher_foot, condition, final_time, nb_shooting, "activation")
position_high = q_ref[:, 0]
idx_low = np.where(q_ref[1, :] == np.min(q_ref[1, :]))[0][0] # pelvic vertical translation
position_low = q_ref[:, idx_low]

# --- Compute CoM position --- #
compute_CoM = biorbd.to_casadi_func("CoM", model.CoM, MX.sym("q", model.nbQ(), 1))
CoM_high = compute_CoM(np.array(position_high))
CoM_low = compute_CoM(np.array(position_low))

# --- visualize model and trajectories --- #
b = bioviz.Viz(model_path=model_path)
b.load_movement(q_ref)
b.exec()

# --- Dynamics --- #
dynamics = DynamicsList()
# dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True, expand=False)
dynamics.add(DynamicsFcn.MUSCLE_DRIVEN, with_torque=False, with_contact=True, expand=False)

# --- Objective function --- #
objective_functions = ObjectiveList()
objective.set_objectif_function(objective_functions, position_high, nb_shooting, muscles=True)

# --- Constraints --- #
constraints = ConstraintList()
constraint.set_constraints(constraints, model)

# --- Path constraints --- #
x_bounds = BoundsList()
u_bounds = BoundsList()
bounds.set_bounds(model, x_bounds, u_bounds, muscles=True)
x_bounds[0][:model.nbQ(), 0] = position_high # fixed initial position
x_bounds[0][model.nbQ():, 0] = [0] * model.nbQ() # no speed

# --- Initial guess --- #
x_init = InitialGuessList()
u_init = InitialGuessList()
initial_guess.set_initial_guess(model, x_init, u_init, q_ref, muscles=True, activation_ref=activation_ref)
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
tic = time()
solver = Solver.IPOPT()
solver.set_linear_solver("ma57")
solver.set_convergence_tolerance(1e-3)
solver.set_hessian_approximation("exact")
solver.set_maximum_iterations(3000)
solver.show_online_optim=False
sol = ocp.solve(solver=solver)
toc = time() - tic

# --- Show results --- #
sol.print()
sol.animate()
sol.graphs()


