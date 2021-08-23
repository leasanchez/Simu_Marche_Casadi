import biorbd_casadi as biorbd
# import bioviz
import numpy as np
from casadi import MX, Function
from matplotlib import pyplot as plt
from time import time
from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    BoundsList,
    ObjectiveList,
    ConstraintList,
    InitialGuessList,
    Solver,
    OdeSolver,
)

from ocp.load_data import data
from ocp.objective_functions import objective
from ocp.constraint_functions import constraint
from ocp.bounds_functions import bounds
from ocp.initial_guess_functions import initial_guess
# from Compute_Results.Plot_results import Affichage
# from Compute_Results.Muscles import muscle


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


# --- Problem parameters --- #
nb_shooting = (19, 19)
final_time = (0.75, 0.75)

# experimental data
# name = "AmeCeg"
# model_path = "/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/squat/Data_test/" + name + "/" + name + ".bioMod"
# model = (biorbd.Model(model_path),
#          biorbd.Model(model_path),)
# # --- Subject positions and initial trajectories --- #
# q_ref = []
# q_kalman = data.get_q(name='AmeCeg', title='squat_controle')
# position_high = q_kalman[:, 0]
# position_low = q_kalman[:, 100]
# a = q_kalman[:, 0::5]
# q_ref.append(a[:, :nb_shooting[0]+1])
# q_ref.append(a[:, nb_shooting[0]+1:])

# model init
model = (biorbd.Model("models/2legs_18dof_flatfootR.bioMod"),
         biorbd.Model("models/2legs_18dof_flatfootR.bioMod"))
# --- Subject positions and initial trajectories --- #
position_high = [0.0, 0.246, 0.0, 0.0, 0.0, -0.4,
                 0.0, 0.0, 0.37, -0.13, 0.0, 0.0, 0.11,
                 0.0, 0.0, 0.37, -0.13, 0.0, 0.0, 0.11]
position_low = [-0.12, -0.13, 0.0, 0.0, 0.0, -0.74,
                0.0, 0.0, 1.82, -1.48, 0.0, 0.0, 0.36,
                0.0, 0.0, 1.82, -1.48, 0.0, 0.0, 0.36]
q_ref = []
q_ref.append(np.linspace(position_high, position_low, nb_shooting[0] + 1).T)
q_ref.append(np.linspace(position_low, position_high, nb_shooting[0] + 1).T)

# --- Compute CoM position --- #
compute_CoM = biorbd.to_casadi_func("CoM", model[0].CoM, MX.sym("q", model[0].nbQ(), 1))
CoM_high = compute_CoM(np.array(position_high))
CoM_low = compute_CoM(np.array(position_low))

# --- Define Optimal Control Problem --- #
# Dynamics
dynamics = DynamicsList()
dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True, phase=0, expand=False)
dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True, phase=1, expand=False)
# dynamics.add(DynamicsFcn.MUSCLE_DRIVEN, with_residual_torque=True, with_contact=True, phase=0, expand=False)
# dynamics.add(DynamicsFcn.MUSCLE_DRIVEN, with_residual_torque=True, with_contact=True, phase=1, expand=False)

# Objective function
objective_functions = ObjectiveList()
objective.set_objectif_function_multiphase(objective_functions, position_high, muscles=False)

# Constraints
constraints = ConstraintList()
constraint.set_constraints_multiphase(constraints, inequality_value=0.0)

# Path constraints
x_bounds = BoundsList()
u_bounds = BoundsList()
bounds.set_bounds(model[0], x_bounds, u_bounds, muscles=False, mapping=False)
bounds.set_bounds(model[1], x_bounds, u_bounds, muscles=False, mapping=False)
x_bounds[0][:model[0].nbQ(), 0] = position_high

# Initial guess
x_init = InitialGuessList()
u_init = InitialGuessList()
initial_guess.set_initial_guess_multiphase(model, x_init, u_init, q_ref, muscles=False, mapping=False)

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
    # ode_solver=OdeSolver.COLLOCATION(method='radau', polynomial_degree=4),
)

# --- Solve the program --- #
tic = time()
sol = ocp.solve(
    solver=Solver.IPOPT,
    solver_options={
        "ipopt.tol": 1e-6,
        "ipopt.max_iter": 2000,
        "ipopt.hessian_approximation": "limited-memory",
        "ipopt.limited_memory_max_history": 50,
        "ipopt.linear_solver": "ma57",
    },
    show_online_optim=False,
)
toc = time() - tic

# --- Save results --- #
save_path = './RES/muscle_driven/multiphase/ankle/'
save_results(ocp, sol, save_path)

# --- Show results --- #
sol.animate()
sol.print()
sol.graphs()