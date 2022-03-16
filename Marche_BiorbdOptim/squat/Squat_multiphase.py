import biorbd_casadi as biorbd
import bioviz
import numpy as np
from casadi import MX
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


def get_results(sol, muscles=False):
    q = sol.states["q"]
    qdot = sol.states["qdot"]
    tau = sol.controls["tau"]
    if muscles:
        muscles = sol.controls["muscles"]
        return q, qdot, tau, muscles
    else:
        return q, qdot, tau

def save_results(ocp, sol, save_path, muscles=False):
    ocp.save(sol, save_path + 'cycle.bo')
    if muscles:
        q, qdot, tau, muscles = get_results(sol.merge_phases(), muscles)
        np.save(save_path + 'muscles', muscles)
    else:
        q, qdot, tau = get_results(sol.merge_phases())
    np.save(save_path + 'qdot', qdot)
    np.save(save_path + 'q', q)
    np.save(save_path + 'tau', tau)

# --- Problem parameters --- #
nb_shooting = (19, 19)
final_time = (0.75, 0.75)

# experimental data
name = "EriHou"
title = "squat_controle"
model_path = "./Data_test/" + name + "/" + name + ".bioMod"
model = (biorbd.Model(model_path),
         biorbd.Model(model_path),)

# --- Subject positions and initial trajectories --- #
q_ref = data.data_per_phase(name, title, final_time, nb_shooting, "q")
marker_ref = data.data_per_phase(name, title, final_time, nb_shooting, "marker")
activation_ref = data.data_per_phase(name, title, final_time, nb_shooting, "activation")
position_high = q_ref[0][:, 0]
position_low = q_ref[0][:, -1]

b=bioviz.Viz(model_path=model_path)
b.load_movement(q_ref[0])
b.exec()

# --- Compute CoM position --- #
compute_CoM = biorbd.to_casadi_func("CoM", model[0].CoM, MX.sym("q", model[0].nbQ(), 1))
CoM_high = compute_CoM(np.array(position_high))
CoM_low = compute_CoM(np.array(position_low))

# --- Define Optimal Control Problem --- #
# Dynamics
dynamics = DynamicsList()
dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True, phase=0, expand=False)
dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True, phase=1, expand=False)

# Objective function
objective_functions = ObjectiveList()
objective.set_objectif_function_multiphase(objective_functions, q_ref, muscles=False)

# Constraints
constraints = ConstraintList()
constraint.set_constraints_multiphase(constraints, model[0])

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
solver = Solver.IPOPT()
solver.set_linear_solver("ma57")
solver.set_convergence_tolerance(1e-3)
solver.set_hessian_approximation("exact")
solver.set_maximum_iterations(3000)
solver.show_online_optim=False
sol = ocp.solve(solver=solver)
toc = time() - tic

# --- Save results --- #
save_path = './RES/torque_driven/multiphase/with_contact/'
save_results(ocp, sol, save_path)

# --- Show results --- #
sol.animate()
sol.print()
