import biorbd_casadi as biorbd
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

from ocp.load_data import data
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
# --- Problem parameters --- #
nb_shooting = 39
final_time = 1.54

# # experimental data
# name = "AmeCeg"
# model_path = "/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/squat/Data_test/" + name + "/" + name + ".bioMod"
# model = (biorbd.Model(model_path),
#          biorbd.Model(model_path),)
# # --- Subject positions and initial trajectories --- #
# q_kalman = data.get_q(name='AmeCeg', title='squat_controle')
# position_high = q_kalman[:, 0]
# position_low = q_kalman[:, 100]

# model init
model = biorbd.Model("models/2legs_18dof_flatfootR.bioMod")
# --- Subject positions and initial trajectories --- #
position_high = [0, -0.054, 0, 0, 0, -0.4,
                 0, 0, 0.37, -0.13, 0, 0.11,
                 0, 0, 0.37, -0.13, 0, 0.11]
position_low = [-0.12, -0.43, 0.0, 0.0, 0.0, -0.74,
                0.0, 0.0, 1.82, -1.48, 0.0, 0.36,
                0.0, 0.0, 1.82, -1.48, 0.0, 0.36]
q_ref = np.concatenate([np.linspace(position_high, position_low,  int(nb_shooting/2) + 1),
                        np.linspace(position_low, position_high,  int(nb_shooting/2) + 1)])

# --- Dynamics --- #
dynamics = DynamicsList()
dynamics.add(DynamicsFcn.MUSCLE_DRIVEN, with_residual_torque=True, with_contact=True)

# --- Objective function --- #
objective_functions = ObjectiveList()

# --- Constraints --- #
constraints = ConstraintList()

# --- Path constraints --- #
x_bounds = BoundsList()
u_bounds = BoundsList()


# --- Initial guess --- #
x_init = InitialGuessList()
u_init = InitialGuessList()

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
)

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
q, q_dot, tau = get_results(sol)

# --- Save results ---
save_path = './RES/torque_driven/'
save_results(ocp, sol, save_path)

# --- Plot CoM --- #
CoM = np.zeros((3, q.shape[1]))
for n in range(nb_shooting + 1):
    CoM[:, n:n+1] = compute_CoM(q[:, n])
plt.figure()
plt.plot(CoM[2, :])

# --- Show results --- #
sol.animate(show_muscles=False)
sol.graphs()


