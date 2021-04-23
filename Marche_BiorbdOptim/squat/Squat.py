import biorbd
import bioviz
import numpy as np
from casadi import MX, Function
import biorbd
import bioviz
from matplotlib import pyplot as plt
from time import time
from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    BoundsList,
    InitialGuessList,
    ObjectiveList,
    ObjectiveFcn,
    ConstraintList,
    ConstraintFcn,
    Solver,
    Node,
    PenaltyNodes,
    InterpolationType,
    QAndQDotBounds,
)

from ocp.objective_functions import objective
from ocp.constraint_functions import constraint
from ocp.bounds_functions import bounds
from ocp.initial_guess_functions import initial_guess
from Compute_Results.Plot_results import Affichage
from Compute_Results.Contact_Forces import contact

def custom_CoM_position(pn: PenaltyNodes, value: float) -> MX:
    nq = pn.nlp.shape["q"]
    compute_CoM = biorbd.to_casadi_func("CoM", pn.nlp.model.CoM, pn.nlp.q)
    com = compute_CoM(pn.x[0][:nq])
    return com[2] - value

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
model = (biorbd.Model("models/2legs_18dof_flatfootR_2D.bioMod"),
         biorbd.Model("models/2legs_18dof_flatfootR_2D.bioMod"), )

# --- Problem parameters --- #
nb_q = model[0].nbQ()
nb_qdot = model[0].nbQdot()
nb_tau = model[0].nbGeneralizedTorque()
nb_mus = model[0].nbMuscleTotal()
nb_shooting = (20, 20)
final_time = (0.5, 0.5)
time_min = (0.2, 0.2)
time_max = (1.0, 1.0)

c = model[0].contactNames()
for (i, name) in enumerate(c):
    print(f"{i} : {name.to_string()}")


# --- Subject positions and initial trajectories --- #
position_zeros = [0]*nb_q
position_high = [[0], [-0.07], [-0.4],
                [0.37], [-0.13], [0.11],
                [0.37], [-0.13], [0.11]]
position_low = [-0.06, -0.36, -0.8,
                1.53, -1.55, 0.68,
                1.53, -1.55, 0.68]

q_init_fall = np.zeros((nb_q, nb_shooting[0] + 1))
q_init_climb = np.zeros((nb_q, nb_shooting[1] + 1))
for i in range(nb_q):
    q_init_fall[i, :] = np.linspace(position_high[i], position_low[i], nb_shooting[0]+1).squeeze()
    q_init_climb[i, :] = np.linspace(position_low[i], position_high[i], nb_shooting[1]+1).squeeze()

# --- Compute CoM position --- #
compute_CoM = biorbd.to_casadi_func("CoM", model[0].CoM, MX.sym("q", nb_q, 1))
CoM_high = compute_CoM(np.array(position_high))
CoM_low = compute_CoM(np.array(position_low))

# --- Dynamics --- #
dynamics = DynamicsList()
dynamics.add(DynamicsFcn.TORQUE_DRIVEN_WITH_CONTACT)
dynamics.add(DynamicsFcn.TORQUE_DRIVEN_WITH_CONTACT)

# --- Objective function --- #
objective_functions = ObjectiveList()
objective_functions.add(custom_CoM_position,
                        custom_type=ObjectiveFcn.Mayer,
                        value=-0.25,
                        node=Node.END,
                        quadratic=True,
                        weight=100,
                        phase=0)

objective_functions.add(ObjectiveFcn.Mayer.TRACK_STATE,
                        node=Node.END,
                        quadratic=True,
                        target=np.concatenate([np.array(position_high).squeeze(), np.zeros(nb_qdot)]).reshape(nb_q + nb_qdot, 1),
                        weight=10,
                        phase=1)

# --- Constraints --- #
constraints = ConstraintList()
n_phases = 2
for i in range(n_phases):
    # Contact forces
    contact_z_axes = (1, 2, 4, 5)
    for c in contact_z_axes:
        constraints.add(  # positive vertical forces
            ConstraintFcn.CONTACT_FORCE,
            min_bound=0,
            max_bound=np.inf,
            node=Node.ALL,
            contact_force_idx=c,
            phase=i,
        )

# --- Path constraints --- #
x_bounds = BoundsList()
u_bounds = BoundsList()
x_bounds.add(bounds=QAndQDotBounds(model[0]))
x_bounds.add(bounds=QAndQDotBounds(model[0]))
x_bounds[0][:, 0] = np.concatenate([np.array(position_high).squeeze(), np.zeros(nb_qdot)])
# x_bounds[1][:, -1] = np.concatenate([np.array(position_high).squeeze(), np.zeros(nb_qdot)])

u_bounds = BoundsList()
u_bounds.add([0] * nb_tau,[1000] * nb_tau)
u_bounds.add([0] * nb_tau,[1000] * nb_tau)

# --- Initial guess --- #
x_init = InitialGuessList()
x = np.zeros((model[0].nbQ() + model[0].nbQ(), nb_shooting[0]+1))
x[:nb_q, :] =q_init_fall
x_init.add(x, interpolation=InterpolationType.EACH_FRAME)
x[:nb_q, :] =q_init_climb
x_init.add(x, interpolation=InterpolationType.EACH_FRAME)

u_init = InitialGuessList()
u_init.add([0] * model[0].nbGeneralizedTorque())
u_init.add([0] * model[0].nbGeneralizedTorque())

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


