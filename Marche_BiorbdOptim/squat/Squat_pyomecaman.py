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
    Axis,
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

def custom_CoM_high(pn: PenaltyNodes) -> MX:
    nq = pn.nlp.shape["q"]
    compute_CoM = biorbd.to_casadi_func("CoM", pn.nlp.model.CoM, pn.nlp.q)
    com = compute_CoM(pn.x[0][:nq])
    return com[2]

def custom_CoM_velocity(pn: PenaltyNodes) -> MX:
    nq = pn.nlp.shape["q"]
    compute_CoM_dot = biorbd.to_casadi_func("CoM_dot", pn.nlp.model.CoMdot, pn.nlp.q, pn.nlp.qdot)
    com_dot = compute_CoM_dot(pn.x[0][:nq], pn.x[0][nq:])
    return com_dot[2]

# OPTIMAL CONTROL PROBLEM
model = biorbd.Model("models/pyomecaman.bioMod")
c = model.contactNames()
for (i, name) in enumerate(c):
    print(f"{i} : {name.to_string()}")

# --- Problem parameters --- #
nb_q = model.nbQ()
nb_qdot = model.nbQdot()
nb_tau = model.nbGeneralizedTorque()
nb_mus = model.nbMuscleTotal()
nb_shooting = 30
final_time=1.0
min_bound, max_bound = 0, np.inf
torque_min, torque_max, torque_init = -1000, 1000, 0

# --- Subject positions and initial trajectories --- #
position_zeros = [0]*nb_q
position_high = [[0], [0.07], [-0.52], [0], [1.3], [0], [1.3], [0.37], [-0.13], [0.11], [0.37], [-0.13], [0.11]]
# position_low = [[-0.06], [-0.16], [-0.83], [0], [0.84], [0], [0.84], [1.53], [-1.55], [0.68], [1.53], [-1.55], [0.68]]
position_low = [[-0.12], [-0.23], [-1.10], [0], [1.85], [0], [1.85], [2.06], [-1.67], [0.55], [2.06], [-1.67], [0.55]]
q_init = np.zeros((nb_q, nb_shooting + 1))
for i in range(nb_q):
    q_init[i, :int(nb_shooting/2)] = np.linspace(position_high[i], position_low[i], int(nb_shooting/2)).squeeze()
    q_init[i, int(nb_shooting/2):] = np.linspace(position_low[i], position_high[i], int(nb_shooting/2) + 1).squeeze()
    # q_init[i, :] = np.linspace(position_high[i], position_low[i], nb_shooting + 1).squeeze()
qdot_init = np.gradient(q_init)[1]

# --- Compute CoM position --- #
compute_CoM = biorbd.to_casadi_func("CoM", model.CoM, MX.sym("q", nb_q, 1))
CoM_high = compute_CoM(np.array(position_high))
CoM_low = compute_CoM(np.array(position_low))

# --- Animate model --- #
# b = bioviz.Viz(loaded_model=model)
# b.set_q(np.array(position_high).squeeze())
# # b.set_q(np.array(position_high))
# b.load_movement(q_init)
# b.exec()

# --- Dynamics --- #
dynamics = DynamicsList()
dynamics.add(DynamicsFcn.TORQUE_DRIVEN_WITH_CONTACT)

# --- Objective function --- #
objective_functions = ObjectiveList()
objective_functions.add(custom_CoM_position,
                        custom_type=ObjectiveFcn.Mayer,
                        value=-0.25,
                        node=Node.MID,
                        quadratic=True,
                        weight=100)


# --- Constraints --- #
constraints = ConstraintList()
# contact forces constraint
contact_z_axes = (1, 2, 4, 5)
for c in contact_z_axes:
    constraints.add( # positive vertical forces
        ConstraintFcn.CONTACT_FORCE,
        min_bound=0,
        max_bound=np.inf,
        node=Node.ALL,
        contact_force_idx=c,
    )

# --- Path constraints --- #
x_bounds = BoundsList()
x_bounds.add(bounds=QAndQDotBounds(model))
x_bounds[0][:, 0] = np.concatenate([np.array(position_high).squeeze(), np.zeros(nb_qdot)])
x_bounds[0][:, -1] = np.concatenate([np.array(position_high).squeeze(), np.zeros(nb_qdot)])

u_bounds = BoundsList()
u_bounds.add(
            [torque_min] * nb_tau,
            [torque_max] * nb_tau,
)

# --- Initial guess --- #
x_init = InitialGuessList()
init_x = np.zeros((nb_q + nb_qdot, nb_shooting + 1))
init_x[:nb_q, :] = q_init
init_x[nb_q:, :] = qdot_init
x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)

u_init = InitialGuessList()
u_init.add([torque_init] * nb_tau)

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

plot_result = Affichage(ocp, sol, muscles=False)

compute_CoM_dot = biorbd.to_casadi_func("CoM_dot", model.CoMdot, MX.sym("q", nb_q, 1), MX.sym("qdot", nb_q, 1))
q = sol.states['q']
qdot = sol.states['q']
com_dot = np.zeros((3, q.shape[1]))
for i in range(q.shape[1]):
    com_dot[:, i:i+1] = compute_CoM_dot(q[:, i], qdot[:, i])
# --- Show results --- #
sol.animate(show_muscles=False)
sol.graphs()
sol.print()


