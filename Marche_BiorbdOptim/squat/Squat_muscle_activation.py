import numpy as np
from casadi import dot, Function, vertcat, MX, mtimes, nlpsol, mmax
import biorbd
import bioviz
from time import time
from matplotlib import pyplot as plt
from Plot_results import Affichage

from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    BoundsList,
    QAndQDotBounds,
    InitialGuessList,
    ObjectiveList,
    ObjectiveFcn,
    InterpolationType,
    Node,
    ConstraintList,
    ConstraintFcn,
    Solver,
    PenaltyNodes,
)


def custom_CoM_low(pn: PenaltyNodes) -> MX:
    nq = pn.nlp.shape["q"]
    compute_CoM = biorbd.to_casadi_func("CoM", pn.nlp.model.CoM, pn.nlp.q)
    com = compute_CoM(pn.x[0][:nq])
    return com[2]


# OPTIMAL CONTROL PROBLEM
model = biorbd.Model("Modeles_S2M/2legs_18dof_flatfootR.bioMod")

# --- Problem parameters --- #
nb_q = model.nbQ()
nb_qdot = model.nbQdot()
nb_tau = model.nbGeneralizedTorque()
nb_mus = model.nbMuscleTotal()
nb_shooting = 30
final_time=1.0
min_bound, max_bound = 0, np.inf
torque_min, torque_max, torque_init = -1000, 1000, 0
activation_min, activation_max, activation_init = 1e-3, 1.0, 0.1

# --- Subject positions and initial trajectories --- #
position_zeros = [0]*nb_q
position_high = [[0], [-0.07], [0], [0], [0], [-0.4],
                [0], [0], [0.37], [-0.13], [0], [0.11],
                [0], [0], [0.37], [-0.13], [0], [0.11]]
position_low = [-0.06, -0.36, 0, 0, 0, -0.8,
                0, 0, 1.53, -1.55, 0, 0.68,
                0, 0, 1.53, -1.55, 0, 0.68]

q_init = np.zeros((nb_q, nb_shooting + 1))
for i in range(nb_q):
    q_init[i, :int(nb_shooting/2)] = np.linspace(position_high[i], position_low[i], int(nb_shooting/2)).squeeze()
    q_init[i, int(nb_shooting/2):] = np.linspace(position_low[i], position_high[i], int(nb_shooting/2) + 1).squeeze()
qdot_init = np.gradient(q_init)[1]
qddot_init = np.gradient(qdot_init)[1]

# --- Compute CoM position --- #
symbolic_q = MX.sym("q", nb_q, 1)
compute_CoM = Function(
    "ComputeCoM",
    [symbolic_q],
    [model.CoM(symbolic_q).to_mx()],
    ["q"],
    ["CoM"],
).expand()

CoM_high = compute_CoM(np.array(position_high))
CoM_low = compute_CoM(np.array(position_low))

# # --- Animate model --- #
# b = bioviz.Viz(loaded_model=model, show_muscles=False, show_segments_center_of_mass=False, show_local_ref_frame=False)
# b.set_q(np.array(position_high).squeeze())
# # b.set_q(np.array(position_high))
# b.load_movement(q_init)
# b.exec()


# --- Objective function --- #
objective_functions = ObjectiveList()
# objective_functions.add(custom_CoM_low,
#                         custom_type=ObjectiveFcn.Mayer,
#                         node=Node.ALL,
#                         quadratic=True,
#                         weight=1000)
# objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE,
#                         quadratic=True,
#                         node=Node.ALL,
#                         index=(0,1,2,5,8,9,11,14,15,17),
#                         weight=0.1)
# objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE,
#                         quadratic=True,
#                         node=Node.ALL,
#                         index=(3,4,6,7,10,12,13,16),
#                         weight=1)
# symmetry
idx_minus = (6,7,10)
for i in idx_minus:
    objective_functions.add(ObjectiveFcn.Lagrange.PROPORTIONAL_STATE,
                            node=Node.ALL,
                            first_dof=i,
                            second_dof=i+6,
                            coef=-1)
idx_plus = (8,9,11)
for i in idx_plus:
    objective_functions.add(ObjectiveFcn.Lagrange.PROPORTIONAL_STATE,
                            node=Node.ALL,
                            first_dof=i,
                            second_dof=i+6,
                            coef=1)

objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE,
                        quadratic=True,
                        node=Node.ALL,
                        weight=1)
objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_MUSCLES_CONTROL,
                        quadratic=True,
                        node=Node.ALL,
                        weight=10)

# --- Dynamics --- #
dynamics = DynamicsList()
dynamics.add(DynamicsFcn.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN_WITH_CONTACT)

# --- Constraints --- #
constraints = ConstraintList()
contact_z_axes = (2, 3, 5, 8, 9, 11)
for c in contact_z_axes:
    constraints.add( # positive vertical forces
        ConstraintFcn.CONTACT_FORCE,
        min_bound=0,
        max_bound=np.inf,
        node=Node.ALL,
        contact_force_idx=c,
    )
constraints.add(
    custom_CoM_low,
    min_bound=-0.35,
    max_bound=-0.25,
    node=Node.MID,
)

# --- Path constraints --- #
x_bounds = BoundsList()
x_bounds.add(bounds=QAndQDotBounds(model))
x_bounds[0].min[:nb_q, 0] = np.array(position_high).squeeze()
x_bounds[0].max[:nb_q, 0] = np.array(position_high).squeeze()

x_bounds[0].min[:nb_q, -1] = np.array(position_high).squeeze()
x_bounds[0].max[:nb_q, -1] = np.array(position_high).squeeze()

u_bounds = BoundsList()
u_bounds.add(
            [torque_min] * nb_tau + [activation_min] * nb_mus,
            [torque_max] * nb_tau + [activation_max] * nb_mus,
)

# # --- Initial guess --- #
# x_init = InitialGuessList()
# init_x = np.zeros((nb_q + nb_qdot, nb_shooting + 1))
# init_x[:nb_q, :] = q_init
# init_x[nb_q:, :] = qdot_init
# x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)
#
# u_init = InitialGuessList()
# u_init.add([torque_init]*nb_tau + [activation_init]*nb_mus)

# Load previous solution
save_path = './RES/muscle_driven/'
x_init = InitialGuessList()
init_x = np.zeros((nb_q + nb_qdot, nb_shooting + 1))
init_x[:nb_q, :] = np.load(save_path + "q.npy")
init_x[nb_q:, :] = np.load(save_path + "qdot.npy")
x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)

u_init = InitialGuessList()
init_u = np.zeros((nb_tau + nb_mus, nb_shooting))
init_u[:nb_tau, :] = np.load(save_path + "tau.npy")[:, :-1]
init_u[nb_tau:, :] = np.load(save_path + "muscle.npy")[:, :-1]
u_init.add(init_u, interpolation=InterpolationType.EACH_FRAME)
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
    n_threads=4,
)

# path_previous = './RES/muscle_driven/cycle.bo'
# ocp_previous, sol_previous = ocp.load(path_previous)
# # --- Plot Results --- #
# Affichage_resultat = Affichage(ocp_previous, sol_previous, muscles=True)
# Affichage_resultat.plot_q_symetry()
# Affichage_resultat.plot_tau_symetry()
# Affichage_resultat.plot_qdot_symetry()
# Affichage_resultat.plot_forces()


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

# --- Get Results --- #
q = sol.states["q"]
q_dot = sol.states["qdot"]
tau = sol.controls["tau"]
muscle = sol.controls["muscles"]

# --- Plot CoM --- #
CoM = np.zeros((3, q.shape[1]))
for n in range(nb_shooting + 1):
    CoM[:, n:n+1] = compute_CoM(q[:, n])
plt.figure()
plt.plot(CoM[2, :])

# --- Plot Results --- #
Affichage_resultat = Affichage(ocp, sol, muscles=True)
Affichage_resultat.plot_q_symetry()
Affichage_resultat.plot_forces()


# --- Save results ---
save_path = './RES/muscle_driven/'
ocp.save(sol, save_path + 'cycle.bo')
np.save(save_path + 'qdot', q_dot)
np.save(save_path + 'q', q)
np.save(save_path + 'tau', tau)
np.save(save_path + 'muscle', muscle)

# --- Show results --- #
sol.animate()
sol.print()
sol.graphs()


