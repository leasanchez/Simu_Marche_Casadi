import numpy as np
from casadi import dot, Function, vertcat, MX, mtimes, nlpsol, mmax
import biorbd
import bioviz
from time import time
from matplotlib import pyplot as plt
from Compute_Results.Plot_results import Affichage

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
    BidirectionalMapping,
)

def set_objectif_function(objective_functions):
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE,
                            quadratic=True,
                            node=Node.ALL,
                            weight=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_MUSCLES_CONTROL,
                            quadratic=True,
                            node=Node.ALL,
                            weight=10)
    # symmetry
    objective_functions.add(ObjectiveFcn.Lagrange.PROPORTIONAL_STATE,
                            node=Node.ALL,
                            first_dof=6,
                            second_dof=12,
                            coef=-1,
                            weight=100000,
                            quadratic=True)
    # objective_functions.add(ObjectiveFcn.Lagrange.PROPORTIONAL_STATE,
    #                         node=Node.ALL,
    #                         first_dof=7,
    #                         second_dof=13,
    #                         coef=-1,
    #                         weight=1000)
    # objective_functions.add(ObjectiveFcn.Lagrange.PROPORTIONAL_STATE,
    #                         node=Node.ALL,
    #                         first_dof=10,
    #                         second_dof=16,
    #                         coef=-1,
    #                         weight=1000)

    # idx_minus = (6, 7, 10)
    # for i in idx_minus:
    #     objective_functions.add(ObjectiveFcn.Lagrange.PROPORTIONAL_STATE,
    #                             node=Node.ALL,
    #                             first_dof=i,
    #                             second_dof=(i + 6),
    #                             coef=-1,
    #                             weight=1000)
    # idx_plus = (8, 9, 11)
    # for i in idx_plus:
    #     objective_functions.add(ObjectiveFcn.Lagrange.PROPORTIONAL_STATE,
    #                             node=Node.ALL,
    #                             first_dof=i,
    #                             second_dof=i + 6,
    #                             coef=1,
    #                             weight=1000)
    return objective_functions


def set_dynamics(dynamics):
    dynamics.add(DynamicsFcn.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN_WITH_CONTACT)
    return dynamics

def set_constraints(constraints):
    # --- contact forces --- #
    contact_z_axes = (2, 3, 5, 8, 9, 11)
    for c in contact_z_axes:
        constraints.add(  # positive vertical forces
            ConstraintFcn.CONTACT_FORCE,
            min_bound=0,
            max_bound=np.inf,
            node=Node.ALL,
            contact_force_idx=c,
        )
    constraints.add(
        get_last_contact_force,
        contact_force_idx=contact_z_axes,
        min_bound=0,
        max_bound=np.inf,
        node=Node.ALL,
    )

    # --- CoM displacements --- #
    constraints.add(
        custom_CoM_low,
        min_bound=-0.35,
        max_bound=-0.25,
        node=Node.MID,
    )

    # # --- pelvis rotations --- #
    # constraints.add(
    #     ConstraintFcn.TRACK_STATE,
    #     index=(3,4),
    #     node=Node.ALL,
    # )
    return constraints


def set_bounds(model, x_bounds, u_bounds, position_init):
    torque_min, torque_max = -1000, 1000
    activation_min, activation_max = 1e-3, 1.0

    x_bounds.add(bounds=QAndQDotBounds(model))
    x_bounds[0].min[:nb_q, 0] = np.array(position_init).squeeze()
    x_bounds[0].max[:nb_q, 0] = np.array(position_init).squeeze()

    x_bounds[0].min[:nb_q, -1] = np.array(position_init).squeeze()
    x_bounds[0].max[:nb_q, -1] = np.array(position_init).squeeze()

    u_bounds.add(
        [torque_min] * nb_tau + [activation_min] * nb_mus,
        [torque_max] * nb_tau + [activation_max] * nb_mus,
    )
    return x_bounds, u_bounds

def set_initial_guess(x_init, u_init, position_high, position_low, nb_shooting):
    init_x = np.zeros((model.nbQ() + model.nbQdot(), nb_shooting + 1))
    for i in range(model.nbQ()):
        init_x[i, :] = np.concatenate((np.linspace(position_high[i], position_low[i], int(nb_shooting/2)),
                                       np.linspace(position_low[i], position_high[i], int(nb_shooting/2) + 1))).squeeze()
    init_x[model.nbQ():, :] = np.gradient(init_x[:model.nbQ(), :])[0]
    x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)

    u_init.add([0]*nb_tau + [0.1]*model.nbMuscleTotal())
    return x_init, u_init

def set_initial_guess_from_previous_solution(x_init, u_init, save_path):
    init_x = np.zeros((nb_q + nb_qdot, nb_shooting + 1))
    init_x[:nb_q, :] = np.load(save_path + "q.npy")
    init_x[nb_q:, :] = np.load(save_path + "qdot.npy")
    x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)

    init_u = np.zeros((nb_tau + nb_mus, nb_shooting))
    init_u[:nb_tau, :] = np.load(save_path + "tau.npy")[:, :-1]
    init_u[nb_tau:, :] = np.load(save_path + "muscle.npy")[:, :-1]
    u_init.add(init_u, interpolation=InterpolationType.EACH_FRAME)
    return x_init, u_init

def set_mapping():
    u_mapping = BidirectionalMapping([None, None, None, None, None, None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                                     [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
    return u_mapping


def get_results(sol):
    q = sol.states["q"]
    qdot = sol.states["qdot"]
    tau = sol.controls["tau"]
    muscle = sol.controls["muscles"]
    return q, qdot, tau, muscle

def save_results(ocp, sol, save_path):
    q, qdot, tau, muscle = get_results(sol)
    ocp.save(sol, save_path + 'cycle.bo')
    np.save(save_path + 'qdot', qdot)
    np.save(save_path + 'q', q)
    np.save(save_path + 'tau', tau)
    np.save(save_path + 'muscle', muscle)

def custom_CoM_low(pn: PenaltyNodes) -> MX:
    nq = pn.nlp.shape["q"]
    compute_CoM = biorbd.to_casadi_func("CoM", pn.nlp.model.CoM, pn.nlp.q)
    com = compute_CoM(pn.x[0][:nq])
    return com[2]

def get_last_contact_force(pn: PenaltyNodes, contact_force_idx) -> MX:
    force = pn.nlp.contact_forces_func(pn.x[-1], pn.u[-1], pn.p)
    return force[contact_force_idx]





# OPTIMAL CONTROL PROBLEM
model = biorbd.Model("Modeles_S2M/2legs_18dof_flatfootR.bioMod")

# --- Problem parameters --- #
nb_q = model.nbQ()
nb_qdot = model.nbQdot()
nb_tau = model.nbGeneralizedTorque() #- model.nbRoot()
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


# --- Objective function --- #
objective_functions = ObjectiveList()
objective_functions = set_objectif_function(objective_functions)

# --- Dynamics --- #
dynamics = DynamicsList()
dynamics = set_dynamics(dynamics)

# --- Constraints --- #
constraints = ConstraintList()
constraints = set_constraints(constraints)

# --- Path constraints --- #
x_bounds = BoundsList()
u_bounds = BoundsList()
x_bounds, u_bounds = set_bounds(model, x_bounds, u_bounds, position_high)

# --- Remove root actuation --- #
# u_mapping = set_mapping()

# --- Initial guess --- #
x_init = InitialGuessList()
u_init = InitialGuessList()
# x_init, u_init = set_initial_guess(x_init, u_init, position_high, position_low, nb_shooting)
x_init, u_init = set_initial_guess_from_previous_solution(x_init, u_init, save_path='./RES/muscle_driven/pelvis_cstr/')

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
    # tau_mapping=u_mapping,
)

# path_previous = './RES/muscle_driven/symetry/cycle.bo'
# ocp_sym, sol_sym = ocp.load(path_previous)
# # --- Plot Results --- #
# Affichage_resultat_sym = Affichage(ocp_sym, sol_sym, muscles=True)
# Affichage_resultat_sym.plot_q_symetry()
# Affichage_resultat_sym.plot_tau_symetry()
# Affichage_resultat_sym.plot_qdot_symetry()
# Affichage_resultat_sym.plot_individual_forces()
# Affichage_resultat_sym.plot_sum_forces()
# Affichage_resultat_sym.plot_muscles_symetry()
#
# path_previous_cstr = './RES/muscle_driven/pelvis_cstr/cycle.bo'
# ocp_pel, sol_pel = ocp.load(path_previous_cstr)
# # --- Plot Results --- #
# Affichage_resultat_pel = Affichage(ocp_pel, sol_pel, muscles=True)
# Affichage_resultat_pel.plot_q_symetry()
# Affichage_resultat_pel.plot_tau_symetry()
# Affichage_resultat_pel.plot_qdot_symetry()
# Affichage_resultat_pel.plot_forces()
# Affichage_resultat_pel.plot_muscles_symetry()


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
q, qdot, tau, muscle = get_results(sol)

# --- Plot CoM --- #
CoM = np.zeros((3, q.shape[1]))
for n in range(nb_shooting + 1):
    CoM[:, n:n+1] = compute_CoM(q[:, n])
plt.figure()
plt.plot(CoM[2, :])

# --- Plot Results --- #
Affichage_resultat = Affichage(ocp, sol, muscles=True)
Affichage_resultat.plot_q_symetry()
Affichage_resultat.plot_tau_symetry()
Affichage_resultat.plot_qdot_symetry()
Affichage_resultat.plot_forces()
Affichage_resultat.plot_muscles_symetry()

# --- Save results ---
save_path = './RES/muscle_driven/symetry/'
save_results(ocp, sol, save_path)

# --- Show results --- #
sol.animate()
sol.print()
sol.graphs()


