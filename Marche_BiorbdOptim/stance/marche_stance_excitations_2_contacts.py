import numpy as np
from casadi import dot, Function, vertcat, MX
from matplotlib import pyplot as plt
import biorbd
from time import time
from Marche_BiorbdOptim.LoadData import Data_to_track

from biorbd_optim import (
    Instant,
    OptimalControlProgram,
    ProblemType,
    Objective,
    Bounds,
    QAndQDotBounds,
    InitialConditions,
    ShowResult,
    Data,
    InterpolationType,
    PlotType,
    Axe,
    Constraint,
    StateTransition,
)


def get_last_contact_forces(ocp, nlp, t, x, u, data_to_track=()):
    force = nlp["contact_forces_func"](x[-1], u[-1])
    val = force - data_to_track[t[-1], :]
    return dot(val, val)


def get_muscles_first_node(ocp, nlp, t, x, u):
    activation = x[0][2 * nlp["nbQ"] :]
    excitation = u[0][nlp["nbQ"] :]
    val = activation - excitation
    return val


def prepare_ocp(
    biorbd_model, final_time, nb_shooting, markers_ref, excitation_ref, q_ref, grf_ref,
):
    # Problem parameters
    nb_phases = len(biorbd_model)
    nb_q = biorbd_model[0].nbQ()
    nb_qdot = biorbd_model[0].nbQdot()
    nb_mus = biorbd_model[0].nbMuscleTotal()
    nb_tau = biorbd_model[0].nbGeneralizedTorque()

    torque_min, torque_max, torque_init = -1000, 1000, 0
    activation_min, activation_max, activation_init = 0, 1, 0.1

    # Add objective functions
    objective_functions = (
        (
        {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 1, "controls_idx": range(6, nb_tau)},
        # {"type": Objective.Lagrange.TRACK_MUSCLES_CONTROL, "weight": 1, "data_to_track": excitation_ref[0][:, :-1].T},
        {"type": Objective.Lagrange.TRACK_MARKERS, "weight": 100, "data_to_track": markers_ref[0]},
        # {"type": Objective.Lagrange.TRACK_STATE, "weight": 0.01, "states_idx": [0, 1, 5, 8, 9, 10], "data_to_track": q_ref.T},
        # {"type": Objective.Lagrange.TRACK_CONTACT_FORCES, "weight": 0.00005, "data_to_track": grf_ref.T},
    ),
    (
        {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 1, "controls_idx": range(6, nb_tau)},
        # {"type": Objective.Lagrange.TRACK_MUSCLES_CONTROL, "weight": 1, "data_to_track": excitation_ref[1][:, :-1].T},
        {"type": Objective.Lagrange.TRACK_MARKERS, "weight": 100, "data_to_track": markers_ref[1]},
        # {"type": Objective.Lagrange.TRACK_STATE, "weight": 0.01, "states_idx": [0, 1, 5, 8, 9, 10], "data_to_track": q_ref.T},
        # {"type": Objective.Lagrange.TRACK_CONTACT_FORCES, "weight": 0.00005, "data_to_track": grf_ref.T},
    ),
    (
        {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 1, "controls_idx": range(6, nb_tau)},
        # {"type": Objective.Lagrange.TRACK_MUSCLES_CONTROL, "weight": 1, "data_to_track": excitation_ref[2][:, :-1].T},
        {"type": Objective.Lagrange.TRACK_MARKERS, "weight": 100, "data_to_track": markers_ref[2]},
        # {"type": Objective.Lagrange.TRACK_STATE, "weight": 0.01, "states_idx": [0, 1, 5, 8, 9, 10], "data_to_track": q_ref.T},
        # {"type": Objective.Lagrange.TRACK_CONTACT_FORCES, "weight": 0.00005, "data_to_track": grf_ref.T},
        # {
        #     "type": Objective.Mayer.CUSTOM,
        #     "weight": 0.00005,
        #     "function": get_last_contact_forces,
        #     "data_to_track": grf_ref.T,
        #     "instant": Instant.ALL,
        # },
    ))

    # Dynamics
    problem_type = (
        ProblemType.torque_driven_with_contact,
        ProblemType.torque_driven_with_contact,
        ProblemType.torque_driven_with_contact,
    )

    # Constraints
    constraints = ()

    # Path constraint
    X_bounds = []
    for i in range(nb_phases):
        XB = QAndQDotBounds(biorbd_model[i])
        # XB.concatenate(Bounds([activation_min] * nb_mus, [activation_max] * nb_mus))
        X_bounds.append(XB)

    U_bounds = []
    for i in range(nb_phases):
        UB = Bounds(
            min_bound=[torque_min] * nb_tau, # + [activation_min] * nb_mus,
            max_bound=[torque_max] * nb_tau, # + [activation_max] * nb_mus,
        )
        U_bounds.append(UB)

    # Initial guess
    X_init = []
    for n_p in range(nb_phases):
        init_x = np.zeros(((nb_q + nb_qdot), nb_shooting[n_p] + 1)) # + nb_mus
        init_x[[0, 1, 5, 8, 9, 11], :] = q_ref[n_p]
            # init_x[-nb_mus:, i] = excitation_ref[n_p][:, i]
        XI = InitialConditions(init_x, interpolation_type=InterpolationType.EACH_FRAME)
        X_init.append(XI)

    U_init = []
    for n_p in range(nb_phases):
        init_u = np.zeros(((nb_tau), nb_shooting[n_p])) #  + nb_mus
        init_u[1, :] = np.repeat(-500, nb_shooting[n_p])
            # init_u[-nb_mus:, i] = excitation_ref[n_p][:, i]
        UI = InitialConditions(init_u, interpolation_type=InterpolationType.EACH_FRAME)
        U_init.append(UI)


    state_transitions = (
        {"type": StateTransition.IMPACT, "phase_pre_idx": 0,},  # Heel Strike to Flat foot
    )

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        problem_type,
        nb_shooting,
        final_time,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        objective_functions,
        constraints,
        state_transitions = state_transitions,
    )


if __name__ == "__main__":
    # Define the problem
    biorbd_model = (
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_1contact_deGroote_3d_Heel.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_2contacts_deGroote_3d.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_1contact_deGroote_3d_Forefoot.bioMod"),
    )
    model_q = biorbd.Model("../../ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact_deGroote.bioMod")
    number_shooting_points = [5, 10, 15]

    # Generate data from file
    Data_to_track = Data_to_track(name_subject="normal01", multiple_contact=True)
    [T, T_stance, T_swing] = Data_to_track.GetTime()
    phase_time = T_stance

    grf_ref = Data_to_track.load_data_GRF(biorbd_model[0], T_stance, number_shooting_points)  # get ground reaction forces

    markers_ref = Data_to_track.load_data_markers(biorbd_model[0], T_stance, number_shooting_points, "stance")
    q_ref = Data_to_track.load_data_q(model_q, T_stance, number_shooting_points, "stance")
    emg_ref = Data_to_track.load_data_emg(biorbd_model[0], T_stance, number_shooting_points, "stance")
    excitation_ref = []
    for i in range(len(phase_time)):
        excitation_ref.append(Data_to_track.load_muscularExcitation(emg_ref[i]))

    # Track these data
    ocp = prepare_ocp(
        biorbd_model,
        phase_time,
        number_shooting_points,
        markers_ref,
        excitation_ref,
        q_ref,
        grf_ref,
    )

    # ocp.add_plot("q", lambda x, u: q_ref, PlotType.STEP, axes_idx=[0, 1, 5, 8, 9, 10])

    # --- Solve the program --- #
    tic = time()
    sol = ocp.solve(
        solver="ipopt",
        options_ipopt={
            "ipopt.tol": 1e-2,
            "ipopt.max_iter": 5000,
            "ipopt.hessian_approximation": "exact",
            "ipopt.limited_memory_max_history": 50,
            "ipopt.linear_solver": "ma57",
        },
        show_online_optim=False,
    )
    toc = time() - tic
    print(f"Time to solve : {toc}sec")

    # --- Save the optimal control program and the solution --- #
    ocp.save(sol, "marche_stance_excitation_2_contacts")

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
    result.graphs()
