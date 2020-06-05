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
    Constraint,
    Bounds,
    QAndQDotBounds,
    InitialConditions,
    ShowResult,
    Data,
    InterpolationType,
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


def get_initial_value():
    q_ig = []
    qdot_ig = []
    activations_ig = []
    tau_ig = []
    excitations_ig = []

    ocp_load_swing, sol_load_swing = OptimalControlProgram.load(
        "../swing/RES/excitations/3D/marche_swing_excitation.bo"
    )
    ocp_load_stance, sol_load_stance = OptimalControlProgram.load(
        "../stance/RES/equincocont01/excitations/3D/marche_stance_excitation.bo"
    )
    ocp_load = [ocp_load_stance, ocp_load_swing]
    sol_load = [sol_load_stance, sol_load_swing]

    for i in range(len(ocp_load)):
        states_sol, controls_sol = Data.get_data(ocp_load[i], sol_load[i]["x"])
        q_ig.append(states_sol["q"])
        qdot_ig.append(states_sol["q_dot"])
        activations_ig.append(states_sol["muscles"])
        tau_ig.append(controls_sol["tau"])
        excitations_ig.append(controls_sol["muscles"])
    return q_ig, qdot_ig, activations_ig, tau_ig, excitations_ig


def prepare_ocp(
    biorbd_model, final_time, nb_shooting, markers_ref, excitation_ref, grf_ref, q_ref,
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
            # {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 1, "controls_idx":[3, 4, 5]},
            # {"type": Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, "weight": 1, "data_to_track":excitation_ref[0].T},
            {"type": Objective.Lagrange.TRACK_MARKERS, "weight": 100, "data_to_track": markers_ref[0]},
            # {"type": Objective.Lagrange.TRACK_STATE, "weight": 0.1, "states_idx": [0, 1, 5, 8, 9, 10], "data_to_track": q_ref[0].T},
            # {"type": Objective.Lagrange.TRACK_CONTACT_FORCES, "weight": 0.00005, "data_to_track": grf_ref[:, :-1].T},
            # {"type": Objective.Mayer.CUSTOM, "function": get_last_contact_forces, "data_to_track":grf_ref.T, "weight": 0.00005, "instant": Instant.ALL}
        ),
        (
            # {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 1, "controls_idx":[3, 4, 5]},
            # {"type": Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, "weight": 1, "data_to_track":excitation_ref[1].T},
            {"type": Objective.Lagrange.TRACK_MARKERS, "weight": 100, "data_to_track": markers_ref[1]},
            # {"type": Objective.Lagrange.TRACK_STATE, "weight": 0.1, "states_idx": [0, 1, 5, 8, 9, 10], "data_to_track": q_ref[1].T},
        ),
    )

    # Dynamics
    problem_type = (
        ProblemType.muscle_excitations_and_torque_driven_with_contact,
        ProblemType.muscle_excitations_and_torque_driven,
    )

    # Constraints
    constraints = ()

    # Path constraint
    X_bounds = []
    for i in range(nb_phases):
        XB = QAndQDotBounds(biorbd_model[i])
        XB.concatenate(
            Bounds([activation_min] * biorbd_model[i].nbMuscles(), [activation_max] * biorbd_model[i].nbMuscles())
        )
        X_bounds.append(XB)

    # Initial guess
    q_ig, qdot_ig, activations_ig, tau_ig, excitations_ig = get_initial_value()
    X_init = []
    for n_p in range(nb_phases):
        init_x = np.zeros(
            (
                biorbd_model[n_p].nbQ() + biorbd_model[n_p].nbQdot() + biorbd_model[n_p].nbMuscleTotal(),
                nb_shooting[n_p] + 1,
            )
        )
        for i in range(nb_shooting[n_p] + 1):
            init_x[:nb_q, i] = q_ig[n_p][:, i]
            init_x[nb_q : nb_q + nb_qdot, i] = qdot_ig[n_p][:, i]
            init_x[-nb_mus:, i] = activations_ig[n_p][:, i]
        XI = InitialConditions(init_x, interpolation_type=InterpolationType.EACH_FRAME)
        X_init.append(XI)

    # X_init = []
    # for n_p in range(nb_phases):
    #     init_x = np.zeros((biorbd_model[n_p].nbQ() + biorbd_model[n_p].nbQdot() + biorbd_model[n_p].nbMuscleTotal(), nb_shooting[n_p] + 1))
    #     for i in range(nb_shooting[n_p] + 1):
    #         init_x[:biorbd_model[n_p].nbQ(), i] = q_ref[n_p][:, i]
    #         init_x[-biorbd_model[n_p].nbMuscleTotal():, i] = excitation_ref[n_p][:, i]
    #     XI = InitialConditions(init_x, interpolation_type=InterpolationType.EACH_FRAME)
    #     X_init.append(XI)

    # Define control path constraint
    U_bounds = [
        Bounds(
            min_bound=[torque_min] * biorbd_model[i].nbGeneralizedTorque()
            + [activation_min] * biorbd_model[i].nbMuscleTotal(),
            max_bound=[torque_max] * biorbd_model[i].nbGeneralizedTorque()
            + [activation_max] * biorbd_model[i].nbMuscleTotal(),
        )
        for i in range(nb_phases)
    ]

    # Initial guess
    U_init = []
    for n_p in range(nb_phases):
        init_u = np.zeros(
            (biorbd_model[n_p].nbGeneralizedTorque() + biorbd_model[n_p].nbMuscleTotal(), nb_shooting[n_p])
        )
        for i in range(nb_shooting[n_p]):
            init_u[:nb_tau, i] = tau_ig[n_p][:, i]
            init_u[-nb_mus:, i] = excitations_ig[n_p][:, i]
        UI = InitialConditions(init_u, interpolation_type=InterpolationType.EACH_FRAME)
        U_init.append(UI)
    # U_init = []
    # for n_p in range(nb_phases):
    #     init_u = np.zeros((biorbd_model[n_p].nbGeneralizedTorque() + biorbd_model[n_p].nbMuscleTotal(), nb_shooting[n_p]))
    #     for i in range(nb_shooting[n_p]):
    #         if n_p == 0:
    #             init_u[1, i] = -500
    #         init_u[-biorbd_model[n_p].nbMuscleTotal():, i] = excitation_ref[n_p][:, i]
    #     UI = InitialConditions(init_u, interpolation_type=InterpolationType.EACH_FRAME)
    #     U_init.append(UI)

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
    )


if __name__ == "__main__":
    # Define the problem
    # Model path
    biorbd_model = (
        biorbd.Model("../../ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact_deGroote_3d.bioMod"),
        biorbd.Model("../../ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact_deGroote_3d.bioMod"),
    )

    # Problem parameters
    number_shooting_points = [25, 25]

    # Generate data from file
    Data_to_track = Data_to_track("equicocont01", multiple_contact=True)
    [T, T_stance, T_swing] = Data_to_track.GetTime()
    phase_time = [T_stance[0], T_stance[1], T_stance[2], T_swing] # get time for each phase

    grf_ref = Data_to_track.load_data_GRF(biorbd_model[0], T_stance, number_shooting_points[:-1]) # get ground reaction forces

    markers_ref = Data_to_track.load_data_markers(biorbd_model[0],T_stance,number_shooting_points[:-1], "stance")
    markers_ref.append(Data_to_track.load_data_markers(biorbd_model[-1], phase_time[-1], number_shooting_points[-1], "swing")) # get markers position

    q_ref = Data_to_track.load_data_q(biorbd_model[0],T_stance,number_shooting_points[:-1],"stance")
    q_ref.append(Data_to_track.load_data_q(biorbd_model[-1], phase_time[-1], number_shooting_points[-1], "swing")) # get q from kalman

    emg_ref = Data_to_track.load_data_emg(biorbd_model[0], T_stance,number_shooting_points[:-1],"stance")
    emg_ref.append(Data_to_track.load_data_emg(biorbd_model[-1], phase_time[-1], number_shooting_points[-1], "swing")) # get emg

    excitation_ref = []
    for i in range(len(phase_time)):
        excitation_ref.append(Data_to_track.load_muscularExcitation(emg_ref[i]))

    # Track these data
    biorbd_model = (
        biorbd.Model("../../ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact_deGroote_3d.bioMod"),
        biorbd.Model("../../ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact_deGroote_3d.bioMod"),
    )
    ocp = prepare_ocp(
        biorbd_model,
        phase_time,
        number_shooting_points,
        markers_ref=[markers_ref_stance, markers_ref_swing],
        excitation_ref=[excitation_ref_stance, excitation_ref_swing],
        grf_ref=grf_ref[[1, 0, 2], :],
        q_ref=[Q_ref_stance, Q_ref_swing],
    )

    # --- Add plot kalman --- #
    # q_ref = np.zeros((model_q.nbQ(), number_shooting_points[0] + number_shooting_points[1] + 1))
    # q_ref[:, :number_shooting_points[0]+1] = q_ref_stance
    # q_ref[:, number_shooting_points[0]:number_shooting_points[0] + number_shooting_points[1] + 1] = q_ref_swing
    # ocp.add_plot("q", lambda x, u: q_ref, PlotType.STEP, axes_idx=[0, 1, 5, 8, 9, 10])

    # --- Solve the program --- #
    tic = time()
    sol = ocp.solve(
        solver="ipopt",
        options_ipopt={
            "ipopt.tol": 1e-2,
            "ipopt.max_iter": 5000,
            "ipopt.hessian_approximation": "limited-memory",
            "ipopt.limited_memory_max_history": 50,
            "ipopt.linear_solver": "ma57",
        },
        show_online_optim=False,
    )
    toc = time() - tic
    print(f"Time to solve : {toc}sec")

    # --- Get Results --- #
    states_sol, controls_sol = Data.get_data(ocp, sol["x"])
    q = states_sol["q"]
    q_dot = states_sol["q_dot"]
    activations = states_sol["muscles"]
    tau = controls_sol["tau"]
    excitations = controls_sol["muscles"]

    # # --- Get markers position from q_sol and q_ref --- #
    # nb_markers = biorbd_model[0].nbMarkers()
    # nb_q = biorbd_model[0].nbQ()
    #
    # markers_sol = np.ndarray((3, nb_markers, ocp.nlp[0]["ns"] + 1))
    # markers_from_q_ref = np.ndarray((3, nb_markers, ocp.nlp[0]["ns"] + 1))
    # markers_ref = np.ndarray((3, nb_markers, ocp.nlp[0]["ns"] + 1))
    # markers_ref[:, : ocp.nlp[0]["ns"] + 1] = markers_ref_stance
    # markers_ref[:, ocp.nlp[0]["ns"]: ocp.nlp[0]["ns"] + ocp.nlp[1]["ns"] + 1] = markers_ref_swing
    #
    # markers_func = []
    # symbolic_states = MX.sym("x", ocp.nlp["nx"], 1)
    # symbolic_controls = MX.sym("u", ocp.nlp["nu"], 1)
    # for i in range(nb_markers):
    #     markers_func.append(
    #         Function(
    #             "ForwardKin",
    #             [symbolic_states],
    #             [biorbd_model[0].marker(symbolic_states[:nb_q], i).to_mx()],
    #             ["q"],
    #             ["marker_" + str(i)],
    #         ).expand()
    #     )
    # for i in range(ocp.nlp[0]['ns']):
    #     for j, mark_func in enumerate(markers_func):
    #         markers_sol[:, j, i] = np.array(mark_func(vertcat(q[:, i], q_dot[:, i]))).squeeze()
    #         Q_ref = np.concatenate([q_ref[:, i], np.zeros(nb_q)])
    #         markers_from_q_ref[:, j, i] = np.array(mark_func(Q_ref)).squeeze()
    #
    # diff_track = (markers_sol - markers_ref) * (markers_sol - markers_ref)
    # diff_sol = (markers_sol - markers_from_q_ref) * (markers_sol - markers_from_q_ref)
    # hist_diff_track = np.zeros((3, nb_markers))
    # hist_diff_sol = np.zeros((3, nb_markers))
    #
    # for n_mark in range(nb_markers):
    #     hist_diff_track[0, n_mark] = sum(diff_track[0, n_mark, :]) / nb_markers
    #     hist_diff_track[1, n_mark] = sum(diff_track[1, n_mark, :]) / nb_markers
    #     hist_diff_track[2, n_mark] = sum(diff_track[2, n_mark, :]) / nb_markers
    #
    #     hist_diff_sol[0, n_mark] = sum(diff_sol[0, n_mark, :]) / nb_markers
    #     hist_diff_sol[1, n_mark] = sum(diff_sol[1, n_mark, :]) / nb_markers
    #     hist_diff_sol[2, n_mark] = sum(diff_sol[2, n_mark, :]) / nb_markers
    #
    # mean_diff_track = [sum(hist_diff_track[0, :]) / nb_markers,
    #                    sum(hist_diff_track[1, :]) / nb_markers,
    #                    sum(hist_diff_track[2, :]) / nb_markers]
    # mean_diff_sol = [sum(hist_diff_sol[0, :]) / nb_markers,
    #                  sum(hist_diff_sol[1, :]) / nb_markers,
    #                  sum(hist_diff_sol[2, :]) / nb_markers]
    #
    # # --- Plot --- #
    # t = np.hstack([t_stance[:-1], T_stance + t_swing])
    #
    # plt.show()

    ocp.save(sol, "marche_gait_equin_excitation")
    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
    result.graphs()
