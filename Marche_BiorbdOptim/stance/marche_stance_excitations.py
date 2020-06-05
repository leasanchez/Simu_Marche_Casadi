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
    Constraint,
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
    biorbd_model, final_time, nb_shooting, markers_ref, excitation_ref, q_ref, grf_ref, nb_threads,
):
    # Problem parameters
    nb_q = biorbd_model.nbQ()
    nb_qdot = biorbd_model.nbQdot()
    nb_mus = biorbd_model.nbMuscleTotal()

    torque_min, torque_max, torque_init = -1000, 1000, 0
    activation_min, activation_max, activation_init = 0, 1, 0.1

    # Add objective functions
    objective_functions = (
        {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 1, "controls_idx": range(6, 11)},
        {"type": Objective.Lagrange.TRACK_MUSCLES_CONTROL, "weight": 1, "data_to_track": excitation_ref[:, :-1].T},
        {"type": Objective.Lagrange.TRACK_MARKERS, "weight": 100, "data_to_track": markers_ref},
        # {"type": Objective.Lagrange.TRACK_STATE, "weight": 0.01, "states_idx": [0, 1, 5, 8, 9, 10], "data_to_track": q_ref.T},
        {"type": Objective.Lagrange.TRACK_CONTACT_FORCES, "weight": 0.00005, "data_to_track": grf_ref.T},
        {
            "type": Objective.Mayer.CUSTOM,
            "weight": 0.00005,
            "function": get_last_contact_forces,
            "data_to_track": grf_ref.T,
            "instant": Instant.ALL,
        },
    )

    # Dynamics
    variable_type = ProblemType.muscle_excitations_and_torque_driven_with_contact

    # Constraints
    constraints = {"type": Constraint.CUSTOM, "function": get_muscles_first_node, "instant": Instant.START}

    # Path constraint
    X_bounds = QAndQDotBounds(biorbd_model)
    X_bounds.concatenate(
        Bounds([activation_min] * biorbd_model.nbMuscles(), [activation_max] * biorbd_model.nbMuscles())
    )

    # Initial guess
    init_x = np.zeros((biorbd_model.nbQ() + biorbd_model.nbQdot() + biorbd_model.nbMuscleTotal(), nb_shooting + 1))
    for i in range(nb_shooting + 1):
        init_x[[0, 1, 5, 8, 9, 10], i] = q_ref[:, i]
        init_x[-biorbd_model.nbMuscleTotal() :, i] = excitation_ref[:, i]
    X_init = InitialConditions(init_x, interpolation_type=InterpolationType.EACH_FRAME)

    # Define control path constraint
    U_bounds = Bounds(
        [torque_min] * biorbd_model.nbGeneralizedTorque() + [activation_min] * biorbd_model.nbMuscleTotal(),
        [torque_max] * biorbd_model.nbGeneralizedTorque() + [activation_max] * biorbd_model.nbMuscleTotal(),
    )
    # Initial guess
    init_u = np.zeros((biorbd_model.nbGeneralizedTorque() + biorbd_model.nbMuscleTotal(), nb_shooting))
    for i in range(nb_shooting):
        init_u[1, i] = -500
        init_u[-biorbd_model.nbMuscleTotal() :, i] = excitation_ref[:, i]
    U_init = InitialConditions(init_u, interpolation_type=InterpolationType.EACH_FRAME)

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        variable_type,
        nb_shooting,
        final_time,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        objective_functions,
        constraints,
        nb_threads=nb_threads,
    )


if __name__ == "__main__":
    # Define the problem
    biorbd_model = biorbd.Model("../../ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact_deGroote_3d.bioMod")
    model_q = biorbd.Model("../../ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact_deGroote.bioMod")
    n_shooting_points = 25
    Gaitphase = "stance"

    # Generate data from file
    Data_to_track = Data_to_track(name_subject="equincocont01")
    [T, T_stance, T_swing] = Data_to_track.GetTime()
    final_time = T_stance

    grf_ref = Data_to_track.load_data_GRF(biorbd_model, T_stance, n_shooting_points)  # get ground reaction forces
    markers_ref = Data_to_track.load_data_markers(
        biorbd_model, T_stance, n_shooting_points, "stance"
    )  # get markers position
    q_ref = Data_to_track.load_data_q(biorbd_model, T_stance, n_shooting_points, "stance")  # get q from kalman
    emg_ref = Data_to_track.load_data_emg(biorbd_model, T_stance, n_shooting_points, "stance")  # get emg
    excitation_ref = Data_to_track.load_muscularExcitation(emg_ref)

    # Track these data
    ocp = prepare_ocp(
        biorbd_model,
        final_time,
        n_shooting_points,
        markers_ref,
        excitation_ref=excitation_ref,
        q_ref=q_ref,
        grf_ref=grf_ref,
        nb_threads=4,
    )
    ocp.add_plot("q", lambda x, u: q_ref, PlotType.STEP, axes_idx=[0, 1, 5, 8, 9, 10])

    # --- Solve the program --- #
    tic = time()
    sol = ocp.solve(
        solver="ipopt",
        options_ipopt={
            "ipopt.tol": 1e-3,
            "ipopt.max_iter": 5000,
            "ipopt.hessian_approximation": "exact",
            "ipopt.limited_memory_max_history": 50,
            "ipopt.linear_solver": "ma57",
        },
        show_online_optim=True,
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

    n_q = ocp.nlp[0]["model"].nbQ()
    n_qdot = ocp.nlp[0]["model"].nbQdot()
    nb_marker = biorbd_model.nbMarkers()
    n_mus = ocp.nlp[0]["model"].nbMuscleTotal()
    n_frames = q.shape[1]

    # --- Get markers position from q_sol and q_ref --- #
    markers_sol = np.ndarray((3, nb_marker, ocp.nlp[0]["ns"] + 1))
    markers_from_q_ref = np.ndarray((3, nb_marker, ocp.nlp[0]["ns"] + 1))

    markers_func_3d = []
    symbolic_states = MX.sym("x", ocp.nlp[0]["nx"], 1)
    symbolic_controls = MX.sym("u", ocp.nlp[0]["nu"], 1)
    for i in range(nb_marker):
        markers_func_3d.append(
            Function(
                "ForwardKin",
                [symbolic_states],
                [biorbd_model.marker(symbolic_states[:n_q], i).to_mx()],
                ["q"],
                ["marker_" + str(i)],
            ).expand()
        )

    markers_func_2d = []
    symbolic_states = MX.sym("x", model_q.nbQ() + model_q.nbQdot() + model_q.nbMuscleTotal(), 1)
    symbolic_controls = MX.sym("u", model_q.nbGeneralizedTorque() + model_q.nbMuscleTotal(), 1)
    for i in range(nb_marker):
        markers_func_2d.append(
            Function(
                "ForwardKin",
                [symbolic_states],
                [model_q.marker(symbolic_states[: model_q.nbQ()], i).to_mx()],
                ["q"],
                ["marker_" + str(i)],
            ).expand()
        )

    for i in range(ocp.nlp[0]["ns"] + 1):
        for j, mark_func in enumerate(markers_func_3d):
            markers_sol[:, j, i] = np.array(mark_func(vertcat(q[:, i], q_dot[:, i], activations[:, i]))).squeeze()
            Q_ref = np.concatenate([q_ref[:, i], np.zeros(model_q.nbQ())])
            markers_from_q_ref[:, j, i] = np.array(markers_func_2d[j](Q_ref)).squeeze()

    diff_track = np.sqrt((markers_sol - markers_ref) * (markers_sol - markers_ref)) * 1e3
    diff_sol = np.sqrt((markers_sol - markers_from_q_ref) * (markers_sol - markers_from_q_ref)) * 1e3
    hist_diff_track = np.zeros((3, nb_marker))
    hist_diff_sol = np.zeros((3, nb_marker))

    for n_mark in range(nb_marker):
        hist_diff_track[0, n_mark] = sum(diff_track[0, n_mark, :]) / n_shooting_points
        hist_diff_track[1, n_mark] = sum(diff_track[1, n_mark, :]) / n_shooting_points
        hist_diff_track[2, n_mark] = sum(diff_track[2, n_mark, :]) / n_shooting_points

        hist_diff_sol[0, n_mark] = sum(diff_sol[0, n_mark, :]) / n_shooting_points
        hist_diff_sol[1, n_mark] = sum(diff_sol[1, n_mark, :]) / n_shooting_points
        hist_diff_sol[2, n_mark] = sum(diff_sol[2, n_mark, :]) / n_shooting_points

    mean_diff_track = [
        sum(hist_diff_track[0, :]) / nb_marker,
        sum(hist_diff_track[1, :]) / nb_marker,
        sum(hist_diff_track[2, :]) / nb_marker,
    ]
    mean_diff_sol = [
        sum(hist_diff_sol[0, :]) / nb_marker,
        sum(hist_diff_sol[1, :]) / nb_marker,
        sum(hist_diff_sol[2, :]) / nb_marker,
    ]

    # --- Plot markers --- #
    label_markers = []
    for mark in range(nb_marker):
        label_markers.append(ocp.nlp[0]["model"].markerNames()[mark].to_string())

    figure, axes = plt.subplots(2, 3)
    axes = axes.flatten()
    title_markers = ["x axis", "y axis", "z axis"]
    for i in range(3):
        axes[i].bar(
            np.linspace(0, nb_marker, nb_marker),
            hist_diff_track[i, :],
            width=1.0,
            facecolor="b",
            edgecolor="k",
            alpha=0.5,
        )
        axes[i].set_xticks(np.arange(nb_marker))
        axes[i].set_xticklabels(label_markers, rotation=90)
        axes[i].set_ylabel("Mean differences in " + title_markers[i] + " (mm)")
        axes[i].plot([0, nb_marker], [mean_diff_track[i], mean_diff_track[i]], "--r")
        axes[i].set_title("markers differences between sol and exp")

        axes[i + 3].bar(
            np.linspace(0, nb_marker, nb_marker),
            hist_diff_sol[i, :],
            width=1.0,
            facecolor="b",
            edgecolor="k",
            alpha=0.5,
        )
        axes[i + 3].set_xticks(np.arange(nb_marker))
        axes[i + 3].set_xticklabels(label_markers, rotation=90)
        axes[i + 3].set_ylabel("Mean differences in " + title_markers[i] + " (mm)")
        axes[i + 3].plot([0, nb_marker], [mean_diff_sol[i], mean_diff_sol[i]], "--r")
        axes[i + 3].set_title("markers differences between sol and ref")
    plt.show()

    figure, axes = plt.subplots(2, 3)
    axes = axes.flatten()
    for i in range(3):
        axes[i].plot(t, diff_track[i, :, :].T)
        axes[i].set_xlabel("time (s)")
        axes[i].set_ylabel("Mean differences in " + title_markers[i] + " (mm)")
        axes[i].set_title("markers differences between sol and exp")

        axes[i + 3].plot(t, diff_sol[i, :, :].T)
        axes[i + 3].set_xlabel("time (s)")
        axes[i + 3].set_ylabel("Meaen differences in " + title_markers[i] + " (mm)")
        axes[i + 3].set_title("markers differences between sol and ref")
    plt.show()

    # --- Save the optimal control program and the solution --- #
    ocp.save(sol, "marche_stance_excitation")

    # --- Load the optimal control program and the solution --- #
    # ocp_load, sol_load = OptimalControlProgram.load("/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/stance/RES/equincocont01/excitations/model_init/marche_stance_excitation.bo")
    # result = ShowResult(ocp_load, sol_load)

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
    result.graphs()
