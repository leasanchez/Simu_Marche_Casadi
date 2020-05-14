import numpy as np
from casadi import dot, Function, vertcat, MX
from matplotlib import pyplot as plt
import sys

sys.path.append('/home/leasanchez/programmation/BiorbdOptim')
import biorbd

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
    OdeSolver,
    Dynamics,
    Data,
)

def get_last_contact_forces(ocp, nlp, t, x, u, data_to_track=()):
    force = nlp["contact_forces_func"](x[-1], u[-1])
    val = force - data_to_track[t[-1], :]
    return dot(val, val)

def prepare_ocp(
    biorbd_model,
    final_time,
    nb_shooting,
    markers_ref,
    activation_ref,
    grf_ref,
    show_online_optim,
):
    # Problem parameters
    torque_min, torque_max, torque_init = -1000, 1000, 0
    activation_min, activation_max, activation_init = 0, 1, 0.1

    # Add objective functions
    objective_functions = (
        {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 100, "controls_idx": [3, 4, 5]},
        {"type": Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, "weight": 0.1, "data_to_track":activation_ref.T},
        {"type": Objective.Lagrange.TRACK_MARKERS, "weight": 100, "data_to_track": markers_ref},
        {"type": Objective.Lagrange.TRACK_CONTACT_FORCES, "weight": 0.05, "data_to_track": grf_ref.T},
        {"type": Objective.Mayer.CUSTOM, "weight": 0.05, "function": get_last_contact_forces, "data_to_track": grf_ref.T, "instant": Instant.ALL}
    )

    # Dynamics
    variable_type = ProblemType.muscles_and_torque_driven_with_contact

    # Constraints
    constraints = ()

    # Path constraint
    X_bounds = QAndQDotBounds(biorbd_model)

    # Initial guess
    X_init = InitialConditions([0] * (biorbd_model.nbQ() + biorbd_model.nbQdot()))

    # Define control path constraint
    U_bounds = Bounds(
        [torque_min] * biorbd_model.nbGeneralizedTorque() + [activation_min] * biorbd_model.nbMuscleTotal(),
        [torque_max] * biorbd_model.nbGeneralizedTorque() + [activation_max] * biorbd_model.nbMuscleTotal(),
    )
    U_init = InitialConditions(
        [torque_init] * biorbd_model.nbGeneralizedTorque() + [activation_init] * biorbd_model.nbMuscleTotal()
    )

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        variable_type,
        nb_shooting,
        final_time,
        objective_functions,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        constraints,
        show_online_optim=show_online_optim,
    )


if __name__ == "__main__":
    # Define the problem
    biorbd_model = biorbd.Model("../../ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact.bioMod")
    n_shooting_points = 50
    Gaitphase = 'stance'

    # Generate data from file
    from Marche_BiorbdOptim.LoadData import load_data_markers, load_data_q, load_data_emg, load_data_GRF

    name_subject = "equincocont01"
    grf_ref, T, T_stance, T_swing = load_data_GRF(name_subject, biorbd_model, n_shooting_points)
    final_time = T_stance

    t, markers_ref = load_data_markers(name_subject, biorbd_model, final_time, n_shooting_points, Gaitphase)
    q_ref = load_data_q(name_subject, biorbd_model, final_time, n_shooting_points, Gaitphase)
    emg_ref = load_data_emg(name_subject, biorbd_model, final_time, n_shooting_points, Gaitphase)
    activation_ref = np.zeros((biorbd_model.nbMuscleTotal(), n_shooting_points))
    idx_emg = 0
    for i in range(biorbd_model.nbMuscleTotal()):
        if (i!=1) and (i!=2) and (i!=3) and (i!=5) and (i!=6) and (i!=11) and (i!=12):
            activation_ref[i, :] = emg_ref[idx_emg, :-1]
            idx_emg += 1

    # Track these data
    biorbd_model = biorbd.Model("../../ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact.bioMod")
    ocp = prepare_ocp(
        biorbd_model,
        final_time,
        n_shooting_points,
        markers_ref,
        activation_ref,
        grf_ref=grf_ref[1:, :],
        show_online_optim=True,
    )

    # --- Solve the program --- #
    sol = ocp.solve()

    # --- Get Results --- #
    states, controls = Data.get_data(ocp, sol["x"])
    q = states["q"].to_matrix()
    q_dot = states["q_dot"].to_matrix()
    tau = controls["tau"].to_matrix()
    mus = controls["muscles"].to_matrix()

    n_q = ocp.nlp[0]["model"].nbQ()
    n_qdot = ocp.nlp[0]["model"].nbQdot()
    n_mark = ocp.nlp[0]["model"].nbMarkers()
    n_mus = ocp.nlp[0]["model"].nbMuscleTotal()
    n_frames = q.shape[1]

    # --- Compute ground reaction forces --- #
    x = vertcat(q, q_dot)
    u = vertcat(tau, mus)
    contact_forces = ocp.nlp[0]["contact_forces_func"](x, u)

    # --- Get markers position from q_sol and q_ref --- #
    markers_sol = np.ndarray((3, n_mark, ocp.nlp[0]["ns"] + 1))
    markers_from_q_ref = np.ndarray((3, n_mark, ocp.nlp[0]["ns"] + 1))

    markers_func = []
    symbolic_states = MX.sym("x", ocp.nlp[0]["nx"], 1)
    symbolic_controls = MX.sym("u", ocp.nlp[0]["nu"], 1)
    for i in range(n_mark):
        markers_func.append(
            Function(
                "ForwardKin",
                [symbolic_states],
                [biorbd_model.marker(symbolic_states[:n_q], i).to_mx()],
                ["q"],
                ["marker_" + str(i)],
            ).expand()
        )
    for i in range(ocp.nlp[0]['ns']):
        for j, mark_func in enumerate(markers_func):
            markers_sol[:, j, i] = np.array(mark_func(vertcat(q[:, i], q_dot[:, i]))).squeeze()
            Q_ref = np.concatenate([q_ref[:, i], np.zeros(n_q)])
            markers_from_q_ref[:, j, i] = np.array(mark_func(Q_ref)).squeeze()

    diff_track = (markers_sol - markers_ref) * (markers_sol - markers_ref)
    diff_sol = (markers_sol - markers_from_q_ref) * (markers_sol - markers_from_q_ref)
    hist_diff_track = np.zeros((3, n_mark))
    hist_diff_sol = np.zeros((3, n_mark))

    for n_mark in range(n_mark):
        hist_diff_track[0, n_mark] = sum(diff_track[0, n_mark, :]) / n_mark
        hist_diff_track[1, n_mark] = sum(diff_track[1, n_mark, :]) / n_mark
        hist_diff_track[2, n_mark] = sum(diff_track[2, n_mark, :]) / n_mark

        hist_diff_sol[0, n_mark] = sum(diff_sol[0, n_mark, :]) / n_mark
        hist_diff_sol[1, n_mark] = sum(diff_sol[1, n_mark, :]) / n_mark
        hist_diff_sol[2, n_mark] = sum(diff_sol[2, n_mark, :]) / n_mark

    mean_diff_track = [sum(hist_diff_track[0, :]) / n_mark,
                       sum(hist_diff_track[1, :]) / n_mark,
                       sum(hist_diff_track[2, :]) / n_mark]
    mean_diff_sol = [sum(hist_diff_sol[0, :]) / n_mark,
                     sum(hist_diff_sol[1, :]) / n_mark,
                     sum(hist_diff_sol[2, :]) / n_mark]

    # --- Plot --- #
    def plot_control(ax, t, x, color='b'):
        nbPoints = len(np.array(x))
        for n in range(nbPoints - 1):
            ax.plot([t[n], t[n + 1], t[n + 1]], [x[n], x[n], x[n + 1]], color)

    figure, axes = plt.subplots(2,3)
    axes = axes.flatten()
    for i in range(biorbd_model.nbQ()):
        name_dof = ocp.nlp[0]["model"].nameDof()[i].to_string()
        axes[i].set_title(name_dof)
        axes[i].plot(t, q[i, :])
        axes[i].plot(t, q_ref[i, :], 'r')

    figure2, axes2 = plt.subplots(4, 5, sharex=True)
    axes2 = axes2.flatten()
    for i in range(biorbd_model.nbMuscleTotal()):
        name_mus = ocp.nlp[0]["model"].muscleNames()[i].to_string()
        plot_control(axes2[i], t[:-1], activation_ref[i, :], color='r')
        plot_control(axes2[i], t[:-1], mus[i, :-1])
        axes2[i].set_title(name_mus)

    plt.figure('Contact forces')
    plt.plot(t, contact_forces.T, 'b')
    plt.plot(t, grf_ref[1:, :].T, 'r')

    # markers
    label_markers = []
    title_markers = ['x', 'y', 'z']
    for mark in range(nb_markers):
        label_markers.append(ocp.nlp[0]["model"].markerNames()[mark].to_string())

    figure, axes = plt.subplots(3, 3)
    axes = axes.flatten()
    title_markers = ['x', 'y', 'z']
    for i in range(3):
        axes[i].bar(np.linspace(0,nb_markers, nb_markers), hist_diff_ref[i, :], width=1.0, facecolor='b', edgecolor='k', alpha=0.5)
        axes[i].set_xticks(np.arange(nb_markers))
        axes[i].set_xticklabels(label_markers, rotation=90)
        axes[i].set_title(title_markers[i])
        axes[i].plot([0, nb_markers], [mean_diff_ref[i], mean_diff_ref[i]], '--r')

        axes[i + 3].bar(np.linspace(0,nb_markers, nb_markers), hist_diff_track[i, :], width=1.0, facecolor='b', edgecolor='k', alpha=0.5)
        axes[i + 3].set_xticks(np.arange(nb_markers))
        axes[i + 3].set_xticklabels(label_markers, rotation=90)
        axes[i + 3].set_title(title_markers[i])
        axes[i + 3].plot([0, nb_markers], [mean_diff_track[i], mean_diff_track[i]], '--r')

        axes[i + 6].bar(np.linspace(0,nb_markers, nb_markers), hist_diff_sol[i, :], width=1.0, facecolor='b', edgecolor='k', alpha=0.5)
        axes[i + 6].set_xticks(np.arange(nb_markers))
        axes[i + 6].set_xticklabels(label_markers, rotation=90)
        axes[i + 6].set_title(title_markers[i])
        axes[i + 6].plot([0, nb_markers], [mean_diff_sol[i], mean_diff_sol[i]], '--r')

        if (i==1):
            axes[i].set_title('markers differences between kalman and exp')
            axes[i + 3].set_title('markers differences between sol and exp')
            axes[i + 6].set_title('markers differences between sol and ref')
    plt.show()

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.graphs()
