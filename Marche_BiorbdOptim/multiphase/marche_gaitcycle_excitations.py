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
    Data,
    Dynamics,
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
    nb_phases = len(biorbd_model)
    torque_min, torque_max, torque_init = -1000, 1000, 0
    activation_min, activation_max, activation_init = 0, 1, 0.1

    # Add objective functions
    objective_functions = ((
        {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 100, "controls_idx":[3, 4, 5]},
        {"type": Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, "weight": 1, "data_to_track":activation_ref[0].T},
        {"type": Objective.Lagrange.TRACK_MARKERS, "weight": 100, "data_to_track": markers_ref[0]},
        {"type": Objective.Lagrange.TRACK_CONTACT_FORCES, "weight": 0.05, "data_to_track": grf_ref[:, :-1].T},
        {"type": Objective.Mayer.CUSTOM, "function": get_last_contact_forces, "data":grf_ref.T, "weight": 0.05, "instant": Instant.ALL}
    ),
    (
        {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 1, "controls_idx":[3, 4, 5]},
        {"type": Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, "weight": 1, "data_to_track":activation_ref[1].T},
        {"type": Objective.Lagrange.TRACK_MARKERS, "weight": 100, "data_to_track": markers_ref[1]}
    ))

    # Dynamics
    problem_type = (
        ProblemType.muscle_excitations_and_torque_driven_with_contact,
        ProblemType.muscle_excitations_and_torque_driven,
    )

    # Constraints
    constraints = ()

    # Path constraint
    X_bounds = [QAndQDotBounds(biorbd_model[i])for i in range(nb_phases)]

    # Initial guess
    X_init = [InitialConditions([0] * (biorbd_model[i].nbQ() + biorbd_model[i].nbQdot()))for i in range(nb_phases)]

    # Define control path constraint
    U_bounds = [
        Bounds(
        min_bound = [torque_min] * biorbd_model[i].nbGeneralizedTorque() + [activation_min] * biorbd_model[i].nbMuscleTotal(),
        max_bound = [torque_max] * biorbd_model[i].nbGeneralizedTorque() + [activation_max] * biorbd_model[i].nbMuscleTotal(),
    )
        for i in range(nb_phases)]

    U_init = [InitialConditions(
        [torque_init] * biorbd_model[i].nbGeneralizedTorque() + [activation_init] * biorbd_model[i].nbMuscleTotal()
    ) for i in range(nb_phases)]

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        problem_type,
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
    # Model path
    biorbd_model = (
        biorbd.Model("../../ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact.bioMod"),
        biorbd.Model("../../ModelesS2M/ANsWER_Rleg_6dof_17muscle_0contact.bioMod"),
    )

    # Problem parameters
    number_shooting_points = [25, 25]

    # Generate data from file
    from Marche_BiorbdOptim.LoadData import load_data_markers, load_data_q, load_data_emg, load_data_GRF

    name_subject = "equincocont01"
    grf_ref, T, T_stance, T_swing = load_data_GRF(name_subject, biorbd_model, number_shooting_points[0])
    phase_time = [T_stance, T_swing]

    # phase stance
    t_stance, markers_ref_stance = load_data_markers(name_subject, biorbd_model[0], phase_time[0], number_shooting_points[0], 'stance')
    q_ref_stance = load_data_q(name_subject, biorbd_model[0], phase_time[0], number_shooting_points[0], 'stance')
    emg_ref_stance = load_data_emg(name_subject, biorbd_model[0], phase_time[0], number_shooting_points[0], 'stance')
    excitation_ref_stance = np.zeros((biorbd_model[0].nbMuscleTotal(), number_shooting_points[0] + 1))
    idx_emg = 0
    for i in range(biorbd_model[0].nbMuscleTotal()):
        if (i!=1) and (i!=2) and (i!=3) and (i!=5) and (i!=6) and (i!=11) and (i!=12):
            excitation_ref_stance[i, :] = emg_ref_stance[idx_emg, :]
            idx_emg += 1

    # phase swing
    t_swing, markers_ref_swing = load_data_markers(name_subject, biorbd_model[1], phase_time[1], number_shooting_points[1], 'swing')
    q_ref_swing = load_data_q(name_subject, biorbd_model[1], phase_time[1], number_shooting_points[1], 'swing')
    emg_ref_swing = load_data_emg(name_subject, biorbd_model[1], phase_time[1], number_shooting_points[1], 'swing')
    excitation_ref_swing = np.zeros((biorbd_model[1].nbMuscleTotal(), number_shooting_points[1] + 1))
    idx_emg = 0
    for i in range(biorbd_model[0].nbMuscleTotal()):
        if (i!=1) and (i!=2) and (i!=3) and (i!=5) and (i!=6) and (i!=11) and (i!=12):
            excitation_ref_swing[i, :] = emg_ref_swing[idx_emg, :]
            idx_emg += 1

    # Track these data
    biorbd_model = (
        biorbd.Model("../../ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact.bioMod"),
        biorbd.Model("../../ModelesS2M/ANsWER_Rleg_6dof_17muscle_0contact.bioMod"),
    )
    ocp = prepare_ocp(
        biorbd_model,
        phase_time,
        number_shooting_points,
        markers_ref = [markers_ref_stance, markers_ref_swing],
        excitation_ref = [excitation_ref_stance[:, :-1], excitation_ref_swing[:, :-1]],
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

    # --- Compute ground reaction forces --- #
    contact_forces = np.zeros((2, sum([nlp["ns"] for nlp in ocp.nlp]) + 1))
    grf = np.zeros((2, sum([nlp["ns"] for nlp in ocp.nlp]) + 1))

    q_contact = states["q"].to_matrix(phase_idx=0)
    q_dot_contact = states["q_dot"].to_matrix(phase_idx=0)
    tau_contact = controls["tau"].to_matrix(phase_idx=0)
    mus_contact = controls["muscles"].to_matrix(phase_idx=0)

    x = vertcat(q_contact, q_dot_contact)
    u = vertcat(tau_contact, mus_contact)
    contact_forces[:, : ocp.nlp[0]["ns"] + 1] = ocp.nlp[0]["contact_forces_func"](x, u)
    grf[:, : ocp.nlp[0]["ns"] + 1] = grf_ref[1:, :]

    # --- Get References --- #
    q_ref = np.zeros((biorbd_model[0].nbQ(), sum([nlp["ns"] for nlp in ocp.nlp]) + 1))
    activation_ref = np.zeros((biorbd_model[0].nbMuscleTotal(), sum([nlp["ns"] for nlp in ocp.nlp]) + 1))
    q_ref[:, : ocp.nlp[0]["ns"] + 1] = q_ref_stance
    activation_ref[:, : ocp.nlp[0]["ns"] + 1] = activation_ref_stance
    q_ref[:, ocp.nlp[0]["ns"]: ocp.nlp[0]["ns"] + ocp.nlp[1]["ns"] + 1] = q_ref_swing
    activation_ref[:, ocp.nlp[0]["ns"]: ocp.nlp[0]["ns"] + ocp.nlp[1]["ns"] + 1] = activation_ref_swing

    # --- Get markers position from q_sol and q_ref --- #
    nb_markers = biorbd_model[0].nbMarkers()
    nb_q = biorbd_model[0].nbQ()

    markers_sol = np.ndarray((3, nb_markers, ocp.nlp[0]["ns"] + 1))
    markers_from_q_ref = np.ndarray((3, nb_markers, ocp.nlp[0]["ns"] + 1))
    markers_ref = np.ndarray((3, nb_markers, ocp.nlp[0]["ns"] + 1))
    markers_ref[:, : ocp.nlp[0]["ns"] + 1] = markers_ref_stance
    markers_ref[:, ocp.nlp[0]["ns"]: ocp.nlp[0]["ns"] + ocp.nlp[1]["ns"] + 1] = markers_ref_swing

    markers_func = []
    symbolic_states = MX.sym("x", ocp.nlp["nx"], 1)
    symbolic_controls = MX.sym("u", ocp.nlp["nu"], 1)
    for i in range(nb_markers):
        markers_func.append(
            Function(
                "ForwardKin",
                [symbolic_states],
                [biorbd_model[0].marker(symbolic_states[:nb_q], i).to_mx()],
                ["q"],
                ["marker_" + str(i)],
            ).expand()
        )
    for i in range(ocp.nlp[0]['ns']):
        for j, mark_func in enumerate(markers_func):
            markers_sol[:, j, i] = np.array(mark_func(vertcat(q[:, i], q_dot[:, i]))).squeeze()
            Q_ref = np.concatenate([q_ref[:, i], np.zeros(nb_q)])
            markers_from_q_ref[:, j, i] = np.array(mark_func(Q_ref)).squeeze()

    diff_track = (markers_sol - markers_ref) * (markers_sol - markers_ref)
    diff_sol = (markers_sol - markers_from_q_ref) * (markers_sol - markers_from_q_ref)
    hist_diff_track = np.zeros((3, nb_markers))
    hist_diff_sol = np.zeros((3, nb_markers))

    for n_mark in range(nb_markers):
        hist_diff_track[0, n_mark] = sum(diff_track[0, n_mark, :]) / nb_markers
        hist_diff_track[1, n_mark] = sum(diff_track[1, n_mark, :]) / nb_markers
        hist_diff_track[2, n_mark] = sum(diff_track[2, n_mark, :]) / nb_markers

        hist_diff_sol[0, n_mark] = sum(diff_sol[0, n_mark, :]) / nb_markers
        hist_diff_sol[1, n_mark] = sum(diff_sol[1, n_mark, :]) / nb_markers
        hist_diff_sol[2, n_mark] = sum(diff_sol[2, n_mark, :]) / nb_markers

    mean_diff_track = [sum(hist_diff_track[0, :]) / nb_markers,
                       sum(hist_diff_track[1, :]) / nb_markers,
                       sum(hist_diff_track[2, :]) / nb_markers]
    mean_diff_sol = [sum(hist_diff_sol[0, :]) / nb_markers,
                     sum(hist_diff_sol[1, :]) / nb_markers,
                     sum(hist_diff_sol[2, :]) / nb_markers]

    # --- Plot --- #
    t = np.hstack([t_stance[:-1], T_stance + t_swing])

    def plot_control(ax, t, x, color='b'):
        nbPoints = len(np.array(x))
        for n in range(nbPoints - 1):
            ax.plot([t[n], t[n + 1], t[n + 1]], [x[n], x[n], x[n + 1]], color)

    figure, axes = plt.subplots(2,3)
    axes = axes.flatten()
    for i in range(biorbd_model[0].nbQ()):
        name_dof = ocp.nlp[0]["model"].nameDof()[i].to_string()
        axes[i].set_title(name_dof)
        axes[i].plot(t, q[i, :])
        axes[i].plot(t, q_ref[i, :], 'r')
        axes[i].plot([T_stance, T_stance], [np.min(q[i, :]), np.max(q[i, :])], 'k--')

    figure2, axes2 = plt.subplots(4, 5, sharex=True)
    axes2 = axes2.flatten()
    for i in range(biorbd_model[0].nbMuscleTotal()):
        name_mus = ocp.nlp[0]["model"].muscleNames()[i].to_string()
        plot_control(axes2[i], t, activation_ref[i, :], color='r')
        plot_control(axes2[i], t, mus[i, :])
        axes2[i].set_title(name_mus)
        axes2[i].plot([T_stance, T_stance], [0, 1], 'k--')

    plt.figure('Contact forces')
    plt.plot(t, contact_forces.T, 'b')
    plt.plot(t, grf.T, 'r')
    plt.plot([T_stance, T_stance], [np.min(grf[1, :]), np.max(grf[1, :])], 'k--')

    plt.show()

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
    # result.graphs()
