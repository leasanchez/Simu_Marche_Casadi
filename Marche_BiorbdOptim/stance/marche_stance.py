from scipy.integrate import solve_ivp
import numpy as np
from casadi import dot, Function, vertcat
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
    CS_func = Function(
        "Contact_force",
        [ocp.symbolic_states, ocp.symbolic_controls],
        [Dynamics.forces_from_forward_dynamics_torque_muscle_driven_with_contact(
                ocp.symbolic_states, ocp.symbolic_controls, nlp
            )],
        ["x", "u"],
        ["CS"],
    ).expand()
    force = CS_func(x[-1], u[-1])
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
        {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 1, "controls_idx":[3, 4, 5]},
        {"type": Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, "weight": 0.1, "data_to_track":activation_ref.T},
        {"type": Objective.Lagrange.TRACK_MARKERS, "weight": 100, "data_to_track": markers_ref},
        {"type": Objective.Lagrange.TRACK_CONTACT_FORCES, "weight": 0.05, "data_to_track": grf_ref.T},
        {"type": Objective.Lagrange.CUSTOM, "weight": 0.05, "function": get_last_contact_forces, "data_to_track": grf_ref.T, "instant": Instant.ALL}
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
    biorbd_model = biorbd.Model("ANsWER_Rleg_6dof_17muscle_1contact.bioMod")
    n_shooting_points = 25
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
    biorbd_model = biorbd.Model("ANsWER_Rleg_6dof_17muscle_1contact.bioMod")
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
    states, controls = Data.get_data_from_V(ocp, sol["x"])
    q = states["q"].to_matrix()
    q_dot = states["q_dot"].to_matrix()
    tau = controls["tau"].to_matrix()
    mus = controls["muscles"].to_matrix()

    # --- Compute ground reaction forces --- #
    CS_func = Function(
        "Contact_force",
        [ocp.symbolic_states, ocp.symbolic_controls],
        [ocp.nlp[0]["model"].getConstraints().getForce().to_mx()],
        ["x", "u"],
        ["CS"],
    ).expand()

    x = vertcat(q, q_dot)
    u = vertcat(tau, mus)
    contact_forces = CS_func(x, u)

    # --- Get markers position from q_sol and q_ref --- #
    nb_markers = biorbd_model.nbMarkers()
    nb_q = biorbd_model.nbQ()

    markers_sol = np.ndarray((3, nb_markers, ocp.nlp[0]["ns"] + 1))
    markers_from_q_ref = np.ndarray((3, nb_markers, ocp.nlp[0]["ns"] + 1))

    markers_func = []
    for i in range(nb_markers):
        markers_func.append(
            Function(
                "ForwardKin",
                [ocp.symbolic_states],
                [biorbd_model.marker(ocp.symbolic_states[:nb_q], i).to_mx()],
                ["q"],
                ["marker_" + str(i)],
            ).expand()
        )
    for i in range(ocp.nlp[0]['ns']):
        for j, mark_func in enumerate(markers_func):
            markers_sol[:, j, i] = np.array(mark_func(vertcat(q[:, i], q_dot[:, i]))).squeeze()
            Q_ref = np.concatenate([q_ref[:, i], np.zeros(nb_q)])
            markers_from_q_ref[:, j, i] = np.array(mark_func(Q_ref)).squeeze()

    diff_ref = (markers_from_q_ref - markers_ref) * (markers_from_q_ref - markers_ref)
    diff_track = (markers_sol - markers_ref) * (markers_sol - markers_ref)
    hist_diff_ref = np.zeros((3,nb_markers))
    hist_diff_track = np.zeros((3, nb_markers))

    for n_mark in range(nb_markers):
        hist_diff_ref[0, n_mark] = sum(diff_ref[0, n_mark, :])/nb_markers
        hist_diff_ref[1, n_mark] = sum(diff_ref[1, n_mark, :])/nb_markers
        hist_diff_ref[2, n_mark] = sum(diff_ref[2, n_mark, :])/nb_markers

        hist_diff_track[0, n_mark] = sum(diff_track[0, n_mark, :])/nb_markers
        hist_diff_track[1, n_mark] = sum(diff_track[1, n_mark, :])/nb_markers
        hist_diff_track[2, n_mark] = sum(diff_track[2, n_mark, :])/nb_markers

    mean_diff_ref = [sum(hist_diff_ref[0, :])/nb_markers,
                     sum(hist_diff_ref[1, :])/nb_markers,
                     sum(hist_diff_ref[2, :])/nb_markers]
    mean_diff_track = [sum(hist_diff_track[0, :]) / nb_markers,
                       sum(hist_diff_track[1, :]) / nb_markers,
                       sum(hist_diff_track[2, :]) / nb_markers]

    # --- Plot --- #
    def plot_control(ax, t, x, color='b'):
        nbPoints = len(np.array(x))
        for n in range(nbPoints - 1):
            ax.plot([t[n], t[n + 1], t[n + 1]], [x[n], x[n], x[n + 1]], color)
    axes = axes.flatten()
    for i in range(biorbd_model.nbQ()):
        axes[i].plot(t, q[i, :])
        axes[i].plot(t, q_ref[i, :], 'r')
        axes[i + biorbd_model.nbQ()].plot(t, qdot[i, :])
        axes[i + 2*biorbd_model.nbQ()].plot(t, tau[i, :])

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


    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate(show_meshes=False)
    result.graphs()
