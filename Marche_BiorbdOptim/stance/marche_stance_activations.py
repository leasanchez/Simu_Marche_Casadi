import numpy as np
from casadi import dot, Function, vertcat, MX, tanh
from matplotlib import pyplot as plt
import sys
from scipy.integrate import solve_ivp

sys.path.append('/home/leasanchez/programmation/BiorbdOptim')
import biorbd

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
)

def get_last_contact_forces(ocp, nlp, t, x, u, data_to_track=()):
    force = nlp["contact_forces_func"](x[-1], u[-1])
    val = force - data_to_track[t[-1], :]
    return dot(val, val)

def generate_activation(biorbd_model, final_time, nb_shooting, emg_ref):
    # Aliases
    nb_mus = biorbd_model.nbMuscleTotal()
    nb_musclegrp = biorbd_model.nbMuscleGroups()
    dt = final_time / nb_shooting

    # init
    ta = td = []
    activation_ref = np.ndarray((nb_mus, nb_shooting + 1))

    for n_grp in range(nb_musclegrp):
        for n_muscle in range(biorbd_model.muscleGroup(n_grp).nbMuscles()):
            ta.append(biorbd_model.muscleGroup(n_grp).muscle(n_muscle).characteristics().torqueActivation().to_mx())
            td.append(biorbd_model.muscleGroup(n_grp).muscle(n_muscle).characteristics().torqueDeactivation().to_mx())

    def compute_activationDot(a, e, ta, td):
        activationDot = []
        for i in range(nb_mus):
            f = 0.5 * tanh(0.1*(e[i] - a[i]))
            da = (f + 0.5) / (ta[i] * (0.5 + 1.5 * a[i]))
            dd = (-f + 0.5) * (0.5 + 1.5 * a[i]) / td[i]
            activationDot.append((da + dd) * (e[i] - a[i]))
        return vertcat(*activationDot)

    # casadi
    symbolic_states = MX.sym("a", nb_mus, 1)
    symbolic_controls = MX.sym("e", nb_mus, 1)
    dynamics_func = Function(
        "ActivationDyn",
        [symbolic_states, symbolic_controls],
        [compute_activationDot(symbolic_states, symbolic_controls, ta, td)],
        ["a", "e"],
        ["adot"],
    ).expand()

    def dyn_interface(t, a, e):
        return np.array(dynamics_func(a, e)).squeeze()

    # Integrate and collect the position of the markers accordingly
    activation_init = emg_ref[:, 0]
    activation_ref[:, 0] = activation_init
    sol_act = []
    for i in range(nb_shooting):
        e = emg_ref[:, i]
        sol = solve_ivp(dyn_interface, (0, dt), activation_init, method="RK45", args=(e,))
        sol_act.append(sol["y"])
        activation_init = sol["y"][:, -1]
        activation_ref[:, i + 1]=activation_init

    # t = np.linspace(0, final_time, nb_shooting + 1)
    #
    # def plot_control(ax, t, x, color='b'):
    #     nbPoints = len(np.array(x))
    #     for n in range(nbPoints - 1):
    #         ax.plot([t[n], t[n + 1], t[n + 1]], [x[n], x[n], x[n + 1]], color)
    #
    # figure2, axes2 = plt.subplots(4, 5, sharex=True)
    # axes2 = axes2.flatten()
    # for i in range(biorbd_model.nbMuscleTotal()):
    #     name_mus = biorbd_model.muscle(i).name().to_string()
    #     plot_control(axes2[i], t, emg_ref[i, :], color='r')
    #     axes2[i].set_title(name_mus)
    #     axes2[i].set_ylim([0, 1])
    #     axes2[i].set_yticks(np.arange(0, 1, step=1 / 5, ))
    #     axes2[i].grid(color="k", linestyle="--", linewidth=0.5)
    #     for j in range(nb_shooting):
    #         t2 = np.linspace(t[j], t[j+1], sol_act[j].shape[1])
    #         axes2[i].plot(t2, sol_act[j][i, :], 'b-')
    #         axes2[i].plot(t2[-1], sol_act[j][i, -1], 'b.')
    # axes2[-1].remove()
    # axes2[-2].remove()
    # axes2[-3].remove()

    return activation_ref

def prepare_ocp(
    biorbd_model,
    final_time,
    nb_shooting,
    markers_ref,
    activation_ref,
    grf_ref,
    q_ref,
    nb_threads,
):
    # Problem parameters
    torque_min, torque_max, torque_init = -1000, 1000, 0
    activation_min, activation_max, activation_init = 0, 1, 0.1

    # Add objective functions
    objective_functions = (
        {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 100, "controls_idx": [3, 4, 5]},
        {"type": Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, "weight": 5, "data_to_track":activation_ref.T},
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
    init_x = np.zeros((biorbd_model.nbQ() + biorbd_model.nbQdot(), nb_shooting + 1))
    for i in range(nb_shooting + 1):
        init_x[:biorbd_model.nbQ(), i] = q_ref[:, i]
    X_init = InitialConditions(init_x, interpolation_type=InterpolationType.EACH_FRAME)

    # Define control path constraint
    U_bounds = Bounds(
        [torque_min] * biorbd_model.nbGeneralizedTorque() + [activation_min] * biorbd_model.nbMuscleTotal(),
        [torque_max] * biorbd_model.nbGeneralizedTorque() + [activation_max] * biorbd_model.nbMuscleTotal(),
    )

    init_u = np.zeros((biorbd_model.nbGeneralizedTorque() + biorbd_model.nbMuscleTotal(), nb_shooting))
    for i in range(nb_shooting):
        init_u[:biorbd_model.nbQ(), i] = [0, -500, 0, 0, 0, 0]
        init_u[-biorbd_model.nbMuscleTotal():, i] = activation_ref[:, i]  #0.1
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
    biorbd_model = biorbd.Model("../../ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact.bioMod")
    n_shooting_points = 25
    Gaitphase = 'stance'

    # Generate data from file
    from Marche_BiorbdOptim.LoadData import load_data_markers, load_data_q, load_data_emg, load_data_GRF, load_muscularExcitation

    name_subject = "equincocont01"
    grf_ref, T, T_stance, T_swing = load_data_GRF(name_subject, biorbd_model, n_shooting_points)
    final_time = T_stance

    t, markers_ref = load_data_markers(name_subject, biorbd_model, final_time, n_shooting_points, Gaitphase)
    q_ref = load_data_q(name_subject, biorbd_model, final_time, n_shooting_points, Gaitphase)
    emg_ref = load_data_emg(name_subject, biorbd_model, final_time, n_shooting_points, Gaitphase)
    excitation_ref = load_muscularExcitation(emg_ref)
    activation_ref = generate_activation(biorbd_model=biorbd_model, final_time=final_time, nb_shooting=n_shooting_points, emg_ref=excitation_ref)
    
    # activation_ref = excitation_ref
    # def plot_control(ax, t, x, color='b'):
    #     nbPoints = len(np.array(x))
    #     for n in range(nbPoints - 1):
    #         ax.plot([t[n], t[n + 1], t[n + 1]], [x[n], x[n], x[n + 1]], color)
    #
    # figure2, axes2 = plt.subplots(4, 5, sharex=True)
    # axes2 = axes2.flatten()
    # for i in range(biorbd_model.nbMuscleTotal()):
    #     name_mus = biorbd_model.muscle(i).name().to_string()
    #     plot_control(axes2[i], t, activation_ref[i, :], color='b')
    #     plot_control(axes2[i], t, excitation_ref[i, :], color='r')
    #     axes2[i].set_title(name_mus)
    #     axes2[i].set_ylim([0, 1])
    #     axes2[i].set_yticks(np.arange(0, 1, step=1 / 5, ))
    #     axes2[i].grid(color="k", linestyle="--", linewidth=0.5)
    # axes2[-1].remove()
    # axes2[-2].remove()
    # axes2[-3].remove()

    # Track these data
    biorbd_model = biorbd.Model("../../ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact.bioMod")
    ocp = prepare_ocp(
        biorbd_model,
        final_time,
        n_shooting_points,
        markers_ref,
        activation_ref,
        grf_ref=grf_ref[1:, :],
        q_ref=q_ref,
        nb_threads=4
    )

    # --- Solve the program --- #
    sol = ocp.solve(
        solver="ipopt",
        options_ipopt={
            "ipopt.tol": 1e-4,
            "ipopt.max_iter": 5000,
            "ipopt.hessian_approximation": "exact",
            "ipopt.limited_memory_max_history": 50,
            "ipopt.linear_solver": "ma57", },
        show_online_optim=True,
    )

    # --- Get Results --- #
    states, controls = Data.get_data(ocp, sol["x"])
    q = states["q"]
    q_dot = states["q_dot"]
    tau = controls["tau"]
    mus = controls["muscles"]

    n_q = ocp.nlp[0]["model"].nbQ()
    n_qdot = ocp.nlp[0]["model"].nbQdot()
    nb_marker = ocp.nlp[0]["model"].nbMarkers()
    nb_mus = ocp.nlp[0]["model"].nbMuscleTotal()
    n_frames = q.shape[1]

    # --- Compute ground reaction forces --- #
    x = vertcat(q, q_dot)
    u = vertcat(tau, mus)
    contact_forces = ocp.nlp[0]["contact_forces_func"](x, u)

    # --- Get markers position from q_sol and q_ref --- #
    markers_sol = np.ndarray((3, nb_marker, ocp.nlp[0]["ns"] + 1))
    markers_from_q_ref = np.ndarray((3, nb_marker, ocp.nlp[0]["ns"] + 1))

    markers_func = []
    symbolic_states = MX.sym("x", ocp.nlp[0]["nx"], 1)
    symbolic_controls = MX.sym("u", ocp.nlp[0]["nu"], 1)
    for i in range(nb_marker):
        markers_func.append(
            Function(
                "ForwardKin",
                [symbolic_states],
                [biorbd_model.marker(symbolic_states[:n_q], i).to_mx()],
                ["q"],
                ["marker_" + str(i)],
            ).expand()
        )
    for i in range(ocp.nlp[0]['ns'] + 1):
        for j, mark_func in enumerate(markers_func):
            markers_sol[:, j, i] = np.array(mark_func(vertcat(q[:, i], q_dot[:, i]))).squeeze()
            Q_ref = np.concatenate([q_ref[:, i], np.zeros(n_q)])
            markers_from_q_ref[:, j, i] = np.array(mark_func(Q_ref)).squeeze()

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

    mean_diff_track = [sum(hist_diff_track[0, :]) / nb_marker,
                       sum(hist_diff_track[1, :]) / nb_marker,
                       sum(hist_diff_track[2, :]) / nb_marker]
    mean_diff_sol = [sum(hist_diff_sol[0, :]) / nb_marker,
                     sum(hist_diff_sol[1, :]) / nb_marker,
                     sum(hist_diff_sol[2, :]) / nb_marker]

    # --- Plot markers --- #
    label_markers = []
    for mark in range(nb_marker):
        label_markers.append(ocp.nlp[0]["model"].markerNames()[mark].to_string())

    figure, axes = plt.subplots(2, 2)
    axes = axes.flatten()
    title_markers = ['x axis', 'z axis']
    for i in range(2):
        axes[i].bar(np.linspace(0, nb_marker, nb_marker), hist_diff_track[2 * i, :], width=1.0, facecolor='b',
                    edgecolor='k', alpha=0.5)
        axes[i].set_xticks(np.arange(nb_marker))
        axes[i].set_xticklabels(label_markers, rotation=90)
        axes[i].set_ylabel('Mean differences in ' + title_markers[i] + ' mm')
        axes[i].plot([0, nb_marker], [mean_diff_track[2 * i], mean_diff_track[2 * i]], '--r')
        axes[i].set_title('markers differences between sol and exp')

        axes[i + 2].bar(np.linspace(0, nb_marker, nb_marker), hist_diff_sol[2 * i, :], width=1.0, facecolor='b',
                        edgecolor='k', alpha=0.5)
        axes[i + 2].set_xticks(np.arange(nb_marker))
        axes[i + 2].set_xticklabels(label_markers, rotation=90)
        axes[i + 2].set_ylabel('Mean differences in ' + title_markers[i] + 'mm')
        axes[i + 2].plot([0, nb_marker], [mean_diff_sol[2 * i], mean_diff_sol[2 * i]], '--r')
        axes[i + 2].set_title('markers differences between sol and ref')
    plt.show()

    figure, axes = plt.subplots(2, 2)
    axes = axes.flatten()
    title_markers = ['x axis', 'z axis']
    for i in range(2):
        axes[i].plot(t, diff_track[i, :, :].T)
        axes[i].set_xlabel('time (s)')
        axes[i].set_ylabel('Squared differences in ' + title_markers[i])
        axes[i].set_title('markers differences between sol and exp')

        axes[i + 2].plot(t, diff_sol[2 * i, :, :].T)
        axes[i + 2].set_xlabel('time (s)')
        axes[i + 2].set_ylabel('Squared differences in ' + title_markers[i])
        axes[i + 2].set_title('markers differences between sol and ref')
    plt.show()

    # --- Plot muscles activation --- #
    diff_act = np.sqrt((activation_ref - mus) * (activation_ref - mus))
    hist_diff_act = np.zeros(nb_mus)
    label_muscles = []
    for n_mus in range(nb_mus):
        label_muscles.append(biorbd_model.muscle(n_mus).name().to_string())
        hist_diff_act[n_mus] = sum(diff_act[n_mus, :]) / n_shooting_points
    mean_diff_act = np.sum(hist_diff_act) / nb_mus

    plt.figure()
    plt.bar(np.linspace(0, (nb_mus - 1) , nb_mus), hist_diff_act, width=1.0, facecolor='b',edgecolor='k', alpha=0.5)
    plt.xticks(np.arange(nb_mus), label_muscles, rotation=90)
    plt.ylabel('Mean differences in activation')
    plt.plot([0, nb_mus], [mean_diff_act, mean_diff_act], '--r')
    plt.title('activations differences between solution and reference')

    # def plot_control(ax, t, x, color='b'):
    #     nbPoints = len(np.array(x))
    #     for n in range(nbPoints - 1):
    #         ax.plot([t[n], t[n + 1], t[n + 1]], [x[n], x[n], x[n + 1]], color)
    #
    # figure2, axes2 = plt.subplots(4, 5, sharex=True)
    # axes2 = axes2.flatten()
    # for i in range(biorbd_model.nbMuscleTotal()):
    #     name_mus = biorbd_model.muscle(i).name().to_string()
    #     plot_control(axes2[i], t, mus[i, :], color='r')
    #     plot_control(axes2[i], t, mus_int[i, :], color='b')
    #     axes2[i].set_title(name_mus)
    #     axes2[i].set_ylim([0, 1])
    #     axes2[i].set_yticks(np.arange(0, 1, step=1 / 5, ))
    #     axes2[i].grid(color="k", linestyle="--", linewidth=0.5)
    # axes2[-1].remove()
    # axes2[-2].remove()
    # axes2[-3].remove()

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.graphs()
    result.animate()
