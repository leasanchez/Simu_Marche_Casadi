from scipy.integrate import solve_ivp
import numpy as np
from casadi import MX, Function, tanh, vertcat
from matplotlib import pyplot as plt
import biorbd
import sys
sys.path.append('/home/leasanchez/programmation/BiorbdOptim')


from biorbd_optim import (
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
)

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

    return activation_ref


def prepare_ocp(
    biorbd_model,
    final_time,
    nb_shooting,
    markers_ref,
    activation_ref,
    q_ref,
):
    # Problem parameters
    torque_min, torque_max, torque_init = -1000, 1000, 0
    activation_min, activation_max, activation_init = 0, 1, 0.1

    # Add objective functions
    objective_functions = (
        {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 100, "controls_idx":[3, 4, 5]},
        {"type": Objective.Lagrange.TRACK_MUSCLES_CONTROL, "weight": 10, "data_to_track": activation_ref.T},
        {"type": Objective.Lagrange.TRACK_MARKERS, "weight": 50, "data_to_track": markers_ref}
    )

    # Dynamics
    variable_type = ProblemType.muscle_activations_and_torque_driven

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
        init_u[-biorbd_model.nbMuscleTotal():, i] = activation_ref[:, i]
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
    )


if __name__ == "__main__":
    # Define the problem
    biorbd_model = biorbd.Model("../../ModelesS2M/ANsWER_Rleg_6dof_17muscle_0contact.bioMod")
    final_time = 0.37
    n_shooting_points = 25
    Gaitphase = 'swing'

    # Generate data from file
    from Marche_BiorbdOptim.LoadData import load_data_markers, load_data_q, load_data_emg, load_muscularExcitation
    name_subject = "equincocont01"
    t, markers_ref = load_data_markers(name_subject, biorbd_model, final_time, n_shooting_points, Gaitphase)
    q_ref = load_data_q(name_subject, biorbd_model, final_time, n_shooting_points, Gaitphase)
    emg_ref = load_data_emg(name_subject, biorbd_model, final_time, n_shooting_points, Gaitphase)
    excitation_ref = load_muscularExcitation(emg_ref)
    activation_ref = generate_activation(biorbd_model=biorbd_model, final_time=final_time, nb_shooting=n_shooting_points, emg_ref=excitation_ref)

    # Track these data
    biorbd_model = biorbd.Model("../../ModelesS2M/ANsWER_Rleg_6dof_17muscle_0contact.bioMod")
    ocp = prepare_ocp(
        biorbd_model,
        final_time,
        n_shooting_points,
        markers_ref,
        activation_ref,
        q_ref,
    )
    def get_markers_pos(x, idx_coord):
        marker_pos = []
        for i in range(x.shape[1]):
            marker_pos.append(markers_func_all(x[:, i]))
        marker_pos = vertcat(*marker_pos)
        return marker_pos[idx_coord::3, :].T

    # --- Add plots --- #
    ocp.add_plot("q", lambda x, u: q_ref, PlotType.STEP, color="tab:red")


    symbolic_states = MX.sym("x", ocp.nlp[0]["nx"], 1)
    symbolic_controls = MX.sym("u", ocp.nlp[0]["nu"], 1)
    markers_func_all = Function(
        "ForwardKin_all", [symbolic_states], [biorbd_model.markers(symbolic_states[:biorbd_model.nbQ()])], ["q"], ["markers"],
    ).expand()

    label_markers = []
    title_markers = ['x', 'y', 'z']
    for mark in range(biorbd_model.nbMarkers()):
        label_markers.append(ocp.nlp[0]["model"].markerNames()[mark].to_string())

    ocp.add_plot("Markers plot coordinates", update_function=lambda x, u: markers_ref[0, :, :], plot_type=PlotType.STEP, color="black", legend = label_markers)
    ocp.add_plot("Markers plot coordinates", update_function=lambda x, u: get_markers_pos(x, 0), plot_type=PlotType.PLOT, color="tab:red")
    ocp.add_plot("Markers plot coordinates", update_function=lambda x, u: markers_ref[1, :, :], plot_type=PlotType.STEP, color="black", legend = label_markers)
    ocp.add_plot("Markers plot coordinates", update_function=lambda x, u: get_markers_pos(x, 1), plot_type=PlotType.PLOT, color="tab:green")
    ocp.add_plot("Markers plot coordinates", update_function=lambda x, u: markers_ref[2, :, :], plot_type=PlotType.STEP, color="black", legend = label_markers)
    ocp.add_plot("Markers plot coordinates", update_function=lambda x, u: get_markers_pos(x, 2), plot_type=PlotType.PLOT, color="tab:blue")

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
    states_sol, controls_sol = Data.get_data(ocp, sol["x"])
    q = states_sol["q"]
    q_dot = states_sol["q_dot"]
    tau = controls_sol["tau"]
    activations = controls_sol["muscles"]

    n_q = ocp.nlp[0]["model"].nbQ()
    n_qdot = ocp.nlp[0]["model"].nbQdot()
    n_mark = ocp.nlp[0]["model"].nbMarkers()
    n_frames = q.shape[1]

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
        plot_control(axes2[i], t[:-1], activations[i, :-1])
        axes2[i].set_title(name_mus)

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

    diff_track = (markers_sol - markers_ref) * (markers_sol - markers_ref)
    diff_sol = (markers_sol - markers_from_q_ref) * (markers_sol - markers_from_q_ref)
    hist_diff_track = np.zeros((3, nb_markers))
    hist_diff_sol = np.zeros((3, nb_markers))

    for n_mark in range(nb_markers):
        hist_diff_track[0, n_mark] = sum(diff_track[0, n_mark, :])/nb_markers
        hist_diff_track[1, n_mark] = sum(diff_track[1, n_mark, :])/nb_markers
        hist_diff_track[2, n_mark] = sum(diff_track[2, n_mark, :])/nb_markers

        hist_diff_sol[0, n_mark] = sum(diff_sol[0, n_mark, :])/nb_markers
        hist_diff_sol[1, n_mark] = sum(diff_sol[1, n_mark, :])/nb_markers
        hist_diff_sol[2, n_mark] = sum(diff_sol[2, n_mark, :])/nb_markers

    mean_diff_track = [sum(hist_diff_track[0, :]) / nb_markers,
                       sum(hist_diff_track[1, :]) / nb_markers,
                       sum(hist_diff_track[2, :]) / nb_markers]
    mean_diff_sol = [sum(hist_diff_sol[0, :]) / nb_markers,
                       sum(hist_diff_sol[1, :]) / nb_markers,
                       sum(hist_diff_sol[2, :]) / nb_markers]
    # markers
    label_markers = []
    title_markers = ['x', 'y', 'z']
    for mark in range(nb_markers):
        label_markers.append(ocp.nlp[0]["model"].markerNames()[mark].to_string())

    figure, axes = plt.subplots(2, 3)
    axes = axes.flatten()
    title_markers = ['x', 'y', 'z']
    for i in range(3):
        axes[i].bar(np.linspace(0,nb_markers, nb_markers), hist_diff_track[i, :], width=1.0, facecolor='b', edgecolor='k', alpha=0.5)
        axes[i].set_xticks(np.arange(nb_markers))
        axes[i].set_xticklabels(label_markers, rotation=90)
        axes[i].set_title(title_markers[i])
        axes[i].plot([0, nb_markers], [mean_diff_track[i], mean_diff_track[i]], '--r')

        axes[i + 3].bar(np.linspace(0,nb_markers, nb_markers), hist_diff_sol[i, :], width=1.0, facecolor='b', edgecolor='k', alpha=0.5)
        axes[i + 3].set_xticks(np.arange(nb_markers))
        axes[i + 3].set_xticklabels(label_markers, rotation=90)
        axes[i + 3].set_title(title_markers[i])
        axes[i + 3].plot([0, nb_markers], [mean_diff_sol[i], mean_diff_sol[i]], '--r')

        if (i==1):
            axes[i].set_title('markers differences between sol and exp')
            axes[i + 3].set_title('markers differences between sol and ref')

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate(show_meshes=False)
    result.graphs()
