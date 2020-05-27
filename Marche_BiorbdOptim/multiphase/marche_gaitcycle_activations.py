import numpy as np
from scipy.integrate import solve_ivp
from casadi import dot, Function, vertcat, MX, tanh
from matplotlib import pyplot as plt
import sys

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
    PlotType,
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

    return activation_ref

def prepare_ocp(
    biorbd_model,
    final_time,
    nb_shooting,
    markers_ref,
    activation_ref,
    grf_ref,
    q_ref,
):
    # Problem parameters
    nb_phases = len(biorbd_model)
    nb_q = biorbd_model[0].nbQ()
    torque_min, torque_max, torque_init = -1000, 1000, 0
    activation_min, activation_max, activation_init = 0, 1, 0.1

    # Add objective functions
    objective_functions = ((
        {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 100, "controls_idx":[3, 4, 5]},
        {"type": Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, "weight": 10, "data_to_track":activation_ref[0].T},
        {"type": Objective.Lagrange.TRACK_MARKERS, "weight": 50, "data_to_track": markers_ref[0]},
        {"type": Objective.Lagrange.TRACK_CONTACT_FORCES, "weight": 0.05, "data_to_track": grf_ref[:, :-1].T},
    ),
    (
        {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 100, "controls_idx":[3, 4, 5]},
        {"type": Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, "weight": 10, "data_to_track":activation_ref[1].T},
        {"type": Objective.Lagrange.TRACK_MARKERS, "weight": 50, "data_to_track": markers_ref[1]}
    ))

    # Dynamics
    problem_type = (
        ProblemType.muscles_and_torque_driven_with_contact,
        ProblemType.muscle_activations_and_torque_driven,
    )

    # Constraints
    constraints = ()

    # Path constraint
    X_bounds = [QAndQDotBounds(biorbd_model[i])for i in range(nb_phases)]

    # Initial guess
    init_x_stance = np.zeros((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), nb_shooting[0] + 1))
    for i in range(nb_shooting[0] + 1):
        init_x_stance[:biorbd_model[0].nbQ(), i] = q_ref[0][:, i]
    init_x_swing= np.zeros((biorbd_model[1].nbQ() + biorbd_model[1].nbQdot(), nb_shooting[1] + 1))
    for i in range(nb_shooting[1] + 1):
        init_x_swing[:biorbd_model[1].nbQ(), i] = q_ref[1][:, i]

    X_init = [InitialConditions(init_x_stance, interpolation_type=InterpolationType.EACH_FRAME),
              InitialConditions(init_x_swing, interpolation_type=InterpolationType.EACH_FRAME)]

    # Define control path constraint
    U_bounds = [
        Bounds(
        min_bound = [torque_min] * biorbd_model[i].nbGeneralizedTorque() + [activation_min] * biorbd_model[i].nbMuscleTotal(),
        max_bound = [torque_max] * biorbd_model[i].nbGeneralizedTorque() + [activation_max] * biorbd_model[i].nbMuscleTotal(),
    )
        for i in range(nb_phases)]

    init_u_stance = np.zeros((biorbd_model[0].nbGeneralizedTorque() + biorbd_model[0].nbMuscleTotal(), nb_shooting[0]))
    for i in range(nb_shooting[0]):
        init_u_stance[:biorbd_model[0].nbQ(), i] = [0, -500, 0, 0, 0, 0]
        init_u_stance[-biorbd_model[0].nbMuscleTotal():, i] = activation_ref[0][:, i]

    init_u_swing = np.zeros((biorbd_model[1].nbGeneralizedTorque() + biorbd_model[1].nbMuscleTotal(), nb_shooting[1]))
    for i in range(nb_shooting[1]):
        init_u_swing[-biorbd_model[1].nbMuscleTotal():, i] = activation_ref[1][:, i]

    U_init = [InitialConditions(init_u_stance, interpolation_type=InterpolationType.EACH_FRAME),
              InitialConditions(init_u_swing, interpolation_type=InterpolationType.EACH_FRAME)]

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
    # Model path
    biorbd_model = (
        biorbd.Model("../../ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact.bioMod"),
        biorbd.Model("../../ModelesS2M/ANsWER_Rleg_6dof_17muscle_0contact.bioMod"),
    )

    # Problem parameters
    number_shooting_points = [25, 25]

    # Generate data from file
    from Marche_BiorbdOptim.LoadData import load_data_markers, load_data_q, load_data_emg, load_data_GRF, load_muscularExcitation

    name_subject = "equincocont01"
    grf_ref, T, T_stance, T_swing = load_data_GRF(name_subject, biorbd_model, number_shooting_points[0])
    phase_time = [T_stance, T_swing]

    # phase stance
    t_stance, markers_ref_stance = load_data_markers(name_subject, biorbd_model[0], phase_time[0], number_shooting_points[0], 'stance')
    q_ref_stance = load_data_q(name_subject, biorbd_model[0], phase_time[0], number_shooting_points[0], 'stance')
    emg_ref_stance = load_data_emg(name_subject, biorbd_model[0], phase_time[0], number_shooting_points[0], 'stance')
    excitation_ref_stance = load_muscularExcitation(emg_ref_stance)
    activation_ref_stance = generate_activation(biorbd_model=biorbd_model[0], final_time=phase_time[0], nb_shooting=number_shooting_points[0], emg_ref=excitation_ref_stance)

    # phase swing
    t_swing, markers_ref_swing = load_data_markers(name_subject, biorbd_model[1], phase_time[1], number_shooting_points[1], 'swing')
    q_ref_swing = load_data_q(name_subject, biorbd_model[1], phase_time[1], number_shooting_points[1], 'swing')
    emg_ref_swing = load_data_emg(name_subject, biorbd_model[1], phase_time[1], number_shooting_points[1], 'swing')
    excitation_ref_swing = load_muscularExcitation(emg_ref_swing)
    activation_ref_swing = generate_activation(biorbd_model=biorbd_model[1], final_time=phase_time[1], nb_shooting=number_shooting_points[1], emg_ref=excitation_ref_swing)


    # Track these data
    biorbd_model = (
        biorbd.Model("../../ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact.bioMod"),
        biorbd.Model("../../ModelesS2M/ANsWER_Rleg_6dof_17muscle_0contact.bioMod"),
    )
    ocp = prepare_ocp(
        biorbd_model,
        phase_time,
        number_shooting_points,
        markers_ref=[markers_ref_stance, markers_ref_swing],
        activation_ref=[activation_ref_stance[:, :-1], activation_ref_swing[:, :-1]],
        grf_ref=grf_ref[1:, :],
        q_ref=[q_ref_stance, q_ref_swing],
    )

    # # --- Add plots --- #
    # q_ref_plot = np.zeros((biorbd_model[0].nbQ(), number_shooting_points[0] + number_shooting_points[1] + 1))
    # q_ref_plot[:, :number_shooting_points[0] + 1] = q_ref_stance
    # q_ref_plot[:, number_shooting_points[0]:] = q_ref_swing
    # ocp.add_plot("q", lambda x, u: q_ref_plot, PlotType.STEP, color="tab:red")

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

    # --- Compute ground reaction forces --- #
    contact_forces = np.zeros((2, sum([nlp["ns"] for nlp in ocp.nlp]) + 1))
    grf = np.zeros((2, sum([nlp["ns"] for nlp in ocp.nlp]) + 1))

    x = vertcat(q[:, :number_shooting_points[0] + 1], q_dot[:, :number_shooting_points[0] + 1])
    u = vertcat(tau[:, :number_shooting_points[0] + 1], mus[:, :number_shooting_points[0] + 1])
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

    markers_sol = np.ndarray((3, nb_markers, sum([nlp["ns"] for nlp in ocp.nlp]) + 1))
    markers_from_q_ref = np.ndarray((3, nb_markers, sum([nlp["ns"] for nlp in ocp.nlp]) + 1))
    markers_ref = np.ndarray((3, nb_markers, sum([nlp["ns"] for nlp in ocp.nlp]) + 1))
    markers_ref[:, :, :ocp.nlp[0]["ns"] + 1] = markers_ref_stance
    markers_ref[:, :, ocp.nlp[0]["ns"]: ocp.nlp[0]["ns"] + ocp.nlp[1]["ns"] + 1] = markers_ref_swing

    markers_func = []
    symbolic_states = MX.sym("x", ocp.nlp[0]["nx"], 1)
    symbolic_controls = MX.sym("u", ocp.nlp[0]["nu"], 1)
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
