import numpy as np
from casadi import dot, Function, vertcat, MX, mtimes, nlpsol
import biorbd
from matplotlib import pyplot as plt
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
    StateTransition,
    OdeSolver,
)


def get_dispatch_contact_forces(grf_ref, M_ref, coord, nb_shooting):
    p = 0.5  # repartition entre les 2 points

    F_Meta1 = MX.sym("F_Meta1", 3 * (nb_shooting + 1), 1)
    F_Meta5 = MX.sym("F_Meta1", 3 * (nb_shooting + 1), 1)
    M_Meta1 = MX.sym("M_Meta1", 3 * (nb_shooting + 1), 1)
    M_Meta5 = MX.sym("M_Meta1", 3 * (nb_shooting + 1), 1)

    objective = 0
    lbg = []
    ubg = []
    constraint = []
    for i in range(nb_shooting + 1):
        # Aliases
        fm1 = F_Meta1[3 * i : 3 * (i + 1)]
        fm5 = F_Meta5[3 * i : 3 * (i + 1)]

        # sum forces = 0 --> Fp1 + Fp2 = Ftrack
        sf = fm1 + fm5
        jf = sf - grf_ref[:, i]
        constraint += (jf[0], jf[1], jf[2])
        lbg += [0] * 3
        ubg += [0] * 3

        # use of p to dispatch forces --> p*Fp1 - (1-p)*Fp2 = 0
        jf2 = p * fm1 - (1 - p) * fm5
        objective += mtimes(jf2.T, jf2)

        # positive vertical force
        constraint += (fm1[2], fm5[2])
        lbg += [0] * 2
        ubg += [1000] * 2

        # non slipping --> -0.4*Fz < Fx < 0.4*Fz
        constraint += ((-0.4 * fm1[2] - fm1[0]), (-0.4 * fm5[2] - fm5[0]))
        lbg += [-1000] * 2
        ubg += [0] * 2

        constraint += ((0.4 * fm1[2] - fm1[0]), (0.4 * fm5[2] - fm5[0]))
        lbg += [0] * 2
        ubg += [1000] * 2

    w = [F_Meta1, F_Meta5]
    nlp = {"x": vertcat(*w), "f": objective, "g": vertcat(*constraint)}
    opts = {"ipopt.tol": 1e-8, "ipopt.hessian_approximation": "exact"}
    solver = nlpsol("solver", "ipopt", nlp, opts)
    res = solver(x0=np.zeros(6 * (number_shooting_points[2] + 1)), lbx=-1000, ubx=1000, lbg=lbg, ubg=ubg)

    FM1 = res["x"][: 3 * (nb_shooting + 1)]
    FM5 = res["x"][3 * (nb_shooting + 1) :]

    grf_dispatch_ref = np.zeros((3 * 2, nb_shooting + 1))
    for i in range(3):
        grf_dispatch_ref[i, :] = np.array(FM1[i::3]).squeeze()
        grf_dispatch_ref[i + 3, :] = np.array(FM5[i::3]).squeeze()
    return grf_dispatch_ref


def get_last_contact_forces(ocp, nlp, t, x, u, p, data_to_track=()):
    force = nlp["contact_forces_func"](x[-1], u[-1], p)
    val = force - data_to_track[t[-1], :]
    return dot(val, val)


def get_muscles_first_node(ocp, nlp, t, x, u, p):
    activation = x[0][2 * nlp["nbQ"] :]
    excitation = u[0][nlp["nbQ"] :]
    val = activation - excitation
    return val


def modify_isometric_force(biorbd_model, value, fiso_init):
    n_muscle = 0
    for nGrp in range(biorbd_model.nbMuscleGroups()):
        for nMus in range(biorbd_model.muscleGroup(nGrp).nbMuscles()):
            biorbd_model.muscleGroup(nGrp).muscle(nMus).characteristics().setForceIsoMax(
                value[n_muscle] * fiso_init[n_muscle]
            )
            n_muscle += 1


def plot_control(ax, t, x, color="b"):
    nbPoints = len(np.array(x))
    for n in range(nbPoints - 1):
        ax.plot([t[n], t[n + 1], t[n + 1]], [x[n], x[n], x[n + 1]], color)


def prepare_ocp(
    biorbd_model, final_time, nb_shooting, markers_ref, excitation_ref, grf_ref, q_ref, nb_threads, fiso_init,
):

    # Problem parameters
    nb_q = biorbd_model.nbQ()
    nb_qdot = biorbd_model.nbQdot()
    nb_tau = biorbd_model.nbGeneralizedTorque()
    nb_mus = biorbd_model.nbMuscleTotal()

    torque_min, torque_max, torque_init = -1000, 1000, 0
    activation_min, activation_max, activation_init = 0, 1, 0.1

    # Add objective functions
    objective_functions = (
        {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 1, "controls_idx": range(6, nb_tau)},
        {"type": Objective.Lagrange.TRACK_MUSCLES_CONTROL, "weight": 0.1, "data_to_track": excitation_ref[:, :-1].T,},
        # {"type": Objective.Lagrange.TRACK_MARKERS, "weight": 100, "data_to_track": markers_ref},
        {"type": Objective.Lagrange.TRACK_STATE, "weight": 10, "states_idx": range(nb_q), "data_to_track": q_ref.T},
        {"type": Objective.Lagrange.TRACK_CONTACT_FORCES, "weight": 0.00005, "data_to_track": grf_ref.T},
    )

    # Dynamics
    problem_type = (ProblemType.muscle_excitations_and_torque_driven_with_contact,)

    # Constraints
    constraints = {"type": Constraint.CUSTOM, "function": get_muscles_first_node, "instant": Instant.START}

    # Define the parameter to optimize
    bound_length = Bounds(
        min_bound=np.repeat(0.2, nb_mus), max_bound=np.repeat(5, nb_mus), interpolation_type=InterpolationType.CONSTANT
    )
    parameters = {
        "name": "force_isometric",  # The name of the parameter
        "function": modify_isometric_force,  # The function that modifies the biorbd model
        "bounds": bound_length,  # The bounds
        "initial_guess": InitialConditions(np.repeat(1, nb_mus)),  # The initial guess
        "size": nb_mus,  # The number of elements this particular parameter vector has
        "fiso_init": fiso_init,
    }
    # Path constraint
    X_bounds = QAndQDotBounds(biorbd_model)
    X_bounds.concatenate(
        Bounds([activation_min] * biorbd_model.nbMuscles(), [activation_max] * biorbd_model.nbMuscles())
    )

    # Initial guess
    init_x = np.zeros((nb_q + nb_qdot + nb_mus, nb_shooting + 1,))
    # init_x[[0, 1, 5, 8, 9, 11], :] = q_ref
    init_x[:nb_q, :] = q_ref
    init_x[-nb_mus:, :] = excitation_ref
    X_init = InitialConditions(init_x, interpolation_type=InterpolationType.EACH_FRAME)

    # Define control path constraint
    U_bounds = Bounds(
        min_bound=[torque_min] * nb_tau + [activation_min] * nb_mus,
        max_bound=[torque_max] * nb_tau + [activation_max] * nb_mus,
    )

    # Initial guess
    init_u = np.zeros((nb_tau + nb_mus, nb_shooting))
    init_u[1, :] = np.repeat(-500, nb_shooting)
    init_u[-nb_mus:, :] = excitation_ref[:, :-1]
    U_init = InitialConditions(init_u, interpolation_type=InterpolationType.EACH_FRAME)

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
        nb_threads=nb_threads,
        parameters=parameters,
    )


if __name__ == "__main__":
    # Define the problem
    # Model path
    biorbd_model = (
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_1contact_deGroote_3d_Heel.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_3contacts_deGroote_3d.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_contact_deGroote_3d_Forefoot.bioMod"),
    )

    # Problem parameters
    number_shooting_points = [5, 10, 15]

    # Generate data from file
    Data_to_track = Data_to_track("normal01", multiple_contact=True)
    [T, T_stance, T_swing] = Data_to_track.GetTime()
    phase_time = [T_stance[0], T_stance[1], T_stance[2]]  # get time for each phase

    grf_ref = Data_to_track.load_data_GRF(
        biorbd_model[0], T_stance, number_shooting_points
    )  # get ground reaction forces
    M_ref = Data_to_track.load_data_Moment(biorbd_model[0], T_stance, number_shooting_points)
    markers_ref = Data_to_track.load_data_markers(biorbd_model[0], T_stance, number_shooting_points, "stance")
    q_ref = Data_to_track.load_q_kalman(biorbd_model[0], T_stance, number_shooting_points, "stance")
    emg_ref = Data_to_track.load_data_emg(biorbd_model[0], T_stance, number_shooting_points, "stance")
    excitation_ref = []
    for i in range(len(phase_time)):
        excitation_ref.append(Data_to_track.load_muscularExcitation(emg_ref[i]))

    Meta1 = np.array([np.mean(markers_ref[2][0, 21, :]), np.mean(markers_ref[2][1, 21, :]), 0])
    Meta5 = np.array([np.mean(markers_ref[2][0, 24, :]), np.mean(markers_ref[2][1, 24, :]), 0])
    grf_dispatch_ref = get_dispatch_contact_forces(grf_ref[2], M_ref[2], [Meta1, Meta5], number_shooting_points[2])
    grf_dispatch_ref = grf_dispatch_ref[[0, 2, 3, 5], :-1]

    # plt.figure()
    # plt.plot(grf_ref[2][2, :].T, "k--")
    # plt.plot(grf_dispatch_ref.T)
    # plt.legend(("platform", "Meta1", "Meta5"))
    # plt.show()

    # Get initial isometric forces
    fiso_init = []
    n_muscle = 0
    for nGrp in range(biorbd_model[2].nbMuscleGroups()):
        for nMus in range(biorbd_model[2].muscleGroup(nGrp).nbMuscles()):
            fiso_init.append(biorbd_model[2].muscleGroup(nGrp).muscle(nMus).characteristics().forceIsoMax().to_mx())

    ocp = prepare_ocp(
        biorbd_model=biorbd_model[2],
        final_time=phase_time[2],
        nb_shooting=number_shooting_points[2],
        markers_ref=markers_ref[2],
        excitation_ref=excitation_ref[2],
        grf_ref=grf_dispatch_ref,
        q_ref=q_ref[2],
        nb_threads=4,
        fiso_init=fiso_init,
    )

    # --- Solve the program --- #
    sol = ocp.solve(
        solver="ipopt",
        options_ipopt={
            "ipopt.tol": 1e-3,
            "ipopt.max_iter": 5000,
            "ipopt.hessian_approximation": "exact",
            "ipopt.limited_memory_max_history": 50,
            "ipopt.linear_solver": "ma57",
        },
        show_online_optim=False,
    )

    # --- Get Results --- #
    states, controls, params = Data.get_data(ocp, sol, get_parameters=True)
    q, q_dot, activations, tau, excitations = (
        states["q"],
        states["q_dot"],
        states["muscles"],
        controls["tau"],
        controls["muscles"],
    )
    params = params[ocp.nlp[0]["p"].name()]

    # --- Save Results --- #
    np.save("./RES/forefoot/excitations", excitations)
    np.save("./RES/forefoot/activations", activations)
    np.save("./RES/forefoot/tau", tau)
    np.save("./RES/forefoot/q_dot", q_dot)
    np.save("./RES/forefoot/q", q)
    np.save("./RES/forefoot/params", params)

    x = np.concatenate((q, q_dot, activations))
    u = np.concatenate((tau, excitations))
    contact_forces = np.array(ocp.nlp[0]["contact_forces_func"](x[:, :-1], u[:, :-1], params))

    # --- Plot ---
    # Muscles
    figure, axes = plt.subplots(4, 5, sharex=True)
    axes = axes.flatten()
    t = np.linspace(0, phase_time[2], number_shooting_points[2] + 1)
    for i in range(biorbd_model[2].nbMuscleTotal()):
        name_mus = biorbd_model[2].muscle(i).name().to_string()
        param_value = str(np.round(params[i], 2))

        plot_control(axes[i], t, excitation_ref[2][i, :], color="k--")
        plot_control(axes[i], t, excitations[i, :], color="r--")
        axes[i].plot(t, activations[i, :], "r.-", linewidth=0.6)  # without parameters
        axes[i].text(0.03, 0.9, param_value)

        axes[i].set_title(name_mus)
        axes[i].set_ylim([0, 1])
        axes[i].set_yticks(np.arange(0, 1, step=1 / 5,))
        axes[i].grid(color="k", linestyle="--", linewidth=0.5)
    axes[-1].remove()
    axes[-2].remove()
    axes[-3].remove()
    plt.show()

    # Q
    figure, axes = plt.subplots(4, 3, sharex=True)
    axes = axes.flatten()
    for i in range(biorbd_model[2].nbQ()):
        param_value = str(np.round(params[i], 2))

        if i > 2:
            plot_control(axes[i], t, q_ref[2][i, :] * 180 / np.pi, color="k--")
            axes[i].plot(t, q[i, :] * 180 / np.pi, "r.-", linewidth=0.6)
        else:
            plot_control(axes[i], t, q_ref[2][i, :], color="k--")
            axes[i].plot(t, q[i, :], "r.-", linewidth=0.6)  # without parameters

        axes[i].grid(color="k", linestyle="--", linewidth=0.5)
    plt.show()

    # Contact forces
    figure, axes = plt.subplots(2, 2, sharex=True)
    axes = axes.flatten()
    for i in range(biorbd_model[2].nbContacts()):
        plot_control(axes[i], t, grf_dispatch_ref[i, :], color="k--")
        axes[i].plot(t[:-1], contact_forces[i, :], "r-", linewidth=0.6)

        axes[i].grid(color="k", linestyle="--", linewidth=0.5)
    plt.show()

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()