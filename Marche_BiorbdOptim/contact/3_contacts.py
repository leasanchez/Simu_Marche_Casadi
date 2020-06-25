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
    PlotType,
    StateTransition,
)


def get_dispatch_contact_forces(grf_ref, M_ref, coord, nb_shooting):
    # init
    grf_dispatch_ref = np.zeros((6, nb_shooting + 1))
    Heel = coord[0]
    Meta1 = coord[1]
    Meta5 = coord[2]

    # p_heel = np.linspace(0, 1, nb_shooting + 1)
    # p_heel = 1 - p_heel
    x = np.linspace(-number_shooting_points[1], number_shooting_points[1], number_shooting_points[1] + 1, dtype=int)
    p_heel = 1 / (1 + np.exp(-x))
    p = 0.5

    # Forces
    F_Heel = MX.sym("F_Heel", 3 * (nb_shooting + 1), 1)    #xyz
    F_Meta1 = MX.sym("F_Meta1", 1 * (nb_shooting + 1), 1)  #z
    F_Meta5 = MX.sym("F_Meta5", 2 * (nb_shooting + 1), 1)  #xz

    objective = 0
    lbg = []
    ubg = []
    constraint = []
    for i in range(nb_shooting + 1):
        # Aliases
        fh = F_Heel[3 * i : 3 * (i + 1)]
        fm1 = F_Meta1[i]
        fm5 = F_Meta5[2 * i : 2 * (i + 1)]

        # --- Torseur equilibre ---
        # sum forces = 0 --> Fp1 + Fp2 = Ftrack
        jf0 = (fm5[0] + fh[0]) - grf_ref[0, i]
        jf1 = fh[1] - grf_ref[1, i]
        jf2 = (fm1 + fm5[1] + fh[2]) - grf_ref[2, i]
        objective += jf0 * jf0 + jf1 * jf1 + jf2 * jf2

        # sum moments = 0 --> Mp1_P1 + CP1xFp1 + Mp2_P2 + CP2xFp2 = Mtrack
        jm0 = (Heel[1] * fh[2] - Heel[2] * fh[1] + Meta1[1] * fm1 + Meta5[1] * fm5[1]) - M_ref[0, i]
        jm1 = (Heel[2] * fh[0] - Heel[0] * fh[2] - Meta1[0] * fm1 + Meta5[2] * fm5[0] - Meta5[0] * fm5[1]) - M_ref[1, i]
        jm2 = (Heel[0] * fh[1] - Heel[1] * fh[0] - Meta5[1] * fm5[0]) - M_ref[2, i]
        objective += jm0 * jm0 + jm1 * jm1 + jm2 * jm2

        # --- Dispatch on different contact points ---
        # use of p to dispatch forces --> p_heel*Fh - (1-p_heel)*Fm = 0
        jf = p_heel[i] * fh[2] - ((1 - p_heel[i]) * (p * fm1 + (1 - p) * fm5[1]))
        objective += jf * jf

        # --- Forces constraints ---
        # positive vertical force
        constraint += (fh[2], fm1, fm5[1])
        lbg += [0] * 3
        ubg += [1000] * 3

        # non slipping --> -0.4*Fz < Fx < 0.4*Fz
        constraint += ((-0.4 * fh[2] - fh[0]), (-0.4 * fm5[1] - fm5[0]))
        lbg += [-1000] * 2
        ubg += [0] * 2

        constraint += ((0.4 * fh[2] - fh[0]), (0.4 * fm5[1] - fm5[0]))
        lbg += [0] * 2
        ubg += [1000] * 2

    w = [F_Heel, F_Meta1, F_Meta5]
    nlp = {"x": vertcat(*w), "f": objective, "g": vertcat(*constraint)}
    opts = {"ipopt.tol": 1e-8, "ipopt.hessian_approximation": "exact"}
    solver = nlpsol("solver", "ipopt", nlp, opts)
    res = solver(x0=np.zeros(6 * (number_shooting_points[1] + 1)), lbx=-1000, ubx=1000, lbg=lbg, ubg=ubg)

    FH = res["x"][: 3 * (nb_shooting + 1)]
    FM1 = res["x"][3 * (nb_shooting + 1) : 4 * (nb_shooting + 1)]
    FM5 = res["x"][4 * (nb_shooting + 1) : ]

    grf_dispatch_ref[0, :] = np.array(FH[0::3]).squeeze()
    grf_dispatch_ref[1, :] = np.array(FH[1::3]).squeeze()
    grf_dispatch_ref[2, :] = np.array(FH[2::3]).squeeze()
    grf_dispatch_ref[3, :] = np.array(FM1).squeeze()
    grf_dispatch_ref[4, :] = np.array(FM5[0::2]).squeeze()
    grf_dispatch_ref[5, :] = np.array(FM5[1::2]).squeeze()
    return grf_dispatch_ref

def get_dispatch_contact_forces_3d(grf_ref, M_ref, coord, nb_shooting):
    # init
    grf_dispatch_ref = np.zeros((9, nb_shooting + 1))
    Heel = coord[0]
    Meta1 = coord[1]
    Meta5 = coord[2]

    # p_heel = np.linspace(0, 1, nb_shooting + 1)
    # p_heel = 1 - p_heel
    x = np.linspace(-number_shooting_points[1], number_shooting_points[1], number_shooting_points[1] + 1, dtype=int)
    p_heel = 1 / (1 + np.exp(-x))
    p = 0.5

    # Forces
    F_Heel = MX.sym("F_Heel", 3 * (nb_shooting + 1), 1)    #xyz
    F_Meta1 = MX.sym("F_Meta1", 3 * (nb_shooting + 1), 1)  #z
    F_Meta5 = MX.sym("F_Meta5", 3 * (nb_shooting + 1), 1)  #xz

    objective = 0
    lbg = []
    ubg = []
    constraint = []
    for i in range(nb_shooting + 1):
        # Aliases
        fh = F_Heel[3 * i : 3 * (i + 1)]
        fm1 = F_Meta1[3 * i : 3 * (i + 1)]
        fm5 = F_Meta5[3 * i : 3 * (i + 1)]

        # --- Torseur equilibre ---
        # sum forces = 0 --> Fp1 + Fp2 = Ftrack
        jf = (fh + fm1 + fm5) - grf_ref[:, i]
        objective += mtimes(jf.T, jf)

        # sum moments = 0 --> Mp1_P1 + CP1xFp1 + Mp2_P2 + CP2xFp2 = Mtrack
        jm = (dot(Heel, fh) + dot(Meta1, fm1) + dot(Meta5, fm5)) - M_ref[:, i]
        objective += mtimes(jm.T, jm)

        # --- Dispatch on different contact points ---
        jf2 = p_heel[i] * fh - ((1 - p_heel[i]) * (p * fm1 + (1 - p) * fm5))
        objective += mtimes(jf2.T, jf2)

        # --- Forces constraints ---
        # positive vertical force
        constraint += (fh[2], fm1[2], fm5[2])
        lbg += [0] * 3
        ubg += [1000] * 3

        # non slipping --> -0.4*Fz < Fx < 0.4*Fz
        constraint += ((-0.4 * fh[2] - fh[0]), (-0.4 * fm1[2] - fm1[0]), (-0.4 * fm5[2] - fm5[0]))
        lbg += [-1000] * 3
        ubg += [0] * 3

        constraint += ((0.4 * fh[2] - fh[0]), (0.4 * fm1[2] - fm1[0]), (0.4 * fm5[2] - fm5[0]))
        lbg += [0] * 3
        ubg += [1000] * 3

    w = [F_Heel, F_Meta1, F_Meta5]
    nlp = {"x": vertcat(*w), "f": objective, "g": vertcat(*constraint)}
    opts = {"ipopt.tol": 1e-8, "ipopt.hessian_approximation": "exact"}
    solver = nlpsol("solver", "ipopt", nlp, opts)
    res = solver(x0=np.zeros(9 * (number_shooting_points[1] + 1)), lbx=-1000, ubx=1000, lbg=lbg, ubg=ubg)

    FH = res["x"][: 3 * (nb_shooting + 1)]
    FM1 = res["x"][3 * (nb_shooting + 1) : 6 * (nb_shooting + 1)]
    FM5 = res["x"][6 * (nb_shooting + 1) : ]

    grf_dispatch_ref[0, :] = np.array(FH[0::3]).squeeze()
    grf_dispatch_ref[1, :] = np.array(FH[1::3]).squeeze()
    grf_dispatch_ref[2, :] = np.array(FH[2::3]).squeeze()

    grf_dispatch_ref[3, :] = np.array(FM1[0::3]).squeeze()
    grf_dispatch_ref[4, :] = np.array(FM1[1::3]).squeeze()
    grf_dispatch_ref[5, :] = np.array(FM1[2::3]).squeeze()

    grf_dispatch_ref[6, :] = np.array(FM5[0::3]).squeeze()
    grf_dispatch_ref[7, :] = np.array(FM5[1::3]).squeeze()
    grf_dispatch_ref[8, :] = np.array(FM5[2::3]).squeeze()
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


def prepare_ocp(
    biorbd_model, final_time, nb_shooting, markers_ref, excitation_ref, grf_ref, q_ref, fiso_init,
):

    # Problem parameters
    nb_q = biorbd_model[0].nbQ()
    nb_qdot = biorbd_model[0].nbQdot()
    nb_tau = biorbd_model[0].nbGeneralizedTorque()
    nb_mus = biorbd_model[0].nbMuscleTotal()
    nb_phases = len(biorbd_model)

    torque_min, torque_max, torque_init = -1000, 1000, 0
    activation_min, activation_max, activation_init = 0, 1, 0.1

    # Add objective functions
    objective_functions = (
        (
            {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 1, "controls_idx": range(6, nb_tau)},
            {
                "type": Objective.Lagrange.TRACK_MUSCLES_CONTROL,
                "weight": 0.1,
                "data_to_track": excitation_ref[0][:, :-1].T,
            },
            {"type": Objective.Lagrange.TRACK_MARKERS, "weight": 500, "data_to_track": markers_ref[0]},
            {"type": Objective.Lagrange.TRACK_CONTACT_FORCES, "weight": 0.000005, "data_to_track": grf_ref[0].T},
        ),
        (
            {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 1, "controls_idx": range(6, nb_tau)},
            {
                "type": Objective.Lagrange.TRACK_MUSCLES_CONTROL,
                "weight": 0.1,
                "data_to_track": excitation_ref[1][:, :-1].T,
            },
            {"type": Objective.Lagrange.TRACK_MARKERS, "weight": 500, "data_to_track": markers_ref[1]},
            {"type": Objective.Lagrange.TRACK_CONTACT_FORCES, "weight": 0.0000005, "data_to_track": grf_ref[1].T},
            {
                "type": Objective.Mayer.CUSTOM,
                "weight": 0.0000005,
                "function": get_last_contact_forces,
                "data_to_track": grf_ref[1].T,
                "instant": Instant.ALL,
            },
        ),
    )

    # Dynamics
    problem_type = (
        ProblemType.muscle_excitations_and_torque_driven_with_contact,
        ProblemType.muscle_excitations_and_torque_driven_with_contact,
    )

    # Constraints
    constraints = ({"type": Constraint.CUSTOM, "function": get_muscles_first_node, "instant": Instant.START},)

    # State Transitions
    state_transitions = ({"type": StateTransition.IMPACT, "phase_pre_idx": 0,},)

    # Define the parameter to optimize
    bound_length = Bounds(
        min_bound=np.repeat(0.2, nb_mus), max_bound=np.repeat(5, nb_mus), interpolation_type=InterpolationType.CONSTANT
    )
    parameters = (
        {
            "name": "force_isometric",  # The name of the parameter
            "function": modify_isometric_force,  # The function that modifies the biorbd model
            "bounds": bound_length,  # The bounds
            "initial_guess": InitialConditions(np.repeat(1, nb_mus)),  # The initial guess
            "size": nb_mus,  # The number of elements this particular parameter vector has
            "fiso_init": fiso_init,
        },
    )

    # Path constraint
    X_bounds = []
    for i in range(nb_phases):
        XB = QAndQDotBounds(biorbd_model[i])
        XB.concatenate(
            Bounds([activation_min] * biorbd_model[i].nbMuscles(), [activation_max] * biorbd_model[i].nbMuscles())
        )
        X_bounds.append(XB)

    # Initial guess
    X_init = []
    for n_p in range(len(biorbd_model)):
        Q = np.zeros((nb_q, number_shooting_points[n_p] + 1))
        Q[:12] = q_ref[n_p]
        init_x = np.zeros((nb_q + nb_qdot + nb_mus, nb_shooting[n_p] + 1,))
        init_x[:nb_q, :] = Q
        init_x[-nb_mus:, :] = excitation_ref[n_p]
        XI = InitialConditions(init_x, interpolation_type=InterpolationType.EACH_FRAME)
        X_init.append(XI)

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
    for n_p in range(len(biorbd_model)):
        init_u = np.zeros((nb_tau + nb_mus, nb_shooting[n_p]))
        init_u[1, :] = np.repeat(-500, nb_shooting[n_p])
        init_u[-nb_mus:, :] = excitation_ref[n_p][:, :-1]
        UI = InitialConditions(init_u, interpolation_type=InterpolationType.EACH_FRAME)
        U_init.append(UI)

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        problem_type,
        nb_shooting,
        final_time,
        X_init,
        U_init,
        X_bounds=X_bounds,
        U_bounds=U_bounds,
        objective_functions=objective_functions,
        constraints=(constraints, ()),
        parameters=(parameters, parameters),
        state_transitions=state_transitions,
    )


if __name__ == "__main__":
    # Define the problem
    # Model path
    biorbd_model = (
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_1contact_deGroote_3d_Heel.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_3contacts_deGroote_3d.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_contact_deGroote_3d_Forefoot.bioMod"),
    )
    model_segments = (
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_1contact_deGroote_3d_Heel_3segments.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_3contacts_deGroote_3d_3segments.bioMod"),
    )
    # Problem parameters
    number_shooting_points = [10, 5, 15]

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

    Heel = np.array([np.mean(markers_ref[1][0, 19, :] + 0.04), np.mean(markers_ref[1][1, 19, :]), 0])
    Meta1 = np.array([np.mean(markers_ref[1][0, 21, :]), np.mean(markers_ref[1][1, 21, :]), 0])
    Meta5 = np.array([np.mean(markers_ref[1][0, 25, :]), np.mean(markers_ref[1][1, 25, :]), 0])
    grf_dispatch_ref = get_dispatch_contact_forces(
        grf_ref[1], M_ref[1], [Heel, Meta1, Meta5], number_shooting_points[1]
    )
    grf_dispatch_ref_3d = get_dispatch_contact_forces_3d(
        grf_ref[1], M_ref[1], [Heel, Meta1, Meta5], number_shooting_points[1]
    )

    # Get initial isometric forces
    fiso_init = []
    n_muscle = 0
    for nGrp in range(biorbd_model[2].nbMuscleGroups()):
        for nMus in range(biorbd_model[2].muscleGroup(nGrp).nbMuscles()):
            fiso_init.append(biorbd_model[2].muscleGroup(nGrp).muscle(nMus).characteristics().forceIsoMax().to_mx())

    ocp = prepare_ocp(
        biorbd_model=model_segments,
        final_time=(phase_time[0], phase_time[1]),
        nb_shooting=(number_shooting_points[0], number_shooting_points[1]),
        markers_ref=(markers_ref[0], markers_ref[1]),
        excitation_ref=(excitation_ref[0], excitation_ref[1]),
        grf_ref=(grf_ref[0], grf_dispatch_ref_3d),
        q_ref=(q_ref[0], q_ref[1]),
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
    states_sol, controls_sol, params_sol = Data.get_data(ocp, sol["x"], get_parameters=True)
    q = states_sol["q"]
    q_dot = states_sol["q_dot"]
    activations = states_sol["muscles"]
    tau = controls_sol["tau"]
    excitations = controls_sol["muscles"]
    params = params_sol[ocp.nlp[0]["p"].name()]

    # --- Save Results --- #
    np.save("./RES/heel_strike/excitations", excitations)
    np.save("./RES/heel_strike/activations", activations)
    np.save("./RES/heel_strike/tau", tau)
    np.save("./RES/heel_strike/q_dot", q_dot)
    np.save("./RES/heel_strike/q", q)
    np.save("./RES/heel_strike/params", params)

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
    result.graphs()
