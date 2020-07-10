import numpy as np
from casadi import dot, Function, vertcat, MX, mtimes, nlpsol
import biorbd
from matplotlib import pyplot as plt
from Marche_BiorbdOptim.LoadData import Data_to_track

from biorbd_optim import (
    OptimalControlProgram,
    DynamicsTypeList,
    DynamicsType,
    BoundsList,
    Bounds,
    QAndQDotBounds,
    InitialConditionsList,
    InitialConditions,
    ShowResult,
    ObjectiveList,
    Objective,
    InterpolationType,
    Data,
    ParametersList,
    Instant,
    ConstraintList,
    Constraint,
    StateTransitionList,
    StateTransition,
)


def get_dispatch_forefoot_contact_forces(grf_ref, M_CoP, coord, CoP, nb_shooting):
    grf_dispatch_ref = np.zeros((5, nb_shooting + 1))
    Meta1_pos = coord[0]
    Meta5_pos = coord[1]

    F_Meta1 = MX.sym("F_Meta1", 2 * (number_shooting_points[2] + 1), 1)
    F_Meta5 = MX.sym("F_Meta5", 3 * (number_shooting_points[2] + 1), 1)

    objective = 0
    lbg = []
    ubg = []
    constraint = []
    for i in range(number_shooting_points[2] + 1):
        # Aliases
        fm1 = F_Meta1[2 * i: 2 * (i + 1)]
        fm5 = F_Meta5[3 * i: 3 * (i + 1)]
        Meta1 = CoP[:, i] - Meta1_pos
        Meta5 = CoP[:, i] - Meta5_pos

        # sum forces = 0 --> Fp1 + Fp2 = Ftrack
        jf0 = (fm1[0] + fm5[0]) - grf_ref[0, i]
        jf1 = fm5[1] - grf_ref[1, i]
        jf2 = (fm1[1] + fm5[2]) - grf_ref[2, i]
        objective += jf0 * jf0 + jf1 * jf1 + jf2 * jf2

        # sum moments = 0
        jm0 = (Meta1[1] * fm1[1] + Meta5[1] * fm5[2] - Meta5[2] * fm5[1])
        jm1 = (Meta1[2] * fm1[0] - Meta1[0] * fm1[1] + Meta5[2] * fm5[0] - Meta5[0] * fm5[2])
        jm2 = (Meta5[0] * fm5[1] - Meta1[1] * fm1[0] - Meta5[1] * fm5[0]) - M_CoP[2, i]
        objective += jm0 * jm0 + jm1 * jm1 + jm2 * jm2

    x0 = np.concatenate(
        (grf_ref[0, :] / 2, grf_ref[2, :] / 2, grf_ref[0, :] / 2, grf_ref[1, :], grf_ref[2, :] / 2))

    w = [F_Meta1, F_Meta5]
    nlp = {"x": vertcat(*w), "f": objective, "g": vertcat(*constraint)}
    opts = {"ipopt.tol": 1e-8, "ipopt.hessian_approximation": "exact"}
    solver = nlpsol("solver", "ipopt", nlp, opts)
    res = solver(x0=x0, lbx=-5000, ubx=5000, lbg=lbg, ubg=ubg)

    FM1 = res["x"][: 2 * (number_shooting_points[2] + 1)]
    FM5 = res["x"][2 * (number_shooting_points[2] + 1):]

    grf_dispatch_ref[0, :] = np.array(FM1[0::2]).squeeze()
    grf_dispatch_ref[1, :] = np.array(FM1[1::2]).squeeze()
    grf_dispatch_ref[2, :] = np.array(FM5[0::3]).squeeze()
    grf_dispatch_ref[3, :] = np.array(FM5[1::3]).squeeze()
    grf_dispatch_ref[4, :] = np.array(FM5[2::3]).squeeze()
    return grf_dispatch_ref


def get_dispatch_flatfoot_contact_forces(grf_ref, M_CoP, coord, CoP, nb_shooting):
    # init
    grf_dispatch_ref = np.zeros((6, nb_shooting + 1))
    Heel_pos = coord[0]
    Meta1_pos = coord[1]
    Meta5_pos = coord[2]

    for i in range(number_shooting_points[1] + 1):
        Heel = CoP[:, i] - Heel_pos
        Meta1 = CoP[:, i] - Meta1_pos
        Meta5 = CoP[:, i] - Meta5_pos
        A = np.array([[1, 0, 0, 0, 1, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 1, 0, 1],
                      [0, -Heel[2], Heel[1], Meta1[1], 0, Meta5[1]],
                      [Heel[2], 0, -Heel[0], -Meta1[0], Meta5[2], -Meta5[0]],
                      [-Heel[1], Heel[0], 0, 0, -Meta5[1], 0]])
        grf_dispatch_ref[:, i] = np.linalg.solve(A, np.concatenate((grf_ref[:, i], M_CoP[:, i])))
    return grf_dispatch_ref


def get_last_contact_forces(ocp, nlp, t, x, u, p, data_to_track=()):
    force = nlp["contact_forces_func"](x[-1], u[-1], p)
    val = force - data_to_track[t[-1], :]
    return mtimes(val.T, val)


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
    biorbd_model, final_time, nb_shooting, markers_ref, excitation_ref, grf_ref, q_ref, qdot_ref, fiso_init,
):

    # Problem parameters
    nb_phases = len(biorbd_model)
    nb_q = biorbd_model[0].nbQ()
    nb_qdot = biorbd_model[0].nbQdot()
    nb_tau = biorbd_model[0].nbGeneralizedTorque()
    nb_mus = biorbd_model[0].nbMuscleTotal()

    torque_min, torque_max, torque_init = -1000, 1000, 0
    activation_min, activation_max, activation_init = 0, 1, 0.1

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Mayer.CUSTOM, custom_function=get_last_contact_forces, instant=Instant.ALL, weight=0.00005, data_to_track=grf_ref[2].T, phase=2)
    for p in range(nb_phases):
        objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=1, controls_idx=range(6, nb_q), phase=p)
        objective_functions.add(Objective.Lagrange.TRACK_CONTACT_FORCES, weight=0.00005, data_to_track=grf_ref[p].T, phase=p)
        objective_functions.add(Objective.Lagrange.TRACK_MUSCLES_CONTROL, weight=0.001, data_to_track=excitation_ref[p][:, :-1].T, phase=p)
        objective_functions.add(Objective.Lagrange.TRACK_MARKERS, weight=500, data_to_track=markers_ref[p], phase=p)

    # Dynamics
    dynamics = DynamicsTypeList()
    for p in range(nb_phases):
        dynamics.add(DynamicsType.MUSCLE_EXCITATIONS_AND_TORQUE_DRIVEN_WITH_CONTACT, phase=p)

    # Constraints
    constraints = ConstraintList()
    constraints.add(Constraint.CUSTOM, custom_function=get_muscles_first_node, instant=Instant.START)

    # State Transitions
    state_transitions = StateTransitionList()
    state_transitions.add(StateTransition.IMPACT, phase_pre_idx=0)

    # Define the parameter to optimize
    parameters = ParametersList()
    bound_length = Bounds(
        min_bound=np.repeat(0.2, nb_mus), max_bound=np.repeat(5, nb_mus), interpolation=InterpolationType.CONSTANT
    )
    for p in range(nb_phases):
        parameters.add(
            parameter_name="force_isometric",  # The name of the parameter
            function=modify_isometric_force,  # The function that modifies the biorbd model
            initial_guess=InitialConditions(np.repeat(1, nb_mus)),  # The initial guess
            bounds=bound_length,  # The bounds
            size=nb_mus, # The number of elements this particular parameter vector has
            fiso_init=fiso_init,
       )

    # Path constraint
    x_bounds = BoundsList()
    u_bounds = BoundsList()
    for p in range(nb_phases):
        x_bounds.add(QAndQDotBounds(biorbd_model[p]))
        x_bounds[p].concatenate(
            Bounds([activation_min] * nb_mus, [activation_max] * nb_mus)
        )
        u_bounds.add([
            [torque_min] * nb_tau + [activation_min] * nb_mus,
            [torque_max] * nb_tau + [activation_max] * nb_mus,
        ])

    # Initial guess
    x_init = InitialConditionsList()
    for p in range(nb_phases):
        init_x = np.zeros((nb_q + nb_qdot + nb_mus, nb_shooting[p] + 1))
        init_x[:nb_q, :] = q_ref[p]
        init_x[nb_q:nb_q + nb_qdot, :] = qdot_ref[p]
        init_x[-nb_mus :, :] = excitation_ref[p]
        x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)

    u_init = InitialConditionsList()
    for p in range(nb_phases):
        init_u = np.zeros((nb_tau + nb_mus, nb_shooting[p]))
        init_u[-nb_mus:, :] = excitation_ref[p][:, :-1]
        init_u[1, :] = np.repeat(-500, nb_shooting[p])
        u_init.add(init_u, interpolation=InterpolationType.EACH_FRAME)

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        nb_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        constraints,
        parameters=parameters,
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

    # Problem parameters
    number_shooting_points = (5, 10, 25)

    # Generate data from file
    Data_to_track = Data_to_track("normal01", multiple_contact=True)
    [T, T_stance, T_swing] = Data_to_track.GetTime()
    phase_time = (T_stance[0], T_stance[1], T_stance[2])  # get time for each phase

    grf_ref = Data_to_track.load_data_GRF(
        biorbd_model[0], T_stance, number_shooting_points
    )  # get ground reaction forces
    M_ref = Data_to_track.load_data_Moment(biorbd_model[0], T_stance, number_shooting_points)
    markers_ref = Data_to_track.load_data_markers(biorbd_model[0], T_stance, number_shooting_points, "stance")
    q_ref = Data_to_track.load_q_kalman(biorbd_model[0], T_stance, number_shooting_points, "stance")
    qdot_ref = Data_to_track.load_qdot_kalman(biorbd_model[0], T_stance, number_shooting_points, "stance")
    emg_ref = Data_to_track.load_data_emg(biorbd_model[0], T_stance, number_shooting_points, "stance")
    excitation_ref = []
    for i in range(len(phase_time)):
        excitation_ref.append(Data_to_track.load_muscularExcitation(emg_ref[i]))
    M_CoP = Data_to_track.load_data_Moment_at_CoP(biorbd_model[0], T_stance, number_shooting_points)
    CoP = Data_to_track.load_data_CoP(biorbd_model[0], T_stance, number_shooting_points)

    Heel = np.array([np.mean(markers_ref[1][0, 19, :] + 0.04), np.mean(markers_ref[1][1, 19, :]), 0])
    Meta1 = np.array([np.mean(markers_ref[1][0, 20, :]), np.mean(markers_ref[1][1, 20, :]), 0])
    Meta5 = np.array([np.mean(markers_ref[1][0, 25, :]), np.mean(markers_ref[1][1, 24, :]), 0])
    grf_flatfoot_ref = get_dispatch_flatfoot_contact_forces(
        grf_ref[1], M_CoP[1], [Meta1, Meta5, Heel], CoP[1], number_shooting_points[1]
    )
    grf_forefoot_ref = get_dispatch_forefoot_contact_forces(
        grf_ref[2], M_CoP[2], [Meta1, Meta5], CoP[2], number_shooting_points[2]
    )

    # Get initial isometric forces
    fiso_init = []
    n_muscle = 0
    for nGrp in range(biorbd_model[2].nbMuscleGroups()):
        for nMus in range(biorbd_model[2].muscleGroup(nGrp).nbMuscles()):
            fiso_init.append(biorbd_model[2].muscleGroup(nGrp).muscle(nMus).characteristics().forceIsoMax().to_mx())

    ocp = prepare_ocp(
        biorbd_model,
        final_time=phase_time,
        nb_shooting=number_shooting_points,
        markers_ref=markers_ref,
        excitation_ref=excitation_ref,
        grf_ref=(grf_ref[0], grf_flatfoot_ref, grf_forefoot_ref),
        q_ref=q_ref,
        qdot_ref=qdot_ref,
        fiso_init=fiso_init,
    )

    # ocp.add_plot("q", lambda x, u: q_ref[1], PlotType.STEP, axes_idx=[0, 1, 5, 8, 9, 11])
    # --- Solve the program --- #
    sol = ocp.solve(
        solver="ipopt",
        solver_options={
            "ipopt.tol": 1e-3,
            "ipopt.max_iter": 5000,
            "ipopt.hessian_approximation": "exact",
            "ipopt.limited_memory_max_history": 50,
            "ipopt.linear_solver": "ma57",
        },
        show_online_optim=False,
    )

    # --- Get Results --- #
    states_sol, controls_sol, params = Data.get_data(ocp, sol["x"], get_parameters=True)
    q = states_sol["q"]
    q_dot = states_sol["q_dot"]
    activations = states_sol["muscles"]
    tau = controls_sol["tau"]
    excitations = controls_sol["muscles"]

    # --- Save Results --- #
    np.save("./RES/stance_3phases/excitations", excitations)
    np.save("./RES/stance_3phases/activations", activations)
    np.save("./RES/stance_3phases/tau", tau)
    np.save("./RES/stance_3phases/q_dot", q_dot)
    np.save("./RES/stance_3phases/q", q)
    np.save("./RES/stance_3phases/params", params[ocp.nlp[0]["p"].name()])

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
    result.graphs()
