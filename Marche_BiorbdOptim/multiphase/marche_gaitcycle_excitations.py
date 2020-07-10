import numpy as np
from casadi import dot, Function, vertcat, MX
from matplotlib import pyplot as plt
import biorbd
from time import time
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
)


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


def get_initial_value():
    q_ig = []
    qdot_ig = []
    activations_ig = []
    tau_ig = []
    excitations_ig = []

    file_swing = "../swing/RES/parametres/"
    file_stance = "../stance/RES/equincocont01/parametres/"
    file = (file_stance, file_swing)

    for i in range(2):
        path_file = file[i]
        q_ig.append(np.load(path_file + "q.npy"))
        qdot_ig.append(np.load(path_file + "q_dot.npy"))
        activations_ig.append(np.load(path_file + "activations.npy"))
        tau_ig.append(np.load(path_file + "tau.npy"))
        excitations_ig.append(np.load(path_file + "excitations.npy"))

    return q_ig, qdot_ig, activations_ig, tau_ig, excitations_ig


def prepare_ocp(
    biorbd_model, final_time, nb_shooting, markers_ref, excitation_ref, grf_ref, q_ref, qdot_ref, fiso_init,
):
    # Problem parameters
    nb_phases = len(biorbd_model)
    nb_q = biorbd_model[0].nbQ()
    nb_qdot = biorbd_model[0].nbQdot()
    nb_mus = biorbd_model[0].nbMuscleTotal()
    nb_tau = biorbd_model[0].nbGeneralizedTorque()

    torque_min, torque_max, torque_init = -1000, 1000, 0
    activation_min, activation_max, activation_init = 0, 1, 0.1

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Lagrange.TRACK_CONTACT_FORCES, weight=0.00005, data_to_track=grf_ref[:, :-1].T, phase=0)
    objective_functions.add(Objective.Mayer.CUSTOM, custom_function=get_last_contact_forces, instant=Instant.ALL, weight=0.00005, data_to_track=grf_ref.T, phase=0)
    for p in range(nb_phases):
        objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=1, controls_idx=range(6, nb_q), phase=p)
        objective_functions.add(Objective.Lagrange.TRACK_MUSCLES_CONTROL, weight=0.0001, data_to_track=excitation_ref[p].T, phase=p)
        objective_functions.add(Objective.Lagrange.TRACK_MARKERS, weight=500, data_to_track=markers_ref[0], phase=p)

    # Dynamics
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.MUSCLE_EXCITATIONS_AND_TORQUE_DRIVEN_WITH_CONTACT, phase=0)
    dynamics.add(DynamicsType.MUSCLE_EXCITATIONS_AND_TORQUE_DRIVEN, phase=1)

    # Constraints
    constraints = ConstraintList()
    constraints.add(Constraint.CUSTOM, custom_function=get_muscles_first_node, instant=Instant.START)

    # Impact
    # state_transitions = ({"type": StateTransition.IMPACT, "phase_pre_idx": 1,})

    # Path constraint
    x_bounds = BoundsList()
    for p in range(nb_phases):
        x_bounds.add(QAndQDotBounds(biorbd_model[p]))
        x_bounds[p].concatenate(
            Bounds([activation_min] * nb_mus, [activation_max] * nb_mus)
        )

    u_bounds = BoundsList()
    for p in range(nb_phases):
        u_bounds.add([
                [torque_min] * nb_tau + [activation_min] * nb_mus,
                [torque_max] * nb_tau + [activation_max] * nb_mus,
            ])

    # Initial guess
    param_init = np.load("./RES/equincocont01/params.npy")
    q_init = np.load("./RES/equincocont01/q.npy")
    q_dot_init = np.load("./RES/equincocont01/q_dot.npy")
    activations_init = np.load("./RES/equincocont01/activations.npy")
    excitations_init = np.load("./RES/equincocont01/excitations.npy")
    torque_init = np.load("./RES/equincocont01/tau.npy")

    x_init = InitialConditionsList()
    for p in range(nb_phases):
        init_x = np.zeros((nb_q + nb_qdot + nb_mus, nb_shooting[p] + 1))
        if p == 0:
            s = 0
            f = nb_shooting[p] + 1
        else:
            s = nb_shooting[p - 1]
            f = sum(nb_shooting) + 1

        init_x[:nb_q, :] = q_init[:, s:f]
        init_x[nb_q : nb_q + nb_qdot, :] = q_dot_init[:, s:f]
        init_x[-nb_mus:, :] = activations_init[:, s:f]
        x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)

    u_init = InitialConditionsList()
    for p in range(nb_phases):
        init_u = np.zeros((nb_tau + nb_mus, nb_shooting[p]))
        if p == 0:
            s = 0
            f = nb_shooting[p]
        else:
            s = nb_shooting[p - 1]
            f = sum(nb_shooting)

        init_u[:nb_tau, :] = torque_init[:, s:f]
        init_u[-nb_mus:,] = excitations_init[:, s:f]
        u_init.add(init_u, interpolation=InterpolationType.EACH_FRAME)

    # Define the parameter to optimize
    parameters = ParametersList()
    bound_length = Bounds(
        min_bound=np.repeat(0.2, nb_mus), max_bound=np.repeat(5, nb_mus), interpolation=InterpolationType.CONSTANT
    )
    for p in range(nb_phases):
        parameters.add(
            parameter_name="force_isometric",  # The name of the parameter
            function=modify_isometric_force,  # The function that modifies the biorbd model
            initial_guess=InitialConditions(param_init),  # The initial guess
            bounds=bound_length,  # The bounds
            size=nb_mus, # The number of elements this particular parameter vector has
            fiso_init=fiso_init,
       )

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        number_shooting_points,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        parameters=parameters,
    )


if __name__ == "__main__":
    # Define the problem
    # Model path
    biorbd_model = (
        biorbd.Model("../../ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact_deGroote_3d.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_0contact_deGroote_3d.bioMod"),
    )
    model_q = biorbd.Model("../../ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact.bioMod")

    # Problem parameters
    number_shooting_points = [25, 25]

    # Generate data from file
    markers_ref = []
    q_ref = []
    qdot_ref = []
    emg_ref = []
    excitation_ref = []  # init

    Data_to_track = Data_to_track("equincocont11", multiple_contact=False)
    [T, T_stance, T_swing] = Data_to_track.GetTime()
    phase_time = [T_stance, T_swing]

    grf_ref = Data_to_track.load_data_GRF(biorbd_model[0], T_stance, number_shooting_points[0])
    markers_ref.append(Data_to_track.load_data_markers(biorbd_model[0], T_stance, number_shooting_points[0], "stance"))
    markers_ref.append(
        Data_to_track.load_data_markers(biorbd_model[1], phase_time[1], number_shooting_points[1], "swing")
    )
    q_ref.append(Data_to_track.load_q_kalman(biorbd_model[0], T_stance, number_shooting_points[0], "stance"))
    q_ref.append(Data_to_track.load_q_kalman(biorbd_model[0], phase_time[1], number_shooting_points[1], "swing"))
    qdot_ref.append(Data_to_track.load_qdot_kalman(biorbd_model[0], T_stance, number_shooting_points[0], "stance"))
    qdot_ref.append(Data_to_track.load_qdot_kalman(biorbd_model[0], phase_time[1], number_shooting_points[1], "swing"))
    emg_ref.append(Data_to_track.load_data_emg(biorbd_model[0], T_stance, number_shooting_points[0], "stance"))
    emg_ref.append(Data_to_track.load_data_emg(biorbd_model[-1], phase_time[-1], number_shooting_points[-1], "swing"))
    for i in range(len(phase_time)):
        excitation_ref.append(Data_to_track.load_muscularExcitation(emg_ref[i]))

    # Get initial isometric forces
    fiso_init = []
    n_muscle = 0
    for nGrp in range(biorbd_model[0].nbMuscleGroups()):
        for nMus in range(biorbd_model[0].muscleGroup(nGrp).nbMuscles()):
            fiso_init.append(biorbd_model[0].muscleGroup(nGrp).muscle(nMus).characteristics().forceIsoMax().to_mx())

    # Track these data
    biorbd_model = (
        biorbd.Model("../../ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact_deGroote_3d.bioMod"),
        biorbd.Model("../../ModelesS2M/ANsWER_Rleg_6dof_17muscle_0contact_deGroote_3d.bioMod"),
    )
    ocp = prepare_ocp(
        biorbd_model,
        phase_time,
        number_shooting_points,
        markers_ref=markers_ref,
        excitation_ref=excitation_ref,
        grf_ref=grf_ref,
        q_ref=q_ref,
        qdot_ref=qdot_ref,
        fiso_init=fiso_init,
    )

    # --- Solve the program --- #
    tic = time()
    sol = ocp.solve(
        solver="ipopt",
        solver_options={
            "ipopt.tol": 1e-2,
            "ipopt.max_iter": 5000,
            "ipopt.hessian_approximation": "exact",
            "ipopt.limited_memory_max_history": 50,
            "ipopt.linear_solver": "ma57",
        },
        show_online_optim=False,
    )
    toc = time() - tic
    print(f"Time to solve : {toc}sec")

    # --- Get Results --- #
    states_sol, controls_sol, params_sol = Data.get_data(ocp, sol["x"], get_parameters=True)
    q = states_sol["q"]
    q_dot = states_sol["q_dot"]
    activations = states_sol["muscles"]
    tau = controls_sol["tau"]
    excitations = controls_sol["muscles"]
    params = params_sol[ocp.nlp[0]["p"].name()]

    # --- Save Results --- #
    np.save("./RES/equincocont11/excitations", excitations)
    np.save("./RES/equincocont11/activations", activations)
    np.save("./RES/equincocont11/tau", tau)
    np.save("./RES/equincocont11/q_dot", q_dot)
    np.save("./RES/equincocont11/q", q)
    np.save("./RES/equincocont11/params", params)

    ocp.save(sol, "marche_gait_equin_excitation")

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
    result.graphs()
