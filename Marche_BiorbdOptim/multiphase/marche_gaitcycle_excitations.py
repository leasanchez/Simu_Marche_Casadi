import numpy as np
from casadi import dot, Function, vertcat, MX
from matplotlib import pyplot as plt
import biorbd
from time import time
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
            biorbd_model.muscleGroup(nGrp).muscle(nMus).characteristics().setForceIsoMax(value[n_muscle] * fiso_init[n_muscle])
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
    biorbd_model, final_time, nb_shooting, markers_ref, excitation_ref, grf_ref, q_ref,qdot_ref, fiso_init,
):
    # Problem parameters
    nb_phases = len(biorbd_model)
    nb_q = biorbd_model[0].nbQ()
    nb_qdot = biorbd_model[0].nbQdot()
    nb_mus = biorbd_model[0].nbMuscleTotal()
    nb_tau = biorbd_model[0].nbGeneralizedTorque()

    torque_min, torque_max, torque_init = -10000, 10000, 0
    activation_min, activation_max, activation_init = 0, 1, 0.1

    # Add objective functions
    objective_functions = (
        (
            {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 1, "controls_idx":range(6, nb_q)},
            {"type": Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, "weight": 0.01, "data_to_track":excitation_ref[0].T},
            {"type": Objective.Lagrange.TRACK_MARKERS, "weight": 1000, "data_to_track": markers_ref[0]},
            # {"type": Objective.Lagrange.TRACK_STATE, "weight": 1, "states_idx": range(nb_q), "data_to_track": q_ref[0].T},
            {"type": Objective.Lagrange.TRACK_CONTACT_FORCES, "weight": 0.000005, "data_to_track": grf_ref[:, :-1].T},
            {"type": Objective.Mayer.CUSTOM, "function": get_last_contact_forces, "data_to_track":grf_ref.T, "weight": 0.000005, "instant": Instant.ALL}
        ),
        (
            {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 1, "controls_idx":range(6, nb_q)},
            {"type": Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, "weight": 0.01, "data_to_track":excitation_ref[1].T},
            {"type": Objective.Lagrange.TRACK_MARKERS, "weight": 1000, "data_to_track": markers_ref[1]},
            # {"type": Objective.Lagrange.TRACK_STATE, "weight": 1, "states_idx": range(nb_q), "data_to_track": q_ref[1].T},
        ),
    )

    # Dynamics
    problem_type = (
        ProblemType.muscle_excitations_and_torque_driven_with_contact,
        ProblemType.muscle_excitations_and_torque_driven,
    )

    # Constraints
    constraints = (
        ({"type": Constraint.CUSTOM, "function": get_muscles_first_node, "instant": Instant.START}, ),
        (),
    )
     # Impact
    # state_transitions = ({"type": StateTransition.IMPACT, "phase_pre_idx": 1,})

    # Path constraint
    X_bounds = []
    for i in range(nb_phases):
        XB = QAndQDotBounds(biorbd_model[i])
        XB.concatenate(
            Bounds([activation_min] * biorbd_model[i].nbMuscles(), [activation_max] * biorbd_model[i].nbMuscles())
        )
        X_bounds.append(XB)

    # Initial guess
    param_init = np.load('./RES/params.npy')
    q_init = np.load('./RES/q.npy')
    q_dot_init = np.load('./RES/qdot.npy')
    activations_init = np.load('./RES/activations.npy')
    excitations_init = np.load('./RES/excitations.npy')
    tau_init = np.load('./RES/tau.npy')

    X_init = []
    for n_p in range(nb_phases):
        init_x = np.zeros((nb_q + nb_qdot + nb_mus, nb_shooting[n_p] + 1))
        # init_x[[0, 1, 5, 8, 9, 11], :] = q_ref[n_p]
        if n_p == 0 :
            s = 0
            f = nb_shooting[n_p] + 1
        else:
            s = nb_shooting[n_p - 1]
            f = sum(nb_shooting) + 1

        init_x[:nb_q, :] = q_init[:, s:f]
        init_x[nb_q:nb_q + nb_qdot, :] = q_dot_init[:, s:f]
        init_x[-nb_mus:, :] = activations_init[:, s:f]
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
    for n_p in range(nb_phases):
        init_u = np.zeros((nb_tau + nb_mus, nb_shooting[n_p]))
        if n_p == 0 :
            s = 0
            f = nb_shooting[n_p]
        else:
            s = nb_shooting[n_p - 1]
            f = sum(nb_shooting)

        init_u[:nb_tau, :] = tau_init[:, s:f]
        init_u[-nb_mus:, ] = excitations_init[:, s:f]
        UI = InitialConditions(init_u, interpolation_type=InterpolationType.EACH_FRAME)
        U_init.append(UI)

    # Define the parameter to optimize
    bound_length = Bounds(min_bound=np.repeat(0.2, nb_mus), max_bound=np.repeat(5, nb_mus), interpolation_type=InterpolationType.CONSTANT)
    parameters = ({
        "name": "force_isometric",  # The name of the parameter
        "function": modify_isometric_force,  # The function that modifies the biorbd model
        "bounds": bound_length,  # The bounds
        "initial_guess": InitialConditions(param_init),  # The initial guess
        "size": nb_mus,  # The number of elements this particular parameter vector has
        "fiso_init" : fiso_init,
    },)

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
        parameters=(parameters, parameters),
        # state_transitions = state_transitions,
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
    Data_to_track = Data_to_track("equincocont01", multiple_contact=False)
    [T, T_stance, T_swing] = Data_to_track.GetTime()
    phase_time = [T_stance, T_swing]  # get time for each phase

    grf_ref = Data_to_track.load_data_GRF(
        biorbd_model[0], T_stance, number_shooting_points[0]
    )  # get ground reaction forces
    markers_ref = []
    markers_ref.append(Data_to_track.load_data_markers(biorbd_model[0], T_stance, number_shooting_points[0], "stance"))
    markers_ref.append(
        Data_to_track.load_data_markers(biorbd_model[1], phase_time[1], number_shooting_points[1], "swing")
    )  # get markers position

    q_ref = []
    q_ref.append(Data_to_track.load_data_q(model_q, T_stance, number_shooting_points[0], "stance"))
    q_ref.append(
        Data_to_track.load_data_q(model_q, phase_time[1], number_shooting_points[1], "swing")
    )  # get q from kalman
    q_ig, qdot_ig, activations_ig, tau_ig, excitations_ig = get_initial_value()

    emg_ref = []
    emg_ref.append(Data_to_track.load_data_emg(biorbd_model[0], T_stance, number_shooting_points[0], "stance"))
    emg_ref.append(
        Data_to_track.load_data_emg(biorbd_model[-1], phase_time[-1], number_shooting_points[-1], "swing")
    )  # get emg

    excitation_ref = []
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
        q_ref=q_ig,
        qdot_ref=qdot_ig,
        fiso_init=fiso_init,
    )

    # --- Solve the program --- #
    tic = time()
    sol = ocp.solve(
        solver="ipopt",
        options_ipopt={
            "ipopt.tol": 1e-2,
            "ipopt.max_iter": 5000,
            "ipopt.hessian_approximation": "limited-memory",
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
    np.save('./RES/equincocont01/excitations', excitations)
    np.save('./RES/equincocont01/activations', activations)
    np.save('./RES/equincocont01/tau', tau)
    np.save('./RES/equincocont01/q_dot', q_dot)
    np.save('./RES/equincocont01/q', q)
    np.save('./RES/equincocont01/params', params)


    ocp.save(sol, "marche_gait_equin_excitation")

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
    result.graphs()
