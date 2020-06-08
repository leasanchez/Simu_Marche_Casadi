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

def modify_isometric_force(biorbd_model, value):
    n_muscle = 0
    for nGrp in range(biorbd_model.nbMuscleGroups()):
        for nMus in range(biorbd_model.muscleGroup(nGrp).nbMuscles()):
            fiso_init = biorbd_model.muscleGroup(nGrp).muscle(nMus).characteristics().forceIsoMax().to_mx()
            biorbd_model.muscleGroup(nGrp).muscle(nMus).characteristics().setForceIsoMax(value[n_muscle] * fiso_init)
            n_muscle += 1

def get_initial_value():
    q_ig = []
    qdot_ig = []
    activations_ig = []
    tau_ig = []
    excitations_ig = []

    ocp_load_swing, sol_load_swing = OptimalControlProgram.load(
        "../swing/RES/excitations/3D/marche_swing_excitation.bo"
    )
    ocp_load_stance, sol_load_stance = OptimalControlProgram.load(
        "../stance/RES/equincocont01/excitations/3D/marche_stance_excitation.bo"
    )
    ocp_load = [ocp_load_stance, ocp_load_swing]
    sol_load = [sol_load_stance, sol_load_swing]

    for i in range(len(ocp_load)):
        states_sol, controls_sol = Data.get_data(ocp_load[i], sol_load[i]["x"])
        q_ig.append(states_sol["q"])
        qdot_ig.append(states_sol["q_dot"])
        activations_ig.append(states_sol["muscles"])
        tau_ig.append(controls_sol["tau"])
        excitations_ig.append(controls_sol["muscles"])
    return q_ig, qdot_ig, activations_ig, tau_ig, excitations_ig


def prepare_ocp(
    biorbd_model, final_time, nb_shooting, markers_ref, excitation_ref, grf_ref, q_ref,
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
    objective_functions = (
        (
            {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 1, "controls_idx":range(6, nb_q)},
            {"type": Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, "weight": 1, "data_to_track":excitation_ref[0].T},
            {"type": Objective.Lagrange.TRACK_MARKERS, "weight": 1000, "data_to_track": markers_ref[0]},
            # {"type": Objective.Lagrange.TRACK_STATE, "weight": 0.1, "states_idx": [0, 1, 5, 8, 9, 10], "data_to_track": q_ref[0].T},
            {"type": Objective.Lagrange.TRACK_CONTACT_FORCES, "weight": 0.00005, "data_to_track": grf_ref[:, :-1].T},
            {"type": Objective.Mayer.CUSTOM, "function": get_last_contact_forces, "data_to_track":grf_ref.T, "weight": 0.00005, "instant": Instant.ALL}
        ),
        (
            {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 1, "controls_idx":range(6, nb_q)},
            {"type": Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, "weight": 1, "data_to_track":excitation_ref[1].T},
            {"type": Objective.Lagrange.TRACK_MARKERS, "weight": 1000, "data_to_track": markers_ref[1]},
            # {"type": Objective.Lagrange.TRACK_STATE, "weight": 0.1, "states_idx": [0, 1, 5, 8, 9, 10], "data_to_track": q_ref[1].T},
        ),
    )

    # Dynamics
    problem_type = (
        ProblemType.muscle_excitations_and_torque_driven_with_contact,
        ProblemType.muscle_excitations_and_torque_driven,
    )

    # Constraints
    constraints = ()

    # Path constraint
    X_bounds = []
    for i in range(nb_phases):
        XB = QAndQDotBounds(biorbd_model[i])
        XB.concatenate(
            Bounds([activation_min] * biorbd_model[i].nbMuscles(), [activation_max] * biorbd_model[i].nbMuscles())
        )
        X_bounds.append(XB)

    # Initial guess
    # q_ig, qdot_ig, activations_ig, tau_ig, excitations_ig = get_initial_value()
    # X_init = []
    # for n_p in range(nb_phases):
    #     init_x = np.zeros(
    #         (
    #             biorbd_model[n_p].nbQ() + biorbd_model[n_p].nbQdot() + biorbd_model[n_p].nbMuscleTotal(),
    #             nb_shooting[n_p] + 1,
    #         )
    #     )
    #     for i in range(nb_shooting[n_p] + 1):
    #         init_x[:nb_q, i] = q_ig[n_p][:, i]
    #         init_x[nb_q : nb_q + nb_qdot, i] = qdot_ig[n_p][:, i]
    #         init_x[-nb_mus:, i] = activations_ig[n_p][:, i]
    #     XI = InitialConditions(init_x, interpolation_type=InterpolationType.EACH_FRAME)
    #     X_init.append(XI)

    X_init = []
    for n_p in range(nb_phases):
        init_x = np.zeros((biorbd_model[n_p].nbQ() + biorbd_model[n_p].nbQdot() + biorbd_model[n_p].nbMuscleTotal(), nb_shooting[n_p] + 1))
        for i in range(nb_shooting[n_p] + 1):
            init_x[[0, 1, 5, 8, 9, 10], i] = q_ref[n_p][:, i]
            init_x[-biorbd_model[n_p].nbMuscleTotal():, i] = excitation_ref[n_p][:, i]
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
    # U_init = []
    # for n_p in range(nb_phases):
    #     init_u = np.zeros(
    #         (biorbd_model[n_p].nbGeneralizedTorque() + biorbd_model[n_p].nbMuscleTotal(), nb_shooting[n_p])
    #     )
    #     for i in range(nb_shooting[n_p]):
    #         init_u[:nb_tau, i] = tau_ig[n_p][:, i]
    #         init_u[-nb_mus:, i] = excitations_ig[n_p][:, i]
    #     UI = InitialConditions(init_u, interpolation_type=InterpolationType.EACH_FRAME)
    #     U_init.append(UI)
    U_init = []
    for n_p in range(nb_phases):
        init_u = np.zeros((biorbd_model[n_p].nbGeneralizedTorque() + biorbd_model[n_p].nbMuscleTotal(), nb_shooting[n_p]))
        for i in range(nb_shooting[n_p]):
            if n_p == 0:
                init_u[1, i] = -500
            init_u[-biorbd_model[n_p].nbMuscleTotal():, i] = excitation_ref[n_p][:, i]
        UI = InitialConditions(init_u, interpolation_type=InterpolationType.EACH_FRAME)
        U_init.append(UI)

    # Define the parameter to optimize
    bound_length = Bounds(min_bound=np.repeat(0.2, nb_mus), max_bound=np.repeat(5, nb_mus), interpolation_type=InterpolationType.CONSTANT)
    parameters = ({
        "name": "force_isometric",  # The name of the parameter
        "function": modify_isometric_force,  # The function that modifies the biorbd model
        "bounds": bound_length,  # The bounds
        "initial_guess": InitialConditions(np.repeat(1, nb_mus)),  # The initial guess
        "size": nb_mus,  # The number of elements this particular parameter vector has
    })

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
    )


if __name__ == "__main__":
    # Define the problem
    # Model path
    biorbd_model = (
        biorbd.Model("../../ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact_deGroote_3d.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_0contact_deGroote_3d.bioMod"),
    )
    model_q = biorbd.Model("../../ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact_deGroote.bioMod")

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
        Data_to_track.load_data_markers(biorbd_model[-1], phase_time[-1], number_shooting_points[-1], "swing")
    )  # get markers position

    q_ref = []
    q_ref.append(Data_to_track.load_data_q(model_q, T_stance, number_shooting_points[0], "stance"))
    q_ref.append(
        Data_to_track.load_data_q(model_q, phase_time[-1], number_shooting_points[-1], "swing")
    )  # get q from kalman

    # symbolic_states = MX.sym("x", model_q.nbQ(), 1)
    # Compute_CoM = Function("ComputeCoM",
    #             [symbolic_states],
    #             [model_q.CoM(symbolic_states).to_mx()],
    #             ["q"],
    #             ["CoM"],
    #         ).expand()
    # CoM = np.zeros((3, number_shooting_points[0]))
    # for i in range(number_shooting_points[0]):
    #     CoM[:, i] = np.array(Compute_CoM(q_ref[0][:, i])).squeeze()

    emg_ref = []
    emg_ref.append(Data_to_track.load_data_emg(biorbd_model[0], T_stance, number_shooting_points[0], "stance"))
    emg_ref.append(
        Data_to_track.load_data_emg(biorbd_model[-1], phase_time[-1], number_shooting_points[-1], "swing")
    )  # get emg

    excitation_ref = []
    for i in range(len(phase_time)):
        excitation_ref.append(Data_to_track.load_muscularExcitation(emg_ref[i]))

    # Track these data
    biorbd_model = (
        biorbd.Model("../../ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact_deGroote_3d.bioMod"),
        biorbd.Model("../../ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact_deGroote_3d.bioMod"),
    )
    ocp = prepare_ocp(
        biorbd_model,
        phase_time,
        number_shooting_points,
        markers_ref=markers_ref,
        excitation_ref=excitation_ref,
        grf_ref=grf_ref,
        q_ref=q_ref,
    )

    # --- Add plot kalman --- #
    # q_ref = np.zeros((model_q.nbQ(), number_shooting_points[0] + number_shooting_points[1] + 1))
    # q_ref[:, :number_shooting_points[0]+1] = q_ref_stance
    # q_ref[:, number_shooting_points[0]:number_shooting_points[0] + number_shooting_points[1] + 1] = q_ref_swing
    # ocp.add_plot("q", lambda x, u: q_ref, PlotType.STEP, axes_idx=[0, 1, 5, 8, 9, 10])

    # --- Solve the program --- #
    tic = time()
    sol = ocp.solve(
        solver="ipopt",
        options_ipopt={
            "ipopt.tol": 1e-2,
            "ipopt.max_iter": 10000,
            "ipopt.hessian_approximation": "limited-memory",
            "ipopt.limited_memory_max_history": 50,
            "ipopt.linear_solver": "ma57",
        },
        show_online_optim=False,
    )
    toc = time() - tic
    print(f"Time to solve : {toc}sec")

    ocp.save(sol, "marche_gait_equin_excitation")

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
    result.graphs()
