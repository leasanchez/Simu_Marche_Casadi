import numpy as np
from casadi import dot, Function, vertcat, MX
from matplotlib import pyplot as plt
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
    Data,
    InterpolationType,
)

def get_last_contact_forces(ocp, nlp, t, x, u, data_to_track=()):
    force = nlp["contact_forces_func"](x[-1], u[-1])
    val = force - data_to_track[t[-1], :]
    return dot(val, val)

def get_muscles_first_node(ocp, nlp, t, x, u):
    activation = x[0][2*nlp["nbQ"]:]
    excitation = u[0][nlp["nbQ"]:]
    val = activation - excitation
    return val

def get_qdot_post_impact(ocp, nlp, t, x, u):
    # Aliases
    nb_q = nlp["nbQ"]
    q = x[-1][:nb_q]
    qdot_pre = x[-1][nb_q:2*nb_q]
    qdot_post = nlp["model"].ComputeConstraintImpulsesDirect(q, qdot_pre)
    val = qdot_pre - qdot_post
    return val

def prepare_ocp(
    biorbd_model,
    final_time,
    nb_shooting,
    markers_ref,
    excitation_ref,
    grf_ref,
    q_ref,
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
    objective_functions =(
    (
        {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 1, "controls_idx": range(6, nb_tau)},
        {"type": Objective.Lagrange.TRACK_MUSCLES_CONTROL, "weight": 1, "data_to_track": excitation_ref[0][:, :-1].T},
        {"type": Objective.Lagrange.TRACK_MARKERS, "weight": 100, "data_to_track": markers_ref[0]},
        {"type": Objective.Lagrange.TRACK_CONTACT_FORCES, "weight": 0.00005, "data_to_track": grf_ref[0].T},
    ),
    (
        {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 1, "controls_idx": range(6, nb_tau)},
        {"type": Objective.Lagrange.TRACK_MUSCLES_CONTROL, "weight": 1, "data_to_track": excitation_ref[1][:, :-1].T},
        {"type": Objective.Lagrange.TRACK_MARKERS, "weight": 100, "data_to_track": markers_ref[1]},
        # {"type": Objective.Lagrange.TRACK_CONTACT_FORCES, "weight": 0.00005, "data_to_track": grf_ref.T},
    ),
    (
        {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 1, "controls_idx": range(6, nb_tau)},
        {"type": Objective.Lagrange.TRACK_MUSCLES_CONTROL, "weight": 1, "data_to_track": excitation_ref[2][:, :-1].T},
        {"type": Objective.Lagrange.TRACK_MARKERS, "weight": 100, "data_to_track": markers_ref[2]},
        {"type": Objective.Lagrange.TRACK_CONTACT_FORCES, "weight": 0.00005, "data_to_track": grf_ref[2].T},
        {"type": Objective.Mayer.CUSTOM, "weight": 0.00005, "function": get_last_contact_forces, "data_to_track": grf_ref[2].T, "instant": Instant.ALL}
    ),
    (
        {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 1, "controls_idx":range(6, nb_tau)},
        {"type": Objective.Lagrange.TRACK_MUSCLES_CONTROL, "weight": 1, "data_to_track":excitation_ref[3].T},
        {"type": Objective.Lagrange.TRACK_MARKERS, "weight": 100, "data_to_track": markers_ref[3]},
    ))

    # Dynamics
    problem_type = (
        ProblemType.muscle_excitations_and_torque_driven_with_contact,
        ProblemType.muscle_excitations_and_torque_driven_with_contact,
        ProblemType.muscle_excitations_and_torque_driven_with_contact,
        ProblemType.muscle_excitations_and_torque_driven,
    )

    # Constraints
    constraints = (
        ({"type": Constraint.CUSTOM, "function":get_qdot_post_impact, "instant": Instant.END},),
        (),
        (),
        ({"type": Constraint.CUSTOM, "function":get_qdot_post_impact, "instant": Instant.END},)
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
        min_bound = [torque_min] * biorbd_model[i].nbGeneralizedTorque() + [activation_min] * biorbd_model[i].nbMuscleTotal(),
        max_bound = [torque_max] * biorbd_model[i].nbGeneralizedTorque() + [activation_max] * biorbd_model[i].nbMuscleTotal(),
    )
        for i in range(nb_phases)]

    # Initial guess
    U_init = []
    for n_p in range(nb_phases):
        init_u = np.zeros((biorbd_model[n_p].nbGeneralizedTorque() + biorbd_model[n_p].nbMuscleTotal(), nb_shooting[n_p]))
        for i in range(nb_shooting[n_p]):
            if n_p != 3:
                init_u[1, i] = -500
            init_u[-biorbd_model[n_p].nbMuscleTotal():, i] = excitation_ref[n_p][:, i]
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
        X_bounds,
        U_bounds,
        objective_functions,
        constraints,
    )


if __name__ == "__main__":
    # Define the problem
    # Model path
    biorbd_model = (
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_1contact_deGroote_3d_Heel.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_2contacts_deGroote_3d.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_1contact_deGroote_3d_Forefoot.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_0contact_deGroote_3d.bioMod"),
    )

    # Problem parameters
    number_shooting_points = [5, 10, 15, 20]

    # Generate data from file
    from Marche_BiorbdOptim.LoadData import load_data_markers, load_data_q, load_data_emg, load_data_GRF, load_muscularExcitation

    name_subject = "normal01"
    grf_ref_stance, T, T_stance, T_swing = load_data_GRF(name_subject, biorbd_model,(number_shooting_points[0] + number_shooting_points[1] + number_shooting_points[2]))
    phase_time = [T*0.1, T*0.3, (T_stance - T*0.1 - T*0.3), T_swing]
    q_ref = []
    markers_ref = []
    excitation_ref = []
    grf_ref = []

    # phase stance
    t_stance, markers_ref_stance = load_data_markers(name_subject, biorbd_model[0], T_stance, (number_shooting_points[0] + number_shooting_points[1] + number_shooting_points[2]), 'stance')
    q_ref_stance = load_data_q(name_subject, biorbd_model[0], T_stance, (number_shooting_points[0] + number_shooting_points[1] + number_shooting_points[2]), 'stance')
    emg_ref_stance = load_data_emg(name_subject, biorbd_model[0], T_stance, (number_shooting_points[0] + number_shooting_points[1] + number_shooting_points[2]), 'stance')
    excitation_ref_stance = load_muscularExcitation(emg_ref_stance)

    markers_ref.append(markers_ref_stance[:, :, : number_shooting_points[0] + 1])
    excitation_ref.append(excitation_ref_stance[:, : number_shooting_points[0] + 1])
    q_ref.append(q_ref_stance[:, : number_shooting_points[0] + 1])
    grf_ref.append(grf_ref_stance[:, : number_shooting_points[0] + 1])
    for i in range(1, 3):
        markers_ref.append(markers_ref_stance[:, :, number_shooting_points[i-1] : number_shooting_points[i-1] + number_shooting_points[i] + 1])
        excitation_ref.append(excitation_ref_stance[:, number_shooting_points[i-1] : number_shooting_points[i-1] + number_shooting_points[i] + 1])
        q_ref.append(q_ref_stance[:, number_shooting_points[i-1] : number_shooting_points[i-1] + number_shooting_points[i] + 1])
        grf_ref.append(grf_ref_stance[:,number_shooting_points[i - 1]: number_shooting_points[i - 1] + number_shooting_points[i] + 1])

    # phase swing
    t_swing, markers_ref_swing = load_data_markers(name_subject, biorbd_model[-1], phase_time[-1], number_shooting_points[-1], 'swing')
    markers_ref.append(markers_ref_swing)
    q_ref.append(load_data_q(name_subject, biorbd_model[-1], phase_time[-1], number_shooting_points[-1], 'swing'))
    emg_ref = load_data_emg(name_subject, biorbd_model[-1], phase_time[-1], number_shooting_points[-1], 'swing')
    excitation_ref.append(load_muscularExcitation(emg_ref))

    # Track these data
    biorbd_model = (
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_1contact_deGroote_3d_Heel.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_2contacts_deGroote_3d.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_1contact_deGroote_3d_Forefoot.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_0contact_deGroote_3d.bioMod"),
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

    # --- Solve the program --- #
    sol = ocp.solve(solver="ipopt",
                    options_ipopt={
                        "ipopt.tol": 1e-3,
                        "ipopt.max_iter": 5000,
                        "ipopt.hessian_approximation": "exact",
                        "ipopt.limited_memory_max_history": 50,
                        "ipopt.linear_solver": "ma57",},
                    show_online_optim=False
                    )

    # --- Get Results --- #
    states_sol, controls_sol = Data.get_data(ocp, sol["x"])
    q = states_sol["q"]
    q_dot = states_sol["q_dot"]
    activations = states_sol["muscles"]
    tau = controls_sol["tau"]
    excitations = controls_sol["muscles"]

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
    result.graphs()
