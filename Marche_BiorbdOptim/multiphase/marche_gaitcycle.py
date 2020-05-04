import numpy as np
from casadi import dot, Function, vertcat
from matplotlib import pyplot as plt
import sys

sys.path.append('/home/leasanchez/programmation/BiorbdOptim')
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
    OdeSolver,
    Dynamics,
)

def get_last_contact_forces(ocp, nlp, t, x, u, data):
    CS_func = Function(
        "Contact_force",
        [ocp.symbolic_states, ocp.symbolic_controls],
        [
            Dynamics.forces_from_forward_dynamics_torque_muscle_driven_with_contact(
                ocp.symbolic_states, ocp.symbolic_controls, nlp
            )
        ],
        ["x", "u"],
        ["CS"],
    ).expand()
    force = CS_func(x[-1], u[-1])
    val = force - data[t[-1], :]
    return dot(val, val)


def prepare_ocp(
    biorbd_model,
    final_time,
    nb_shooting,
    markers_ref,
    activation_ref,
    grf_ref,
    show_online_optim,
):
    # Problem parameters
    nb_phases = len(biorbd_model)
    torque_min, torque_max, torque_init = -1000, 1000, 0
    activation_min, activation_max, activation_init = 0, 1, 0.1

    # Add objective functions
    objective_functions = ((
        {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 100, "controls_idx":[3, 4, 5]},
        {"type": Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, "weight": 1, "data_to_track":activation_ref[0].T},
        {"type": Objective.Lagrange.TRACK_MARKERS, "weight": 100, "data_to_track": markers_ref[0]},
        {"type": Objective.Lagrange.TRACK_CONTACT_FORCES, "weight": 0.05, "data_to_track": grf_ref[:, :-1].T},
        {"type": Objective.Mayer.CUSTOM, "function": get_last_contact_forces, "data":grf_ref.T, "weight": 0.05, "instant": Instant.ALL}
    ),
    (
        {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 1, "controls_idx":[3, 4, 5]},
        {"type": Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, "weight": 1, "data_to_track":activation_ref[1].T},
        {"type": Objective.Lagrange.TRACK_MARKERS, "weight": 100, "data_to_track": markers_ref[1]})
    )

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
    X_init = [InitialConditions([0] * (biorbd_model[i].nbQ() + biorbd_model[i].nbQdot()))for i in range(nb_phases)]

    # Define control path constraint
    U_bounds = [
        Bounds(
        min_bound = [torque_min] * biorbd_model[i].nbGeneralizedTorque() + [activation_min] * biorbd_model[i].nbMuscleTotal(),
        max_bound = [torque_max] * biorbd_model[i].nbGeneralizedTorque() + [activation_max] * biorbd_model[i].nbMuscleTotal(),
    )
        for i in range(nb_phases)]

    U_init = [InitialConditions(
        [torque_init] * biorbd_model[i].nbGeneralizedTorque() + [activation_init] * biorbd_model[i].nbMuscleTotal()
    ) for i in range(nb_phases)]

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        problem_type,
        nb_shooting,
        final_time,
        objective_functions,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        constraints,
        show_online_optim=show_online_optim,
    )


if __name__ == "__main__":
    # Define the problem
    # Model path
    biorbd_model = (
        biorbd.Model("ANsWER_Rleg_6dof_17muscle_1contact.bioMod"),
        biorbd.Model("ANsWER_Rleg_6dof_17muscle_0contact.bioMod"),
    )

    # Problem parameters
    number_shooting_points = [25, 25]

    # Generate data from file
    from Marche_BiorbdOptim.LoadData import load_data_markers, load_data_q, load_data_emg, load_data_GRF

    name_subject = "equincocont01"
    grf_ref, T, T_stance, T_swing = load_data_GRF(name_subject, biorbd_model, number_shooting_points[0])
    phase_time = [T_stance, T_swing]

    # phase stance
    t_stance, markers_ref_stance = load_data_markers(name_subject, biorbd_model[0], phase_time[0], number_shooting_points[0], 'stance')
    q_ref_stance = load_data_q(name_subject, biorbd_model[0], phase_time[0], number_shooting_points[0], 'stance')
    emg_ref_stance = load_data_emg(name_subject, biorbd_model[0], phase_time[0], number_shooting_points[0], 'stance')
    activation_ref_stance = np.zeros((biorbd_model[0].nbMuscleTotal(), number_shooting_points[0] + 1))
    idx_emg = 0
    for i in range(biorbd_model[0].nbMuscleTotal()):
        if (i!=1) and (i!=2) and (i!=3) and (i!=5) and (i!=6) and (i!=11) and (i!=12):
            activation_ref_stance[i, :] = emg_ref_stance[idx_emg, :]
            idx_emg += 1

    # phase swing
    t_swing, markers_ref_swing = load_data_markers(name_subject, biorbd_model[1], phase_time[1], number_shooting_points[1], 'swing')
    q_ref_swing = load_data_q(name_subject, biorbd_model[1], phase_time[1], number_shooting_points[1], 'swing')
    emg_ref_swing = load_data_emg(name_subject, biorbd_model[1], phase_time[1], number_shooting_points[1], 'swing')
    activation_ref_swing = np.zeros((biorbd_model[1].nbMuscleTotal(), number_shooting_points[1] + 1))
    idx_emg = 0
    for i in range(biorbd_model[0].nbMuscleTotal()):
        if (i!=1) and (i!=2) and (i!=3) and (i!=5) and (i!=6) and (i!=11) and (i!=12):
            activation_ref_swing[i, :] = emg_ref_swing[idx_emg, :]
            idx_emg += 1

    # Track these data
    biorbd_model = (
        biorbd.Model("ANsWER_Rleg_6dof_17muscle_1contact.bioMod"),
        biorbd.Model("ANsWER_Rleg_6dof_17muscle_0contact.bioMod"),
    )
    ocp = prepare_ocp(
        biorbd_model,
        phase_time,
        number_shooting_points,
        markers_ref = [markers_ref_stance, markers_ref_swing],
        activation_ref = [activation_ref_stance[:, :-1], activation_ref_swing[:, :-1]],
        grf_ref=grf_ref[1:, :],
        show_online_optim=False,
    )

    # --- Solve the program --- #
    sol = ocp.solve()

    # --- Compute ground reaction forces --- #
    contact_forces = np.zeros((2, sum([nlp["ns"] for nlp in ocp.nlp]) + 1))
    grf = np.zeros((2, sum([nlp["ns"] for nlp in ocp.nlp]) + 1))
    q_sol = np.zeros((biorbd_model[0].nbQ(), sum([nlp["ns"] for nlp in ocp.nlp]) + 1))
    qdot_sol = np.zeros((biorbd_model[0].nbQ(), sum([nlp["ns"] for nlp in ocp.nlp]) + 1))
    tau_sol = np.zeros((biorbd_model[0].nbQ(), sum([nlp["ns"] for nlp in ocp.nlp]) + 1))
    mus_sol = np.zeros((biorbd_model[0].nbMuscleTotal(), sum([nlp["ns"] for nlp in ocp.nlp]) + 1))

    q_ref = np.zeros((biorbd_model[0].nbQ(), sum([nlp["ns"] for nlp in ocp.nlp]) + 1))
    activation_ref = np.zeros((biorbd_model[0].nbMuscleTotal(), sum([nlp["ns"] for nlp in ocp.nlp]) + 1))

    CS_func = Function(
        "Contact_force",
        [ocp.symbolic_states, ocp.symbolic_controls],
        [ocp.nlp[0]["model"].getConstraints().getForce().to_mx()],
        ["x", "u"],
        ["CS"],
    ).expand()
    q, qdot, tau, mus = ProblemType.get_data_from_V(ocp, sol["x"], 0)
    x = vertcat(q, qdot)
    u = vertcat(tau, mus)
    contact_forces[:, : ocp.nlp[0]["ns"] + 1] = CS_func(x, u)

    for i, nlp in enumerate(ocp.nlp):
        q, q_dot, tau, mus = ProblemType.get_data_from_V(ocp, sol["x"], i)
        x = vertcat(q, q_dot)
        u = vertcat(mus, tau)
        if i == 0:
            grf[:, : nlp["ns"] + 1] = grf_ref[1:, :]
            q_sol[:, : nlp["ns"] + 1] = q
            qdot_sol[:, : nlp["ns"] + 1] = q_dot
            tau_sol[:, : nlp["ns"] + 1] = tau
            mus_sol[:, : nlp["ns"] + 1] = mus
            q_ref[:, : nlp["ns"] + 1] = q_ref_stance
            activation_ref[:, : nlp["ns"] + 1] = activation_ref_stance
        else:
            q_sol[:, ocp.nlp[i - 1]["ns"] : ocp.nlp[i - 1]["ns"] + nlp["ns"] + 1] = q
            qdot_sol[:, ocp.nlp[i - 1]["ns"]: ocp.nlp[i - 1]["ns"] + nlp["ns"] + 1] = q_dot
            tau_sol[:, ocp.nlp[i - 1]["ns"]: ocp.nlp[i - 1]["ns"] + nlp["ns"] + 1] = tau
            mus_sol[:, ocp.nlp[i - 1]["ns"]: ocp.nlp[i - 1]["ns"] + nlp["ns"]+1] = mus
            q_ref[:, ocp.nlp[i - 1]["ns"] : ocp.nlp[i - 1]["ns"] + nlp["ns"] + 1] = q_ref_swing
            activation_ref[:, ocp.nlp[i - 1]["ns"]: ocp.nlp[i - 1]["ns"] + nlp["ns"] + 1] = activation_ref_swing

    # --- Plot --- #
    t = np.hstack([t_stance[:-1], T_stance + t_swing])

    figure, axes = plt.subplots(3, biorbd_model[0].nbQ())
    axes = axes.flatten()
    for i in range(biorbd_model[0].nbQ()):
        axes[i].plot(t, q_sol[i, :])
        axes[i].plot(t, q_ref[i, :], 'r')
        axes[i + biorbd_model[0].nbQ()].plot(t, qdot_sol[i, :])
        axes[i + 2*biorbd_model[0].nbQ()].plot(t, tau_sol[i, :])
        axes[i].plot([T_stance, T_stance], [np.min(q_sol[i, :]), np.max(q_sol[i, :])], 'k--')

    figure2, axes2 = plt.subplots(4, 5)
    axes2 = axes2.flatten()
    for i in range(biorbd_model[0].nbMuscleTotal()):
        axes2[i].plot(t[:-1], activation_ref[i, :-1], 'r')
        axes2[i].plot(t[:-1], mus_sol[i, :-1])
        axes2[i].plot([T_stance, T_stance], [0, 1], 'k--')

    plt.figure('Contact forces')
    plt.plot(t, contact_forces.T, 'b')
    plt.plot(t, grf.T, 'r')
    plt.plot([T_stance, T_stance], [np.min(grf[1, :]), np.max(grf[1, :])], 'k--')

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate(show_meshes=False)
    # result.graphs()
