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
    Instant,
    ConstraintList,
    Constraint,
    StateTransitionList,
    StateTransition,
    Solver,
)

def heel_contact_forces_nul(ocp, nlp, t, x, u, p):
    force = nlp["contact_forces_func"](x[-1], u[-1], p)
    val = force[2]
    return dot(val, val)

def forefoot_contact_forces_nul(ocp, nlp, t, x, u, p):
    force = nlp["contact_forces_func"](x[-1], u[-1], p)
    val = force
    return dot(val, val)

def get_last_contact_forces_3contacts(ocp, nlp, t, x, u, p, grf):
    force = nlp["contact_forces_func"](x[-1], u[-1], p)
    val = grf[0, t[-1]] - (force[0] + force[4])
    val = vertcat(val, grf[1, t[-1]] - force[1])
    val = vertcat(val, grf[2, t[-1]] - (force[3] + force[5]))
    return dot(val, val)

def get_last_contact_forces_forefoot(ocp, nlp, t, x, u, p):
    force = nlp["contact_forces_func"](x[-1], u[-1], p)
    return dot(force, force)


def track_sum_contact_forces_3contacts(ocp, nlp, t, x, u, p, grf, target=()):
    nq = nlp["nbQ"]
    ns = nlp["ns"]
    val = []
    for n in range(ns):
        force = nlp["contact_forces_func"](x[n], u[n], p)
        val = vertcat(val, grf[0, t[n]] - (force[0] + force[4]))
        val = vertcat(val, grf[1, t[n]] - force[1])
        val = vertcat(val, grf[2, t[n]] - (force[2] + force[3] + force[5]))
    return dot(val, val)

def track_sum_contact_forces_forefoot(ocp, nlp, t, x, u, p, grf, target=()):
    nq = nlp["nbQ"]
    ns = nlp["ns"]
    val = []
    for n in range(ns):
        force = nlp["contact_forces_func"](x[n], u[n], p)
        val = vertcat(val, grf[0, t[n]] - (force[0] + force[2]))
        val = vertcat(val, grf[1, t[n]] - force[3])
        val = vertcat(val, grf[2, t[n]] - (force[1] + force[4]))
    return dot(val, val)


def prepare_ocp(
    biorbd_model, final_time, nb_shooting, markers_ref, grf_ref, q_ref, qdot_ref,
):

    # Problem parameters
    nb_phases = len(biorbd_model)
    nb_q = biorbd_model[0].nbQ()
    nb_qdot = biorbd_model[0].nbQdot()
    nb_tau = biorbd_model[0].nbGeneralizedTorque()

    torque_min, torque_max, torque_init = -1000, 1000, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    for p in range(nb_phases):
        # objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=0.001, controls_idx=range(6, nb_q), phase=p)
        objective_functions.add(Objective.Lagrange.TRACK_MARKERS, weight=800,  target=markers_ref[p], phase=p)
    # track grf
    # objective_functions.add(Objective.Lagrange.TRACK_CONTACT_FORCES,
    #                         weight=0.000001,
    #                         target=grf_ref[0],
    #                         phase=0) # just heel
    # objective_functions.add(track_sum_contact_forces_3contacts,
    #                         grf=grf_ref[1],
    #                         custom_type=Objective.Lagrange,
    #                         instant=Instant.ALL,
    #                         weight=0.000001,
    #                         phase=1) # 3 points
    # objective_functions.add(track_sum_contact_forces_forefoot,
    #                         grf=grf_ref[2],
    #                         custom_type=Objective.Lagrange,
    #                         instant=Instant.ALL,
    #                         weight=0.000001,
    #                         phase=2) # avant pied
    # objective_functions.add(get_last_contact_forces_forefoot,
    #                         custom_type=Objective.Mayer,
    #                         instant=Instant.ALL,
    #                         weight=0.000001,
    #                         phase=2) # 3 points

    # Dynamics
    dynamics = DynamicsTypeList()
    for p in range(nb_phases):
        dynamics.add(DynamicsType.TORQUE_DRIVEN_WITH_CONTACT, phase=p)

    # Constraints
    constraints = ConstraintList()
    constraints.add(
        Constraint.CONTACT_FORCE_INEQUALITY,
        direction="GREATER_THAN",
        instant=Instant.ALL,
        contact_force_idx=2,
        boundary=50,
        phase=0,
    )
    constraints.add(
        Constraint.CONTACT_FORCE_INEQUALITY,
        direction="GREATER_THAN",
        instant=Instant.ALL,
        contact_force_idx=(2, 3, 5),
        boundary=50,
        phase=1,
    )
    # constraints.add(
    #     Constraint.CONTACT_FORCE_INEQUALITY,
    #     direction="GREATER_THAN",
    #     instant=Instant.ALL,
    #     contact_force_idx=(1, 4),
    #     boundary=50,
    #     phase=2,
    # )


    # State Transitions
    state_transitions = StateTransitionList()
    state_transitions.add(StateTransition.IMPACT, phase_pre_idx=0)

    # Path constraint
    x_bounds = BoundsList()
    u_bounds = BoundsList()
    for p in range(nb_phases):
        x_bounds.add(QAndQDotBounds(biorbd_model[p]))
        u_bounds.add([
            [torque_min] * nb_tau,
            [torque_max] * nb_tau,
        ])

    # Initial guess
    x_init = InitialConditionsList()
    for p in range(nb_phases):
        init_x = np.zeros((nb_q + nb_qdot, nb_shooting[p] + 1))
        init_x[:nb_q, :] = q_ref[p]
        init_x[nb_q:nb_q + nb_qdot, :] = qdot_ref[p]
        x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)

    u_init = InitialConditionsList()
    for p in range(nb_phases):
        init_u = np.zeros((nb_tau, number_shooting_points[p]))
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
        state_transitions=state_transitions,
    )


if __name__ == "__main__":
    # Define the problem -- model path
    biorbd_model = (
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_1contact_deGroote_3d_Heel.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_3contacts_deGroote_3d.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_contact_deGroote_3d_Forefoot.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_0contact_deGroote_3d.bioMod")
    )

    # Problem parameters
    dt = 0.01
    nb_q = biorbd_model[0].nbQ()
    nb_qdot = biorbd_model[0].nbQdot()
    nb_tau = biorbd_model[0].nbGeneralizedTorque()

    # Generate data from file
    Data_to_track = Data_to_track("normal01", model=biorbd_model[0], multiple_contact=True)
    phase_time = Data_to_track.GetTime()
    number_shooting_points = []
    for time in phase_time:
        number_shooting_points.append(int(time/0.01))
    grf_ref = Data_to_track.load_data_GRF(number_shooting_points)  # get ground reaction forces
    markers_ref = Data_to_track.load_data_markers(number_shooting_points)
    q_ref = Data_to_track.load_q_kalman(number_shooting_points)
    qdot_ref = Data_to_track.load_qdot_kalman(number_shooting_points)

    biorbd_model = (
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_1contact_deGroote_3d_Heel.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_3contacts_deGroote_3d.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_contact_deGroote_3d_Forefoot.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_0contact_deGroote_3d.bioMod")
    )

    ocp = prepare_ocp(
        biorbd_model=(biorbd_model[0], biorbd_model[1]),
        final_time=(phase_time[0], phase_time[1]),
        nb_shooting=(number_shooting_points[0], number_shooting_points[1]),
        markers_ref=(markers_ref[0], markers_ref[1]),
        grf_ref=(grf_ref[0], grf_ref[1]),
        q_ref=(q_ref[0], q_ref[1]),
        qdot_ref=(qdot_ref[0], qdot_ref[1]),
    )

    # --- Solve the program --- #
    sol = ocp.solve(
        solver=Solver.IPOPT,
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
    states_sol, controls_sol = Data.get_data(ocp, sol["x"])
    q = states_sol["q"]
    q_dot = states_sol["q_dot"]
    tau = controls_sol["tau"]

    # --- Time vector --- #
    nb_phases = len(ocp.nlp)
    t = np.linspace(0, phase_time[0], number_shooting_points[0] + 1)
    for p in range(1, nb_phases):
        t=np.concatenate((t[:-1], t[-1] + np.linspace(0, phase_time[p], number_shooting_points[p] + 1)))

    # --- Plot q --- #
    q_name=[]
    for s in range(biorbd_model[0].nbSegment()):
        seg_name = biorbd_model[0].segment(s).name().to_string()
        for d in range(biorbd_model[0].segment(s).nbDof()):
            dof_name = biorbd_model[0].segment(s).nameDof(d).to_string()
            q_name.append(seg_name + "_" + dof_name)

    figure, axes = plt.subplots(3, 4)
    axes = axes.flatten()
    for i in range(nb_q):
        axes[i].plot(t, q[i, :], 'r-')
        q_plot = q_ref[0][i, :]
        for p in range(1, nb_phases):
            q_plot = np.concatenate([q_plot[:-1], q_ref[p][i, :]])
        axes[i].plot(t, q_plot, 'k--')
        axes[i].set_title(q_name[i])

    # --- plot grf ---
    nb_shooting = number_shooting_points[0] + number_shooting_points[1]
    forces_sim = np.zeros((9, nb_shooting + 1))
    for n in range(number_shooting_points[0] + 1):
        forces_sim[:3, n:n+1] = ocp.nlp[0]['contact_forces_func'](np.concatenate([q[:, n], q_dot[:, n]]), tau[:, n], 0)
    n_f= number_shooting_points[0]
    for n in range(number_shooting_points[1] + 1):
        forces_sim[[0,1,2,5,6,8], n_f + n:n_f + n + 1] = ocp.nlp[1]['contact_forces_func'](
            np.concatenate([q[:, n_f + n], q_dot[:, n_f + n]]),
            tau[:, n_f + n],
            0)
    n_f +=number_shooting_points[1]
    # for n in range(number_shooting_points[2] + 1):
    #     forces_sim[[3,5,6,7,8], n_f + n:n_f + n + 1] = ocp.nlp[2]['contact_forces_func'](
    #         np.concatenate([q[:, n_f + n], q_dot[:, n_f + n]]),
    #         tau[:, n_f + n],
    #         0)

    figure, axes = plt.subplots(3, 3)
    axes = axes.flatten()
    coord_label = ['x', 'y', "z"]
    contact_point_label = ['Heel', 'Meta1', "Meta5"]
    for c in range(3):
        for coord in range(3):
            title = (contact_point_label[c] + "_"+coord_label[coord])
            axes[3*c + coord].plot(t, forces_sim[3*c + coord, :])
            axes[3*c + coord].plot([phase_time[0], phase_time[0]],
                                   [np.min(forces_sim[3*c + coord, :]),
                                    np.max(forces_sim[3*c + coord, :])],
                                   'k--')
            axes[3 * c + coord].plot([phase_time[0] + phase_time[1], phase_time[0] + phase_time[1]],
                                     [np.min(forces_sim[3 * c + coord, :]),
                                      np.max(forces_sim[3 * c + coord, :])],
                                     'k--')
            # axes[3 * c + coord].plot([phase_time[0] + phase_time[1] + phase_time[2], phase_time[0] + phase_time[1] + phase_time[2]],
            #                          [np.min(forces_sim[3 * c + coord, :]),
            #                           np.max(forces_sim[3 * c + coord, :])],
            #                          'k--')
            axes[3*c + coord].set_title(title)


    figure, axes = plt.subplots(1, 3)
    axes = axes.flatten()
    force_plot_x = grf_ref[0][0, :]
    force_plot_y = grf_ref[0][1, :]
    force_plot_z = grf_ref[0][2, :]
    for p in range(1, nb_phases):
        force_plot_x = np.concatenate([force_plot_x[:-1], grf_ref[p][0, :]])
        force_plot_y = np.concatenate([force_plot_y[:-1], grf_ref[p][1, :]])
        force_plot_z = np.concatenate([force_plot_z[:-1], grf_ref[p][2, :]])

    axes[0].plot(t, forces_sim[0, :] + forces_sim[3, :] + forces_sim[6, :], 'r-')
    axes[0].plot(t, force_plot_x, 'b--')
    axes[0].plot([phase_time[0], phase_time[0]],
                             [np.min(forces_sim[0, :] + forces_sim[3, :] + forces_sim[6, :]),
                              np.max(forces_sim[0, :] + forces_sim[3, :] + forces_sim[6, :])],
                             'k--')
    axes[0].plot([phase_time[0] + phase_time[1], phase_time[0] + phase_time[1]],
                             [np.min(forces_sim[0, :] + forces_sim[3, :] + forces_sim[6, :]),
                              np.max(forces_sim[0, :] + forces_sim[3, :] + forces_sim[6, :])],
                             'k--')
    # axes[0].plot(
    #     [phase_time[0] + phase_time[1] + phase_time[2], phase_time[0] + phase_time[1] + phase_time[2]],
    #     [np.min(forces_sim[0, :] + forces_sim[3, :] + forces_sim[6, :]),
    #      np.max(forces_sim[0, :] + forces_sim[3, :] + forces_sim[6, :])],
    #     'k--')
    axes[0].set_title('forces in x (N)')

    axes[1].plot(t, forces_sim[1, :] + forces_sim[4, :] + forces_sim[7, :])
    axes[1].plot(t, force_plot_y, 'k--')
    axes[1].plot([phase_time[0], phase_time[0]],
                             [np.min(forces_sim[1, :] + forces_sim[4, :] + forces_sim[7, :]),
                              np.max(forces_sim[1, :] + forces_sim[4, :] + forces_sim[7, :])],
                             'k--')
    axes[1].plot([phase_time[0] + phase_time[1], phase_time[0] + phase_time[1]],
                             [np.min(forces_sim[1, :] + forces_sim[4, :] + forces_sim[7, :]),
                              np.max(forces_sim[1, :] + forces_sim[4, :] + forces_sim[7, :])],
                             'k--')
    # axes[1].plot(
    #     [phase_time[0] + phase_time[1] + phase_time[2], phase_time[0] + phase_time[1] + phase_time[2]],
    #     [np.min(forces_sim[1, :] + forces_sim[4, :] + forces_sim[7, :]),
    #      np.max(forces_sim[1, :] + forces_sim[4, :] + forces_sim[7, :])],
    #     'k--')
    axes[1].set_title('forces in y (N)')

    axes[2].plot(t, forces_sim[2, :] + forces_sim[5, :] + forces_sim[8, :])
    axes[2].plot(t, force_plot_z, 'k--')
    axes[2].plot([phase_time[0], phase_time[0]],
                             [np.min(forces_sim[2, :] + forces_sim[5, :] + forces_sim[8, :]),
                              np.max(forces_sim[2, :] + forces_sim[5, :] + forces_sim[8, :])],
                             'k--')
    axes[2].plot([phase_time[0] + phase_time[1], phase_time[0] + phase_time[1]],
                             [np.min(forces_sim[2, :] + forces_sim[5, :] + forces_sim[8, :]),
                              np.max(forces_sim[2, :] + forces_sim[5, :] + forces_sim[8, :])],
                             'k--')
    # axes[2].plot(
    #     [phase_time[0] + phase_time[1] + phase_time[2], phase_time[0] + phase_time[1] + phase_time[2]],
    #     [np.min(forces_sim[2, :] + forces_sim[5, :] + forces_sim[8, :]),
    #      np.max(forces_sim[2, :] + forces_sim[5, :] + forces_sim[8, :])],
    #     'k--')
    axes[2].set_title('forces in z (N)')


    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
    result.graphs()
