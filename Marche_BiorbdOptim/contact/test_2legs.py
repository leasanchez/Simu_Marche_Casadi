import numpy as np
from casadi import dot, Function, vertcat, MX, mtimes, nlpsol
import biorbd
import BiorbdViz as BiorbdViz
from matplotlib import pyplot as plt
from Marche_BiorbdOptim.LoadData import Data_to_track

from bioptim import (
    OptimalControlProgram,
    DynamicsTypeList,
    DynamicsType,
    BoundsList,
    Bounds,
    QAndQDotBounds,
    InitialGuessList,
    InitialGuess,
    ShowResult,
    ObjectiveList,
    Objective,
    InterpolationType,
    Data,
    Instant,
    ConstraintList,
    Constraint,
    Solver,
    StateTransitionList,
    StateTransition,
)

# --- force nul at last point ---
def get_last_contact_force_null(ocp, nlp, t, x, u, p, contact_name):
    force = nlp.contact_forces_func(x[-1], u[-1], p)
    if contact_name == 'all':
        val = force
    else:
        cn = nlp.model.contactNames()
        val = []
        for i, c in enumerate(cn):
            for name in contact_name:
                if c.to_string() == name:
                    val = vertcat(val, force[i])
    return val

# --- flatfoot R ---
def track_sum_contact_forces_flatfootR(ocp, nlp, t, x, u, p, grf, target=()):
    ns = nlp.ns
    val = []
    for n in range(ns):
        force = nlp.contact_forces_func(x[n], u[n], p)
        val = vertcat(val, grf[0][0, t[n]] - (force[0] + force[4]))
        val = vertcat(val, grf[0][1, t[n]] - force[1])
        val = vertcat(val, grf[0][2, t[n]] - (force[2] + force[3] + force[5]))
    return dot(val, val)

def track_sum_contact_forces_flatfootR_HeelL(ocp, nlp, t, x, u, p, grf, target=()):
    ns = nlp.ns
    val = []
    for n in range(ns):
        force = nlp.contact_forces_func(x[n], u[n], p)
        val = vertcat(val, grf[0][0, t[n]] - (force[0] + force[4]))
        val = vertcat(val, grf[0][1, t[n]] - force[1])
        val = vertcat(val, grf[0][2, t[n]] - (force[2] + force[3] + force[5]))
        val = vertcat(val, grf[1][0, t[n]] - force[6])
        val = vertcat(val, grf[1][1, t[n]] - force[7])
        val = vertcat(val, grf[1][2, t[n]] - force[8])
    return dot(val, val)

def prepare_ocp(
    biorbd_model, final_time, nb_shooting, markers_ref, grf_ref, q_ref, qdot_ref, nb_threads,
):
    # Problem parameters
    nb_q = biorbd_model[0].nbQ()
    nb_qdot = biorbd_model[0].nbQdot()
    nb_tau = biorbd_model[0].nbGeneralizedTorque()
    nb_phases = len(biorbd_model)

    torque_min, torque_max, torque_init = -1000, 1000, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    for p in range(nb_phases):
        objective_functions.add(Objective.Lagrange.TRACK_STATE, weight=200, states_idx=range(nb_q), target=q_ref[p], phase=p)
    objective_functions.add(track_sum_contact_forces_flatfootR,
                            grf=grf_ref[0],
                            custom_type=Objective.Lagrange,
                            instant=Instant.ALL,
                            weight=0.00001,
                            phase=0)
    objective_functions.add(track_sum_contact_forces_flatfootR_HeelL,
                            grf=grf_ref[1],
                            custom_type=Objective.Lagrange,
                            instant=Instant.ALL,
                            weight=0.00001,
                            phase=1)
    # objective_functions.add(get_last_contact_forces, custom_type=Objective.Mayer, instant=Instant.ALL, grf=grf_ref, weight=0.00001)

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
        contact_force_idx=(2, 3, 5),
        boundary=0,
        phase=0
    )
    constraints.add(
        Constraint.CONTACT_FORCE_INEQUALITY,
        direction="GREATER_THAN",
        instant=Instant.ALL,
        contact_force_idx=(2, 3, 5, 8),
        boundary=0,
        phase=1,
    )

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
    x_init = InitialGuessList()
    for p in range(nb_phases):
        init_x = np.zeros((nb_q + nb_qdot, nb_shooting[p] + 1))
        init_x[:nb_q, :] = q_ref[p]
        init_x[nb_q:nb_q + nb_qdot, :] = qdot_ref[p]
        x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)

    u_init = InitialGuessList()
    for p in range(nb_phases):
        init_u = np.zeros((nb_tau, number_shooting_points[p]))
        u_init.add(init_u, interpolation=InterpolationType.EACH_FRAME)

    state_transitions = StateTransitionList()
    state_transitions.add(StateTransition.IMPACT, phase_pre_idx=0)
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
    # Define the problem --- Model path
    model = biorbd.Model("../../ModelesS2M/Marche_saine/2legs/2legs_18dof_0contact.bioMod")
    biorbd_model = (
        biorbd.Model("../../ModelesS2M/Marche_saine/2legs/2legs_18dof_flatfootR.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/2legs/2legs_18dof_flatfootR_HeelL.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/2legs/2legs_18dof_forefootR_HeelL.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/2legs/2legs_18dof_forefootR_flatfootL.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/2legs/2legs_18dof_flatfootL.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/2legs/2legs_18dof_HeelR_flatfootL.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/2legs/2legs_18dof_HeelR_forefootL.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/2legs/2legs_18dof_flatfootR_forefootL.bioMod"),
    )

    # Problem parameters
    nb_q = model.nbQ()
    nb_qdot = model.nbQdot()
    nb_tau = model.nbGeneralizedTorque()
    nb_phases_init = len(biorbd_model)

    # Generate data from file
    Data_to_track = Data_to_track("normal02", model=model, multiple_contact=True, two_leg=True)
    phase_time = Data_to_track.GetTime()
    number_shooting_points = []
    for time in phase_time:
        number_shooting_points.append(int(time / 0.01))
    markers_ref = Data_to_track.load_data_markers(number_shooting_points) # get markers positions
    q_ref = Data_to_track.load_q_kalman(number_shooting_points) # get joint positions
    qdot_ref = Data_to_track.load_qdot_kalman(number_shooting_points) # get joint velocities
    qddot_ref = Data_to_track.load_qdot_kalman(number_shooting_points) # get joint accelerations
    grf_ref = Data_to_track.load_data_GRF(number_shooting_points)  # get ground reaction forces

    q_simu = np.zeros((nb_q, sum(number_shooting_points) + 1))
    n_shoot = 0
    for p in range(nb_phases_init):
        q_simu[:, n_shoot:n_shoot + number_shooting_points[p] + 1] = q_ref[p]
        n_shoot += number_shooting_points[p]

    # b = BiorbdViz.BiorbdViz(loaded_model=model)
    # b.load_movement(q_simu)
    # b.exec()

    ocp = prepare_ocp(
        biorbd_model=(biorbd_model[0], biorbd_model[1]),
        final_time=(phase_time[0], phase_time[1]),
        nb_shooting=(number_shooting_points[0], number_shooting_points[1]),
        markers_ref=(markers_ref[0], markers_ref[1]),
        grf_ref=(grf_ref[0], grf_ref[1]),
        q_ref=(q_ref[0], q_ref[1]),
        qdot_ref=(qdot_ref[0], qdot_ref[1]),
        nb_threads=1,
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
    states, controls = Data.get_data(ocp, sol)
    q, q_dot, tau = (
        states["q"],
        states["q_dot"],
        controls["tau"],
    )

    # --- Time vector --- #
    nb_phases = len(ocp.nlp)
    t = np.linspace(0, phase_time[0], number_shooting_points[0] + 1)
    for p in range(1, nb_phases):
        t = np.concatenate((t[:-1], t[-1] + np.linspace(0, phase_time[p], number_shooting_points[p] + 1)))

    # --- Plot q --- #
    q_name = []
    for s in range(biorbd_model[0].nbSegment()):
        seg_name = biorbd_model[0].segment(s).name().to_string()
        for d in range(biorbd_model[0].segment(s).nbDof()):
            dof_name = biorbd_model[0].segment(s).nameDof(d).to_string()
            q_name.append(seg_name + "_" + dof_name)

    figure, axes = plt.subplots(3, 6)
    axes = axes.flatten()
    for i in range(nb_q):
        axes[i].plot(t, q[i, :], 'r-')
        q_plot = q_ref[0][i, :]
        for p in range(1, nb_phases):
            q_plot = np.concatenate([q_plot[:-1], q_ref[p][i, :]])
        axes[i].plot(t, q_plot, 'b--')
        axes[i].plot([phase_time[0], phase_time[0]], [np.min(q_plot), np.max(q_plot)], 'k--')
        axes[i].text(phase_time[0] - 0.02, np.min(q_plot), 'LHS')
        axes[i].plot([phase_time[0] + phase_time[1], phase_time[0] + phase_time[1]], [np.min(q_plot), np.max(q_plot)], 'k--')
        axes[i].text(phase_time[0] + phase_time[1] - 0.02, np.min(q_plot), 'RHR')
        axes[i].set_title(q_name[i])
    plt.legend(['simulated', 'reference'])

    # --- plot grf ---
    nb_shooting = number_shooting_points[0] + number_shooting_points[1] + number_shooting_points[2]
    labels_forces = ['Heel_r_X', 'Heel_r_Y', 'Heel_r_Z',
                     'Meta_1_r_X', 'Meta_1_r_Y', 'Meta_1_r_Z',
                     'Meta_5_r_X', 'Meta_5_r_Y', 'Meta_5_r_Z',
                     'Heel_l_X', 'Heel_l_Y', 'Heel_l_Z',
                     'Meta_1_l_X', 'Meta_1_l_Y', 'Meta_1_l_Z',
                     'Meta_5_l_X', 'Meta_5_l_Y', 'Meta_5_l_Z'
                     ]
    forces = {}
    for label in labels_forces:
        forces[label] = np.zeros(nb_shooting + 1)

    n_shoot = 0
    for p in range(nb_phases):
        cn = biorbd_model[p].contactNames()
        for n in range(number_shooting_points[p] + 1):
            forces_sim = ocp.nlp[p]['contact_forces_func'](np.concatenate([q[:, n_shoot + n], q_dot[:, n_shoot + n]]), tau[:, n_shoot + n], 0)
            for i, c in enumerate(cn):
                if c.to_string() in forces:
                    forces[c.to_string()][n_shoot + n] = forces_sim[i]
        n_shoot= number_shooting_points[p]

    figure, axes = plt.subplots(6, 3)
    axes = axes.flatten()
    for i, f in enumerate(forces):
        axes[i].plot(t, forces[f], 'r-')
        axes[i].set_title(f)
        pt = phase_time[0]
        for p in range(nb_phases):
            axes[i].plot([pt, pt], [np.min(forces[f]), np.max(forces[f])], 'k--')
            pt+=phase_time[p+1]

    figure, axes = plt.subplots(2, 3)
    axes = axes.flatten()
    force_plot_x_R = grf_ref[0][0][0, :]
    force_plot_y_R = grf_ref[0][0][1, :]
    force_plot_z_R = grf_ref[0][0][2, :]
    force_plot_x_L = grf_ref[0][1][0, :]
    force_plot_y_L = grf_ref[0][1][1, :]
    force_plot_z_L = grf_ref[0][1][2, :]
    for p in range(1, nb_phases):
        force_plot_x_R = np.concatenate([force_plot_x_R[:-1], grf_ref[p][0][0, :]])
        force_plot_y_R = np.concatenate([force_plot_y_R[:-1], grf_ref[p][0][1, :]])
        force_plot_z_R = np.concatenate([force_plot_z_R[:-1], grf_ref[p][0][2, :]])
        force_plot_x_L = np.concatenate([force_plot_x_L[:-1], grf_ref[p][1][0, :]])
        force_plot_y_L = np.concatenate([force_plot_y_L[:-1], grf_ref[p][1][1, :]])
        force_plot_z_L = np.concatenate([force_plot_z_L[:-1], grf_ref[p][1][2, :]])

    FR = [force_plot_x_R, force_plot_y_R, force_plot_z_R]
    FL = [force_plot_x_L, force_plot_y_L, force_plot_z_L]
    coord_label = ['X', 'Y', 'Z']
    for i in range(3):
        axes[i].plot(t, forces[f"Heel_r_{coord_label[i]}"] + forces[f"Meta_1_r_{coord_label[i]}"] + forces[f"Meta_5_r_{coord_label[i]}"], 'r-')
        axes[i].plot(t, FR[i], 'b--')
        axes[i].set_title("Forces in " + coord_label[i] + " R")

        axes[i + 3].plot(t, forces[f"Heel_l_{coord_label[i]}"] + forces[f"Meta_1_l_{coord_label[i]}"] + forces[f"Meta_5_l_{coord_label[i]}"], 'r-')
        axes[i + 3].plot(t, FL[i], 'k--')
        axes[i + 3].set_title("Forces in " + coord_label[i] + " L")

        pt = phase_time[0]
        for p in range(nb_phases):
            axes[i].plot([pt, pt], [np.min(FR[i]), np.max(FR[i])], 'k--')
            axes[i + 3].plot([pt, pt], [np.min(FL[i]), np.max(FL[i])], 'k--')
            pt+=phase_time[p+1]


    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
    result.graphs()