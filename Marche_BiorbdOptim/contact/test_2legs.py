import numpy as np
from casadi import dot, Function, vertcat, MX, mtimes, nlpsol
import biorbd
import BiorbdViz as BiorbdViz
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
    Solver,
    StateTransitionList,
    StateTransition,
)

def track_sum_contact_forces_flatfootR(ocp, nlp, t, x, u, p, grf, target=()):
    ns = nlp["ns"]
    val = []
    for n in range(ns):
        force = nlp["contact_forces_func"](x[n], u[n], p)
        val = vertcat(val, grf[0][0, t[n]] - (force[0] + force[4]))
        val = vertcat(val, grf[0][1, t[n]] - force[1])
        val = vertcat(val, grf[0][2, t[n]] - (force[2] + force[3] + force[5]))
    return dot(val, val)

def track_sum_contact_forces_flatfootR_HeelL(ocp, nlp, t, x, u, p, grf, target=()):
    ns = nlp["ns"]
    val = []
    for n in range(ns):
        force = nlp["contact_forces_func"](x[n], u[n], p)
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
        boundary=50,
        phase=0
    )
    constraints.add(
        Constraint.CONTACT_FORCE_INEQUALITY,
        direction="GREATER_THAN",
        instant=Instant.ALL,
        contact_force_idx=(2, 3, 5, 8),
        boundary=50,
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

    state_transitions = StateTransitionList()
    state_transitions.add(StateTransition.IMPACT, phase_pre_idx=0)
    # ------------- #

    return OptimalControlProgram(
        biorbd_model=biorbd_model,
        dynamics_type=dynamics,
        number_shooting_points=nb_shooting,
        phase_time=final_time,
        X_init=x_init,
        U_init=u_init,
        X_bounds=x_bounds,
        U_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        nb_threads=nb_threads,
    )

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

ocp = prepare_ocp(
    biorbd_model=(biorbd_model[0], biorbd_model[1], biorbd_model[2]),
    final_time=(phase_time[0], phase_time[1], phase_time[2]),
    nb_shooting=(number_shooting_points[0], number_shooting_points[1], number_shooting_points[2]),
    markers_ref=(markers_ref[0], markers_ref[1], markers_ref[2]),
    grf_ref=(grf_ref[0], grf_ref[1], grf_ref[2]),
    q_ref=(q_ref[0], q_ref[1], q_ref[2]),
    qdot_ref=(qdot_ref[0], qdot_ref[1], qdot_ref[2]),
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
    axes[i].plot(t, q_plot, 'k--')
    axes[i].set_title(q_name[i])

# --- plot grf ---
nb_shooting = number_shooting_points[0] + number_shooting_points[1] + number_shooting_points[2]
forces_sim = np.zeros((3*3 + 3*3, nb_shooting + 1))
for n in range(number_shooting_points[0] + 1):
    forces_sim[[0, 1, 2, 5, 6, 8], n:n+1] = ocp.nlp[0]['contact_forces_func'](np.concatenate([q[:, n], q_dot[:, n]]), tau[:, n], 0)
n_f= number_shooting_points[0]
for n in range(number_shooting_points[1] + 1):
    forces_sim[[0, 1, 2, 5, 6, 8, 9, 10, 11], n_f + n:n_f + n + 1] = ocp.nlp[1]['contact_forces_func'](
        np.concatenate([q[:, n_f + n], q_dot[:, n_f + n]]),
        tau[:, n_f + n],
        0)

figure, axes = plt.subplots(6, 3)
axes = axes.flatten()
coord_label = ['x', 'y', "z"]
contact_point_label = ['Heel', 'Meta1', "Meta5"]
for c in range(3):
    for coord in range(3):
        title = (contact_point_label[c] + "_"+coord_label[coord] + "R")
        axes[3*c + coord].plot(t, forces_sim[3*c + coord, :])
        axes[3*c + coord].plot([phase_time[0], phase_time[0]],
                               [np.min(forces_sim[3*c + coord, :]),
                                np.max(forces_sim[3*c + coord, :])],
                               'k--')
        axes[3 * c + coord].plot([phase_time[0] + phase_time[1], phase_time[0] + phase_time[1]],
                                 [np.min(forces_sim[3 * c + coord, :]),
                                  np.max(forces_sim[3 * c + coord, :])],
                                 'k--')
        axes[3*c + coord].set_title(title)

for c in range(3):
    for coord in range(3):
        title = (contact_point_label[c] + "_"+coord_label[coord] + "L")
        axes[3*c + coord + 9].plot(t, forces_sim[3*c + coord, :])
        axes[3*c + coord + 9].plot([phase_time[0], phase_time[0]],
                               [np.min(forces_sim[3*c + coord, :]),
                                np.max(forces_sim[3*c + coord, :])],
                               'k--')
        axes[3 * c + coord + 9].plot([phase_time[0] + phase_time[1], phase_time[0] + phase_time[1]],
                                 [np.min(forces_sim[3 * c + coord, :]),
                                  np.max(forces_sim[3 * c + coord, :])],
                                 'k--')
        axes[3*c + coord + 9].set_title(title)

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
for i in range(3):
    axes[i].plot(t, forces_sim[i, :] + forces_sim[i+3, :] + forces_sim[i+6, :], 'r-')
    axes[i].plot(t, FR[i], 'k--')
    axes[i].plot([phase_time[0], phase_time[0]],
                             [np.min(forces_sim[i, :] + forces_sim[i+3, :] + forces_sim[i+6, :]),
                              np.max(forces_sim[i, :] + forces_sim[i+3, :] + forces_sim[i+6, :])],
                             'k--')
    axes[i].plot([phase_time[0] + phase_time[1], phase_time[0] + phase_time[1]],
                             [np.min(forces_sim[i, :] + forces_sim[i+3, :] + forces_sim[i+6, :]),
                              np.max(forces_sim[i, :] + forces_sim[i+3, :] + forces_sim[i+6, :])],
                             'k--')
    axes[i].set_title("Forces in " + coord_label[i] + " R")

    axes[i + 3].plot(t, forces_sim[i + 9, :] + forces_sim[i + 12, :] + forces_sim[i + 15, :], 'r-')
    axes[i + 3].plot(t, FL[i], 'k--')
    axes[i + 3].plot([phase_time[0], phase_time[0]],
                             [np.min(forces_sim[i + 9, :] + forces_sim[i + 12, :] + forces_sim[i + 15, :]),
                              np.max(forces_sim[i + 9, :] + forces_sim[i + 12, :] + forces_sim[i + 15, :])],
                             'k--')
    axes[i + 3].plot([phase_time[0] + phase_time[1], phase_time[0] + phase_time[1]],
                             [np.min(forces_sim[i + 9, :] + forces_sim[i + 12, :] + forces_sim[i + 15, :]),
                              np.max(forces_sim[i + 9, :] + forces_sim[i + 12, :] + forces_sim[i + 15, :])],
                             'k--')
    axes[i + 3].set_title("Forces in " + coord_label[i] + " L")


# --- Show results --- #
result = ShowResult(ocp, sol)
result.animate()
result.graphs()