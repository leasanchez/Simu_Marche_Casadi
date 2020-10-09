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

# --- force nul at last point ---
def get_last_contact_force_nul(ocp, nlp, t, x, u, p, contact_name):
    force = nlp.contact_forces_func(x[-1], u[-1], p)
    if contact_name == 'all':
        val = force
    else:
        cn = nlp.model.contactNames()
        for i, c in enumerate(cn):
            if c.to_string() == contact_name:
                val = force[i]
    return val

def track_sum_contact_forces_contact_talon(ocp, nlp, t, x, u, p, grf, target=()):
    ns = nlp["ns"]
    val = []
    for n in range(ns):
        force = nlp["contact_forces_func"](x[n], u[n], p)
        val = vertcat(val, grf[0, t[n]] - (force[0]))
        val = vertcat(val, grf[1, t[n]] - force[1])
        val = vertcat(val, grf[2, t[n]] - (force[2]))
    return dot(val, val)

# --- fcn flatfoot ---
def get_last_contact_forces_flatfoot(ocp, nlp, t, x, u, p, grf):
    force = nlp["contact_forces_func"](x[-1], u[-1], p)
    val = grf[0, t[-1]] - (force[0] + force[4])
    val = vertcat(val, grf[1, t[-1]] - force[1])
    val = vertcat(val, grf[2, t[-1]] - (force[2] + force[3] + force[5]))
    val = vertcat(val, force[2])
    val = vertcat(val, force[0]) # minimise contact talon ?
    return dot(val, val)

def track_sum_contact_forces_flatfoot(ocp, nlp, t, x, u, p, grf, target=()):
    ns = nlp["ns"]
    val = []
    for n in range(ns):
        force = nlp["contact_forces_func"](x[n], u[n], p)
        val = vertcat(val, grf[0, t[n]] - (force[0] + force[4]))
        val = vertcat(val, grf[1, t[n]] - force[1])
        val = vertcat(val, grf[2, t[n]] - (force[2] + force[3] + force[5]))
    return dot(val, val)

def track_sum_moments_flatfoot(ocp, nlp, t, x, u, p, CoP, M_ref, target=()):
    # track moments
    # CoP : evolution of the center of pression evolution
    # M_ref : moments observed at the CoP on the force plateforme

    ns = nlp["ns"] # number of shooting points
    nq = nlp["nbQ"] # number of dof
    val = []
    for n in range(ns):
        q = x[n][:nq]
        markers = nlp["model"].markers(q)  # compute markers positions
        heel = markers[:, 19] - CoP[:, n] + [0.04, 0, 0] # ! modified x position !
        meta1 = markers[:, 21] - CoP[:, n]
        meta5 = markers[:, 24] - CoP[:, n]
        forces = nlp["contact_forces_func"](x[n], u[n], p) # compute forces at each contact points

        # Mcp + CpCOPXFp - MCop = 0
        val = (heel[1] * forces[2] + meta1[1] * forces[3] + meta5[1] * forces[5]) - M_ref[0, n]
        val = vertcat(val, (-heel[0]*forces[2] - meta5[0]*forces[5]) - M_ref[1, n])
        val = vertcat(val, (heel[0]*forces[1] - heel[1]*forces[0] - meta5[1]*forces[4]) - M_ref[2, :])
    return val

# --- fcn forefoot ---
def track_sum_contact_forces_forefoot(ocp, nlp, t, x, u, p, grf, target=()):
    ns = nlp["ns"]
    val = []
    for n in range(ns):
        force = nlp["contact_forces_func"](x[n], u[n], p)
        val = vertcat(val, grf[0, t[n]] - (force[0] + force[2]))
        val = vertcat(val, grf[1, t[n]] - force[3])
        val = vertcat(val, grf[2, t[n]] - (force[1] + force[4]))
    return dot(val, val)

def track_sum_moments_forefoot(ocp, nlp, t, x, u, p, CoP, M_ref, target=()):
    # track moments
    # CoP : evolution of the center of pression evolution
    # M_ref : moments observed at the CoP on the force plateforme

    ns = nlp["ns"] # number of shooting points
    nq = nlp["nbQ"] # number of dof
    val = []
    for n in range(ns):
        q = x[n][:nq]
        markers = nlp["model"].markers(q)  # compute markers positions
        meta1 = markers[:, 21] - CoP[:, n]
        meta5 = markers[:, 24] - CoP[:, n]
        forces = nlp["contact_forces_func"](x[n], u[n], p) # compute forces at each contact points

        # Mcp + CpCOPXFp - MCop = 0
        val = (meta1[1] * forces[1] + meta5[1] * forces[4]) - M_ref[0, n]
        val = vertcat(val, (-meta1[0]*forces[1] - meta5[0]*forces[4]) - M_ref[1, n])
        val = vertcat(val, (- meta1[1]*forces[0] + meta5[0]*forces[3] - meta5[1]*forces[2]) - M_ref[2, :])
    return val

def prepare_ocp(
    biorbd_model, final_time, nb_shooting, markers_ref, grf_ref, q_ref, qdot_ref, M_ref, CoP
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
        objective_functions.add(Objective.Lagrange.TRACK_STATE, weight=200, states_idx=range(nb_q), target=q_ref[p], phase=p)
    # track grf
    objective_functions.add(track_sum_contact_forces_contact_talon,
                            grf=grf_ref[0],
                            custom_type=Objective.Mayer,
                            instant=Instant.ALL,
                            weight=0.00001,
                            phase=0)
    objective_functions.add(track_sum_contact_forces_flatfoot,
                            grf=grf_ref[1],
                            custom_type=Objective.Mayer,
                            instant=Instant.ALL,
                            weight=0.00001,
                            phase=1)
    objective_functions.add(track_sum_moments_flatfoot,
                            CoP=CoP[1],
                            M_ref=M_ref[1],
                            custom_type=Objective.Mayer,
                            instant=Instant.ALL,
                            weight=0.00001,
                            phase=1)
    objective_functions.add(track_sum_contact_forces_forefoot,
                            grf=grf_ref[2],
                            custom_type=Objective.Mayer,
                            instant=Instant.ALL,
                            weight=0.00001,
                            phase=2)
    # objective_functions.add(track_sum_moments_forefoot,
    #                         CoP=CoP[2],
    #                         M_ref=M_ref[2],
    #                         custom_type=Objective.Mayer,
    #                         instant=Instant.ALL,
    #                         weight=0.00001,
    #                         phase=2)

    # Dynamics
    dynamics = DynamicsTypeList()
    for p in range(nb_phases):
        dynamics.add(DynamicsType.TORQUE_DRIVEN_WITH_CONTACT, phase=p)

    # Constraints
    constraints = ConstraintList()
    # --- phase forefoot ---
    constraints.add( # positive vertical forces
        Constraint.CONTACT_FORCE_INEQUALITY,
        direction="GREATER_THAN",
        instant=Instant.ALL,
        contact_force_idx=(1, 4),
        boundary=50,
        phase=2,
    )
    constraints.add( # forces heel at zeros at the end of the phase
        get_last_contact_force_nul,
        instant=Instant.ALL,
        contact_name='Heel_r_X',
        phase=1,
    )
    constraints.add( # forces heel at zeros at the end of the phase
        get_last_contact_force_nul,
        instant=Instant.ALL,
        contact_name='Heel_r_Z',
        phase=1,
    )
    constraints.add( # positive vertical forces
        Constraint.CONTACT_FORCE_INEQUALITY,
        direction="GREATER_THAN",
        instant=Instant.ALL,
        contact_force_idx=(2, 3, 5),
        boundary=50,
        phase=1,
    )
    constraints.add(
        get_last_contact_force_nul,
        instant=Instant.ALL,
        contact_name='all',
        phase=2,
    )

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
    u_init = InitialConditionsList()
    n_shoot=0
    for p in range(nb_phases):
        init_x = np.zeros((nb_q + nb_qdot, nb_shooting[p] + 1))
        init_x[:nb_q, :] = np.load('./RES/1leg/3phases/q.npy')[:, n_shoot:n_shoot + nb_shooting[p] + 1] #q_ref[p]
        init_x[nb_q:nb_q + nb_qdot, :] = np.load('./RES/1leg/3phases/q_dot.npy')[:,  n_shoot:n_shoot + nb_shooting[p] + 1] #qdot_ref[p]
        x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)

        init_u = np.load('./RES/1leg/3phases/tau.npy')[:,  n_shoot:n_shoot + nb_shooting[p]] # np.zeros((nb_tau, nb_shooting[p]))
        u_init.add(init_u, interpolation=InterpolationType.EACH_FRAME)
        n_shoot += nb_shooting[p]

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
    CoP = Data_to_track.load_data_CoP(number_shooting_points)
    M_ref = Data_to_track.load_data_Moment_at_CoP(number_shooting_points)

    biorbd_model = (
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_1contact_deGroote_3d_Heel.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_3contacts_deGroote_3d.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_contact_deGroote_3d_Forefoot.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_0contact_deGroote_3d.bioMod")
    )

    ocp = prepare_ocp(
        biorbd_model=(biorbd_model[0], biorbd_model[1], biorbd_model[2]),
        final_time=(phase_time[0], phase_time[1], phase_time[2]),
        nb_shooting=(number_shooting_points[0], number_shooting_points[1], number_shooting_points[2]),
        markers_ref=(markers_ref[0], markers_ref[1], markers_ref[2]),
        grf_ref=(grf_ref[0], grf_ref[1], grf_ref[2]),
        q_ref=(q_ref[0], q_ref[1], q_ref[2]),
        qdot_ref=(qdot_ref[0], qdot_ref[1], qdot_ref[2]),
        M_ref=(M_ref[0], M_ref[1], M_ref[2]),
        CoP=(CoP[0], CoP[1], CoP[2]),
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
        axes[i].plot(t, q_plot, 'b--')
        pt = phase_time[0]
        for p in range(nb_phases):
            axes[i].plot([pt, pt], [np.min(q_plot), np.max(q_plot)], 'k--')
            pt += phase_time[p + 1]
        axes[i].set_title(q_name[i])
    plt.legend(['simulated', 'reference'])

    # --- plot grf ---
    nb_shooting = number_shooting_points[0] + number_shooting_points[1] + number_shooting_points[2]
    labels_forces = ['Heel_r_X', 'Heel_r_Y', 'Heel_r_Z',
                     'Meta_1_r_X', 'Meta_1_r_Y', 'Meta_1_r_Z',
                     'Meta_5_r_X', 'Meta_5_r_Y', 'Meta_5_r_Z',]
    forces = {}
    for label in labels_forces:
        forces[label] = np.zeros(nb_shooting + 1)

    n_shoot = 0
    for p in range(nb_phases):
        cn = biorbd_model[p].contactNames()
        for n in range(number_shooting_points[p] + 1):
            forces_sim = ocp.nlp[p]['contact_forces_func'](np.concatenate([q[:, n_shoot + n], q_dot[:, n_shoot + n]]),
                                                           tau[:, n_shoot + n], 0)
            for i, c in enumerate(cn):
                if c.to_string() in forces:
                    forces[c.to_string()][n_shoot + n] = forces_sim[i]
        n_shoot += number_shooting_points[p]

    figure, axes = plt.subplots(3, 3)
    axes = axes.flatten()
    for i, f in enumerate(forces):
        axes[i].plot(t, forces[f], 'r-')
        axes[i].set_title(f)
        pt = phase_time[0]
        for p in range(nb_phases):
            axes[i].plot([pt, pt], [np.min(forces[f]), np.max(forces[f])], 'k--')
            pt += phase_time[p + 1]

    figure, axes = plt.subplots(1, 3)
    axes = axes.flatten()
    force_plot_x_R = grf_ref[0][0, :]
    force_plot_y_R = grf_ref[0][1, :]
    force_plot_z_R = grf_ref[0][2, :]
    for p in range(1, nb_phases):
        force_plot_x_R = np.concatenate([force_plot_x_R[:-1], grf_ref[p][0, :]])
        force_plot_y_R = np.concatenate([force_plot_y_R[:-1], grf_ref[p][1, :]])
        force_plot_z_R = np.concatenate([force_plot_z_R[:-1], grf_ref[p][2, :]])

    FR = [force_plot_x_R, force_plot_y_R, force_plot_z_R]
    coord_label = ['X', 'Y', 'Z']
    for i in range(3):
        axes[i].plot(t, forces[f"Heel_r_{coord_label[i]}"] + forces[f"Meta_1_r_{coord_label[i]}"] + forces[
            f"Meta_5_r_{coord_label[i]}"], 'r-')
        axes[i].plot(t, FR[i], 'b--')
        axes[i].set_title("Forces in " + coord_label[i] + " R")

        pt = phase_time[0]
        for p in range(nb_phases):
            axes[i].plot([pt, pt], [np.min(FR[i]), np.max(FR[i])], 'k--')
            pt += phase_time[p + 1]


    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
    result.graphs()
