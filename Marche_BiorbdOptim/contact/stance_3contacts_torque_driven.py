import numpy as np
from casadi import dot, Function, vertcat, MX, mtimes, nlpsol
import biorbd
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
        val = []
        for i, c in enumerate(cn):
            for name in contact_name:
                if c.to_string() == name:
                    val = vertcat(val, force[i])
    return val

# --- fcn contact talon ---
def track_sum_contact_forces_contact_talon(ocp, nlp, t, x, u, p, grf, target=()):
    ns = nlp.ns
    val = []
    for n in range(ns):
        force = nlp.contact_forces_func(x[n], u[n], p)
        val = vertcat(val, grf[0, t[n]] - (force[0]))
        val = vertcat(val, grf[1, t[n]] - force[1])
        val = vertcat(val, grf[2, t[n]] - (force[2]))
    return val

# --- fcn flatfoot ---
def get_last_contact_forces_flatfoot(ocp, nlp, t, x, u, p, grf):
    force = nlp.contact_forces_func(x[-1], u[-1], p)
    val = grf[0, t[-1]] - force[3]
    val = vertcat(val, grf[1, t[-1]] - force[4])
    val = vertcat(val, grf[2, t[-1]] - (force[2] + force[5]))
    return val

def track_sum_contact_forces_flatfoot(ocp, nlp, t, x, u, p, grf, target=()):
    ns = nlp.ns
    val = []
    for n in range(ns):
        force = nlp.contact_forces_func(x[n], u[n], p)
        val = vertcat(val, grf[0, t[n]] - (force[0] + force[3]))
        val = vertcat(val, grf[1, t[n]] - force[4])
        val = vertcat(val, grf[2, t[n]] - (force[1] + force[2] + force[5]))
    return val

def track_sum_moments_flatfoot(ocp, nlp, t, x, u, p, CoP, M_ref, target=()):
    # track moments
    # CoP : evolution of the center of pression evolution
    # M_ref : moments observed at the CoP on the force plateforme

    ns = nlp.ns # number of shooting points
    nq = nlp.model.nbQ() # number of dof
    val = []
    for n in range(ns):
        q = x[n][:nq]
        markers = nlp.model.markers(q)  # compute markers positions
        heel =  markers[:, 19] + [0.04, 0, 0] - CoP[:, n]# ! modified x position !
        meta1 = markers[:, 21] - CoP[:, n]
        meta5 = markers[:, 24] - CoP[:, n]
        forces = nlp.contact_forces_func(x[n], u[n], p) # compute forces at each contact points

        # Mcp + CpCOPXFp - MCop = 0
        val = (heel[1] * forces[1] + meta1[1] * forces[2] + meta5[1] * forces[5]) - M_ref[0, n]
        val = vertcat(val, (-heel[0]*forces[1] - meta1[0]*forces[2] - meta5[0]*forces[5]) - M_ref[1, n])
        val = vertcat(val, (-heel[1]*forces[0] + meta5[0]*forces[4] - meta5[1]*forces[3]) - M_ref[2, :])
    return val

# --- fcn forefoot ---
def track_sum_contact_forces_forefoot(ocp, nlp, t, x, u, p, grf, target=()):
    ns = nlp.ns
    val = []
    for n in range(ns):
        force = nlp.contact_forces_func(x[n], u[n], p)
        val = vertcat(val, grf[0, t[n]] - (force[0] + force[2]))
        val = vertcat(val, grf[1, t[n]] - force[3])
        val = vertcat(val, grf[2, t[n]] - (force[1] + force[4]))
    return val

def track_sum_moments_forefoot(ocp, nlp, t, x, u, p, CoP, M_ref, target=()):
    # track moments
    # CoP : evolution of the center of pression evolution
    # M_ref : moments observed at the CoP on the force plateforme

    ns = nlp.ns # number of shooting points
    nq = nlp.model.nbQ() # number of dof
    val = []
    for n in range(ns):
        q = x[n][:nq]
        markers = nlp.model.markers(q)  # compute markers positions
        meta1 = markers[:, 21] - CoP[:, n]
        meta5 = markers[:, 24] - CoP[:, n]
        forces = nlp.contact_forces_func(x[n], u[n], p) # compute forces at each contact points

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
    # --- contact talon ---
    objective_functions.add(track_sum_contact_forces_contact_talon,
                            grf=grf_ref[0],
                            custom_type=Objective.Lagrange,
                            instant=Instant.ALL,
                            weight=0.00001,
                            quadratic=True,
                            phase=0)

    # --- flatfoot ---
    objective_functions.add(track_sum_contact_forces_flatfoot,
                            grf=grf_ref[1],
                            custom_type=Objective.Lagrange,
                            instant=Instant.ALL,
                            weight=0.00001,
                            quadratic=True,
                            phase=1)
    objective_functions.add(track_sum_moments_flatfoot,
                            CoP=CoP[1],
                            M_ref=M_ref[1],
                            custom_type=Objective.Lagrange,
                            instant=Instant.ALL,
                            weight=0.00001,
                            quadratic=True,
                            phase=1)
    # objective_functions.add(get_last_contact_forces_flatfoot,
    #                         grf=grf_ref[1],
    #                         custom_type=Objective.Mayer,
    #                         instant=Instant.ALL,
    #                         weight=0.00001,
    #                         quadratic = True,
    #                         phase=1)

    # --- forefoot ---
    objective_functions.add(track_sum_contact_forces_forefoot,
                            grf=grf_ref[2],
                            custom_type=Objective.Lagrange,
                            instant=Instant.ALL,
                            weight=0.00001,
                            quadratic=True,
                            phase=2)
    objective_functions.add(track_sum_moments_forefoot,
                            CoP=CoP[2],
                            M_ref=M_ref[2],
                            custom_type=Objective.Lagrange,
                            instant=Instant.ALL,
                            weight=0.00001,
                            quadratic=True,
                            phase=2)

    # Dynamics
    dynamics = DynamicsTypeList()
    for p in range(nb_phases - 1):
        dynamics.add(DynamicsType.TORQUE_DRIVEN_WITH_CONTACT, phase=p)
    dynamics.add(DynamicsType.TORQUE_DRIVEN, phase=3)

    # Constraints
    constraints = ConstraintList()
    # --- phase flatfoot ---
    constraints.add( # positive vertical forces
        Constraint.CONTACT_FORCE_INEQUALITY,
        direction="GREATER_THAN",
        instant=Instant.ALL,
        contact_force_idx=(1, 2, 5),
        boundary=0,
        phase=1,
    )
    constraints.add( # forces heel at zeros at the end of the phase
        get_last_contact_force_nul,
        instant=Instant.ALL,
        contact_name=('Heel_r_X', 'Heel_r_Z'),
        phase=1,
    )

    # --- phase forefoot ---
    constraints.add( # positive vertical forces
        Constraint.CONTACT_FORCE_INEQUALITY,
        direction="GREATER_THAN",
        instant=Instant.ALL,
        contact_force_idx=(1, 4),
        boundary=0,
        phase=2,
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
    state_transitions.add(StateTransition.IMPACT, phase_pre_idx=1)

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
    u_init = InitialGuessList()
    n_shoot=0
    for p in range(nb_phases):
        init_x = np.zeros((nb_q + nb_qdot, nb_shooting[p] + 1))
        init_x[:nb_q, :] = q_ref[p] #np.load('./RES/1leg/3phases/TM1TM2CFI12_100/q.npy')[:, n_shoot:n_shoot + nb_shooting[p] + 1] #
        init_x[nb_q:nb_q + nb_qdot, :] = qdot_ref[p] #np.load('./RES/1leg/3phases/TM1TM2CFI12_100/q_dot.npy')[:,  n_shoot:n_shoot + nb_shooting[p] + 1] #qdot_ref[p]
        x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)

        init_u = np.zeros((nb_tau, number_shooting_points[p])) #np.load('./RES/1leg/3phases/TM1TM2CFI12_100/tau.npy')[:,  n_shoot:n_shoot + nb_shooting[p]] # np.zeros((nb_tau, nb_shooting[p]))
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
        biorbd.Model("../../ModelesS2M/Marche_saine/Florent_1leg_12dof_heel.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/Florent_1leg_12dof_flatfoot.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/Florent_1leg_12dof_forefoot.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/Florent_1leg_12dof_0contact.bioMod")
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
        biorbd.Model("../../ModelesS2M/Marche_saine/Florent_1leg_12dof_heel.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/Florent_1leg_12dof_flatfoot.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/Florent_1leg_12dof_forefoot.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/Florent_1leg_12dof_0contact.bioMod")
    )

    ocp = prepare_ocp(
        biorbd_model=biorbd_model,
        final_time= phase_time,
        nb_shooting=number_shooting_points,
        markers_ref=markers_ref,
        grf_ref=grf_ref,
        q_ref=q_ref,
        qdot_ref=qdot_ref,
        M_ref=M_ref,
        CoP=CoP,
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
        pt=0
        for p in range(nb_phases):
            pt += phase_time[p]
            axes[i].plot([pt, pt], [np.min(q_plot), np.max(q_plot)], 'k--')
        axes[i].set_title(q_name[i])
    plt.legend(['simulated', 'reference'])

    # --- plot grf ---
    # INIT
    nb_shooting = number_shooting_points[0] + number_shooting_points[1] + number_shooting_points[2] + number_shooting_points[3] #total number of shooting points
    labels_forces = ['Heel_r_X', 'Heel_r_Y', 'Heel_r_Z',
                     'Meta_1_r_X', 'Meta_1_r_Y', 'Meta_1_r_Z',
                     'Meta_5_r_X', 'Meta_5_r_Y', 'Meta_5_r_Z',]
    forces = {}     # dictionary for forces
    for label in labels_forces:
        forces[label] = np.zeros(nb_shooting + 1)

    # COMPUTE FORCES FOR EACH PHASE
    n_shoot = 0
    for p in range(nb_phases-1):
        cn = biorbd_model[p].contactNames()           # get contact names for each model
        for n in range(number_shooting_points[p] + 1):
            # compute contact forces
            forces_sim = ocp.nlp[p].contact_forces_func(np.concatenate([q[:, n_shoot + n], q_dot[:, n_shoot + n]]),
                                                           tau[:, n_shoot + n], 0)
            for i, c in enumerate(cn):
                if c.to_string() in forces:
                    forces[c.to_string()][n_shoot + n] = forces_sim[i]  #put corresponding forces in dictionnary
        n_shoot += number_shooting_points[p]

    # PLOT EACH FORCES
    figure, axes = plt.subplots(3, 3)
    axes = axes.flatten()
    for i, f in enumerate(forces):
        axes[i].scatter(t, forces[f], color='r', s=3)
        axes[i].plot(t, forces[f], 'r-', alpha=0.5)
        if (i==2) or (i==5) or (i==8):
            axes[i].plot([t[0], t[-1]], [0, 0], 'k--')
            axes[i].plot([t[0], t[-1]], [-100, -100], 'k--')
        axes[i].set_title(f)
        pt = 0
        for p in range(nb_phases):
            pt += phase_time[p]
            axes[i].plot([pt, pt], [np.min(forces[f]), np.max(forces[f])], 'k--')


    # PLOT SUM FORCES VS PLATEFORME FORCES
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

        pt = 0
        for p in range(nb_phases):
            pt += phase_time[p]
            axes[i].plot([pt, pt], [np.min(FR[i]), np.max(FR[i])], 'k--')


    # COMPUTE MOMENTS
    moments_sim = np.zeros((3, nb_shooting + 1))
    CoP_plot_x_R = CoP[0][0, :]
    CoP_plot_y_R = CoP[0][1, :]
    CoP_plot_z_R = CoP[0][2, :]

    moment_plot_x_R = M_ref[0][0, :]
    moment_plot_y_R = M_ref[0][1, :]
    moment_plot_z_R = M_ref[0][2, :]
    for p in range(1, nb_phases):
        CoP_plot_x_R = np.concatenate([CoP_plot_x_R[:-1], CoP[p][0, :]])
        CoP_plot_y_R = np.concatenate([CoP_plot_y_R[:-1], CoP[p][1, :]])
        CoP_plot_z_R = np.concatenate([CoP_plot_z_R[:-1], CoP[p][2, :]])

        moment_plot_x_R = np.concatenate([moment_plot_x_R[:-1], M_ref[p][0, :]])
        moment_plot_y_R = np.concatenate([moment_plot_y_R[:-1], M_ref[p][1, :]])
        moment_plot_z_R = np.concatenate([moment_plot_z_R[:-1], M_ref[p][2, :]])

    q_sym = MX.sym("q", nb_q, 1)
    func = biorbd.to_casadi_func("markers", ocp.nlp[0].model.markers, q_sym)
    for n in range(nb_shooting):
         # compute markers positions
        markers = func(q[:, n])
        heel = markers[:, 19] - [CoP_plot_x_R[n], CoP_plot_y_R[n], CoP_plot_z_R[n]] + [0.04, 0, 0]
        meta1 = markers[:, 21] - [CoP_plot_x_R[n], CoP_plot_y_R[n], CoP_plot_z_R[n]]
        meta5 = markers[:, 24] - [CoP_plot_x_R[n], CoP_plot_y_R[n], CoP_plot_z_R[n]]  # positions of the contact points VS CoP

        # Mcp + CpCOPXFp - MCop = 0
        moments_sim[0, n] = heel[1] * forces["Heel_r_Z"][n] + meta1[1] * forces["Meta_1_r_Z"][n] + meta5[1] * forces["Meta_5_r_Z"][n]
        moments_sim[1, n] = -heel[0] * forces["Heel_r_Z"][n] - meta1[0]*forces["Meta_1_r_Z"][n] - meta5[0]*forces["Meta_5_r_Z"][n]
        moments_sim[2, n] = heel[0] * forces["Heel_r_Y"][n] - heel[1] * forces["Heel_r_X"][n] \
                            + meta1[0] * forces["Meta_1_r_Y"][n] - meta1[1] * forces["Meta_1_r_X"][n] \
                            + meta5[0] * forces["Meta_5_r_Y"][n] - meta5[1] * forces["Meta_5_r_X"][n]

    figure, axes = plt.subplots(1, 3)
    axes = axes.flatten()
    MR = [moment_plot_x_R, moment_plot_y_R, moment_plot_z_R]
    coord_label = ['X', 'Y', 'Z']
    for i in range(3):
        axes[i].plot(t, moments_sim[i, :], 'r-')
        axes[i].plot(t, MR[i], 'b--')
        axes[i].set_title("Moments in " + coord_label[i] + " R")
        pt = phase_time[0]
        for p in range(nb_phases):
            axes[i].plot([pt, pt], [np.min(moments_sim[i, :]), np.max(moments_sim[i, :])], 'k--')
            pt += phase_time[p + 1]

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
    result.graphs()
