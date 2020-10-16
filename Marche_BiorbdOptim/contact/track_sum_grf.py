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
# --- force nul at last point ---
def get_last_contact_force_null(ocp, nlp, t, x, u, p, contact_name):
    force = nlp.contact_forces_func(x[-1], u[-1], p)
    cn = nlp.model.contactNames()
    val = []
    for i, c in enumerate(cn):
        if isinstance(contact_name, tuple):
            for name in contact_name:
                if name in c.to_string():
                    val = vertcat(val, force[i])
        else:
            if contact_name in c.to_string():
                val = vertcat(val, force[i])
    return val
    ns = nlp.ns
    val = []
    for n in range(ns):
        force = nlp.contact_forces_func(x[n], u[n], p)
        val = vertcat(val, grf[0, t[n]] - (force[0]))
        val = vertcat(val, grf[1, t[n]] - force[1])
        val = vertcat(val, grf[2, t[n]] - (force[2]))
    return dot(val, val)

# --- fcn flatfoot ---
def get_last_contact_forces_flatfoot(ocp, nlp, t, x, u, p, grf):
    force = nlp.contact_forces_func(x[-1], u[-1], p)
    val = grf[0, t[-1]] - force[3]
    val = vertcat(val, grf[1, t[-1]] - force[4])
    val = vertcat(val, grf[2, t[-1]] - (force[2] + force[5]))
    return dot(val, val)

def get_last_contact_forces_talon_flatfoot(ocp, nlp, t, x, u, p):
    force = nlp.contact_forces_func(x[-1], u[-1], p)
    val = force[0]
    val = vertcat(val, force[1])
    return dot(val, val)

def track_sum_contact_forces_flatfoot(ocp, nlp, t, x, u, p, grf, target=()):
    ns = nlp.ns
    val = []
    for n in range(ns):
        force = nlp.contact_forces_func(x[n], u[n], p)
        val = vertcat(val, grf[0, t[n]] - (force[0] + force[3]))
        val = vertcat(val, grf[1, t[n]] - force[4])
        val = vertcat(val, grf[2, t[n]] - (force[1] + force[2] + force[5]))
    return dot(val, val)

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
def get_last_contact_forces_forefoot(ocp, nlp, t, x, u, p):
    force = nlp.contact_forces_func(x[-1], u[-1], p)
    return dot(force, force)

def track_sum_contact_forces_forefoot(ocp, nlp, t, x, u, p, grf, target=()):
    ns = nlp.ns
    val = []
    for n in range(ns):
        force = nlp.contact_forces_func(x[n], u[n], p)
        val = vertcat(val, grf[0, t[n]] - (force[0] + force[2]))
        val = vertcat(val, grf[1, t[n]] - force[3])
        val = vertcat(val, grf[2, t[n]] - (force[1] + force[4]))
    return dot(val, val)

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
    nb_q = biorbd_model.nbQ()
    nb_qdot = biorbd_model.nbQdot()
    nb_tau = biorbd_model.nbGeneralizedTorque()

    torque_min, torque_max, torque_init = -1000, 1000, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Lagrange.TRACK_STATE, weight=200, states_idx=range(nb_q), target=q_ref,)
    # track grf
    # --- flatfoot ---
    objective_functions.add(track_sum_contact_forces_flatfoot, # track forces
                            grf=grf_ref,
                            custom_type=Objective.Lagrange,
                            instant=Instant.ALL,
                            weight=0.0001,)
    objective_functions.add(track_sum_moments_flatfoot, # track moments
                            CoP=CoP,
                            M_ref=M_ref,
                            custom_type=Objective.Lagrange,
                            instant=Instant.ALL,
                            weight=0.0001,)
    objective_functions.add(get_last_contact_forces_flatfoot, # track last node forces
                            grf=grf_ref,
                            custom_type=Objective.Mayer,
                            instant=Instant.ALL,
                            weight=0.0001,)

    # Dynamics
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.TORQUE_DRIVEN_WITH_CONTACT)

    # Constraints
    constraints = ConstraintList()
    # --- phase flatfoot ---
    constraints.add( # positive vertical forces
        Constraint.CONTACT_FORCE_INEQUALITY,
        direction="GREATER_THAN",
        instant=Instant.ALL,
        contact_force_idx=(1, 2, 5),
        boundary=0,
    )
    constraints.add(
        get_last_contact_force_null,
        instant=Instant.ALL,
        contact_name=('Meta_1_l', 'Meta_5_l'),
    )

    # Path constraint
    x_bounds = BoundsList()
    u_bounds = BoundsList()
    x_bounds.add(QAndQDotBounds(biorbd_model))
    u_bounds.add([
        [torque_min] * nb_tau,
        [torque_max] * nb_tau,
    ])

    # Initial guess
    x_init = InitialGuessList()
    u_init = InitialGuessList()
    init_x = np.zeros((nb_q + nb_qdot, nb_shooting + 1))
    init_x[:nb_q, :] = np.load('./RES/1leg/flatfoot/q.npy') #q_ref[p]
    init_x[nb_q:nb_q + nb_qdot, :] = np.load('./RES/1leg/flatfoot/q_dot.npy') #qdot_ref[p]
    x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)

    init_u = np.load('./RES/1leg/flatfoot/tau.npy')[:, :-1] # np.zeros((nb_tau, nb_shooting[p]))
    u_init.add(init_u, interpolation=InterpolationType.EACH_FRAME)

    # ------------- #

    return OptimalControlProgram(
        biorbd_model=biorbd_model,
        dynamics_type=dynamics,
        number_shooting_points=nb_shooting,
        phase_time=final_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        nb_threads=4,
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
    number_shooting_points[1]=35
    grf_ref = Data_to_track.load_data_GRF(number_shooting_points)  # get ground reaction forces
    markers_ref = Data_to_track.load_data_markers(number_shooting_points)
    q_ref = Data_to_track.load_q_kalman(number_shooting_points)
    qdot_ref = Data_to_track.load_qdot_kalman(number_shooting_points)
    CoP = Data_to_track.load_data_CoP(number_shooting_points)
    M_ref = Data_to_track.load_data_Moment_at_CoP(number_shooting_points)

    ocp = prepare_ocp(
        biorbd_model=(biorbd_model[1]),
        final_time=(phase_time[1]),
        nb_shooting=35,
        markers_ref=(markers_ref[1]),
        grf_ref=(grf_ref[1]),
        q_ref=(q_ref[1]),
        qdot_ref=(qdot_ref[1]),
        M_ref=(M_ref[1]),
        CoP=(CoP[1]),
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
    t = np.linspace(0, phase_time[1], number_shooting_points[1] + 1)

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
        axes[i].plot(t, q_ref[1][i, :])
        axes[i].set_title(q_name[i])
    plt.legend(['simulated', 'reference'])

    # --- plot grf ---
    # INIT
    nb_shooting = number_shooting_points[1] + 1 #total number of shooting points
    labels_forces = ['Heel_r_X', 'Heel_r_Y', 'Heel_r_Z',
                     'Meta_1_r_X', 'Meta_1_r_Y', 'Meta_1_r_Z',
                     'Meta_5_r_X', 'Meta_5_r_Y', 'Meta_5_r_Z',]
    forces = {}     # dictionary for forces
    for label in labels_forces:
        forces[label] = np.zeros(nb_shooting)

    # COMPUTE FORCES FOR EACH PHASE
    cn = biorbd_model[1].contactNames()           # get contact names for each model
    for n in range(number_shooting_points[1] + 1):
        # compute contact forces
        forces_sim = ocp.nlp[0].contact_forces_func(np.concatenate([q[:, n], q_dot[:, n]]), tau[:, n], 0)
        for i, c in enumerate(cn):
            if c.to_string() in forces:
                forces[c.to_string()][n] = forces_sim[i]  #put corresponding forces in dictionnary

    # PLOT EACH FORCES
    figure, axes = plt.subplots(3, 3)
    axes = axes.flatten()
    for i, f in enumerate(forces):
        axes[i].scatter(t, forces[f], color='r', s=3)
        axes[i].plot(t, forces[f], 'r-', alpha=0.5)
        axes[i].set_title(f)
        if (i==2) or (i==5) or (i==8):
            axes[i].plot([t[0], t[-1]], [0, 0], 'k--')

    # PLOT SUM FORCES VS PLATEFORME FORCES
    figure, axes = plt.subplots(1, 3)
    axes = axes.flatten()
    coord_label = ['X', 'Y', 'Z']
    for i in range(3):
        axes[i].plot(t, forces[f"Heel_r_{coord_label[i]}"] + forces[f"Meta_1_r_{coord_label[i]}"] + forces[
            f"Meta_5_r_{coord_label[i]}"], 'r-')
        axes[i].plot(t, grf_ref[1][i, :], 'b--')
        axes[i].set_title("Forces in " + coord_label[i] + " R")


    # COMPUTE MOMENTS
    moments_sim = np.zeros((3, nb_shooting))
    q_sym = MX.sym("q", nb_q, 1)
    func = biorbd.to_casadi_func("markers", ocp.nlp[0].model.markers, q_sym)
    for n in range(number_shooting_points[1]):
         # compute markers positions
        markers = func(q[:, n])
        heel = markers[:, 19] - CoP[1][:, n] + [0.04, 0, 0]
        meta1 = markers[:, 21] - CoP[1][:, n]
        meta5 = markers[:, 24] - CoP[1][:, n]  # positions of the contact points VS CoP

        # Mcp + CpCOPXFp - MCop = 0
        moments_sim[0, n] = heel[1] * forces["Heel_r_Z"][n] + meta1[1] * forces["Meta_1_r_Z"][n] + meta5[1] * forces["Meta_5_r_Z"][n]
        moments_sim[1, n] = -heel[0] * forces["Heel_r_Z"][n] - meta1[0]*forces["Meta_1_r_Z"][n] - meta5[0]*forces["Meta_5_r_Z"][n]
        moments_sim[2, n] = heel[0] * forces["Heel_r_Y"][n] - heel[1] * forces["Heel_r_X"][n] \
                            + meta1[0] * forces["Meta_1_r_Y"][n] - meta1[1] * forces["Meta_1_r_X"][n] \
                            + meta5[0] * forces["Meta_5_r_Y"][n] - meta5[1] * forces["Meta_5_r_X"][n]

    figure, axes = plt.subplots(1, 3)
    axes = axes.flatten()
    coord_label = ['X', 'Y', 'Z']
    for i in range(3):
        axes[i].plot(t, moments_sim[i, :], 'r-')
        axes[i].plot(t, M_ref[1][i, :], 'b--')
        axes[i].set_title("Moments in " + coord_label[i] + " R")

    # COMPARISON BETWEEN MARKERS POSITIONS
    func = biorbd.to_casadi_func("markers", ocp.nlp[0].model.markers, q_sym)
    marker = np.zeros((3, biorbd_model[0].nbMarkers(), number_shooting_points[1]))
    for n in range(number_shooting_points[1]):
        marker[:, :, n] = func(q[:, n])

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
    result.graphs()
