import numpy as np
from casadi import dot, Function, vertcat, MX, mtimes, nlpsol
import biorbd
import BiorbdViz as BiorbdViz
from matplotlib import pyplot as plt
from Marche_BiorbdOptim.LoadData import Data_to_track
import Marche_BiorbdOptim.contact.Affichage_resultats as Affichage_resultat

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
            if isinstance(contact_name, tuple):
                for name in contact_name:
                    if name in c.to_string():
                        val = vertcat(val, force[i])
            else:
                if contact_name in c.to_string():
                    val = vertcat(val, force[i])
    return val


# --- track grf ---
def track_sum_contact_forces(ocp, nlp, t, x, u, p, grf):
    ns = nlp.ns  # number of shooting points for the phase
    val = []     # init
    cn = nlp.model.contactNames() # contact name for the model

    # --- compute forces ---
    forces={} # define dictionnary with all the contact point possible
    labels_forces = ['Heel_r_X', 'Heel_r_Y', 'Heel_r_Z',
                     'Meta_1_r_X', 'Meta_1_r_Y', 'Meta_1_r_Z',
                     'Meta_5_r_X', 'Meta_5_r_Y', 'Meta_5_r_Z',
                     'Toe_r_X', 'Toe_r_Y', 'Toe_r_Z',]
    for label in labels_forces:
        forces[label] = [] # init

    for n in range(ns):
        for f in forces:
            forces[f].append(0.0) # init: put 0 if the contact point is not activated

        force = nlp.contact_forces_func(x[n], u[n], p) # compute force
        for i, c in enumerate(cn):
            if c.to_string() in forces: # check if contact point is activated
                forces[c.to_string()][n] = force[i]  # put corresponding forces in dictionnary

        # --- tracking forces ---
        val = vertcat(val, grf[0, t[n]] - (forces["Heel_r_X"][n] + forces["Meta_1_r_X"][n] + forces["Meta_5_r_X"][n] + forces["Toe_r_X"][n]))
        val = vertcat(val, grf[1, t[n]] - (forces["Heel_r_Y"][n] + forces["Meta_1_r_Y"][n] + forces["Meta_5_r_Y"][n] + forces["Toe_r_Y"][n]))
        val = vertcat(val, grf[2, t[n]] - (forces["Heel_r_Z"][n] + forces["Meta_1_r_Z"][n] + forces["Meta_5_r_Z"][n] + forces["Toe_r_Z"][n]))
    return val

# --- track moments ---
def track_sum_contact_moments(ocp, nlp, t, x, u, p, CoP, M_ref):
    # --- aliases ---
    ns = nlp.ns  # number of shooting points for the phase
    nq = nlp.model.nbQ()  # number of dof
    cn = nlp.model.contactNames() # contact name for the model
    val = []  # init

    # --- init forces ---
    forces={} # define dictionnary with all the contact point possible
    labels_forces = ['Heel_r_X', 'Heel_r_Y', 'Heel_r_Z',
                     'Meta_1_r_X', 'Meta_1_r_Y', 'Meta_1_r_Z',
                     'Meta_5_r_X', 'Meta_5_r_Y', 'Meta_5_r_Z',
                     'Toe_r_X', 'Toe_r_Y', 'Toe_r_Z',]
    for label in labels_forces:
        forces[label] = [] # init

    for n in range(ns):
        # --- compute contact point position ---
        q = x[n][:nq]
        markers = nlp.model.markers(q)  # compute markers positions
        heel =  markers[-4].to_mx() - CoP[:, t[n]]
        meta1 = markers[-3].to_mx() - CoP[:, t[n]]
        meta5 = markers[-2].to_mx() - CoP[:, t[n]]
        toe =   markers[-1].to_mx() - CoP[:, t[n]]

        # --- compute forces ---
        for f in forces:
            forces[f].append(0.0) # init: put 0 if the contact point is not activated
        force = nlp.contact_forces_func(x[n], u[n], p) # compute force
        for i, c in enumerate(cn):
            if c.to_string() in forces: # check if contact point is activated
                forces[c.to_string()][n] = force[i]  # put corresponding forces in dictionnary

        # --- tracking moments ---
        Mx = heel[1]*forces["Heel_r_Z"][n] + meta1[1]*forces["Meta_1_r_Z"][n] + meta5[1]*forces["Meta_5_r_Z"][n] + toe[1]*forces["Toe_r_Z"][n]
        My = -heel[0]*forces["Heel_r_Z"][n] - meta1[0]*forces["Meta_1_r_Z"][n] - meta5[0]*forces["Meta_5_r_Z"][n] - toe[0]*forces["Toe_r_Z"][n]
        Mz = heel[0]*forces["Heel_r_Y"][n] - heel[1]*forces["Heel_r_X"][n]\
             + meta1[0]*forces["Meta_1_r_Y"][n] - meta1[1]*forces["Meta_1_r_X"][n]\
             + meta5[0]*forces["Meta_5_r_Y"][n] - meta5[1]*forces["Meta_5_r_X"][n]\
             + toe[0]*forces["Toe_r_Y"][n] - toe[1]*forces["Toe_r_X"][n]
        val = vertcat(val, M_ref[0, t[n]] - Mx)
        val = vertcat(val, M_ref[1, t[n]] - My)
        val = vertcat(val, M_ref[2, t[n]] - Mz)
    return val

def prepare_ocp(
    biorbd_model, final_time, nb_shooting, markers_ref, grf_ref, q_ref, qdot_ref, nb_threads,
):
    # Problem parameters
    nb_q = biorbd_model.nbQ()
    nb_qdot = biorbd_model.nbQdot()
    nb_tau = biorbd_model.nbGeneralizedTorque()

    torque_min, torque_max, torque_init = -1000, 1000, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Lagrange.TRACK_STATE, weight=200, states_idx=range(nb_q), target=q_ref)
    objective_functions.add(track_sum_contact_forces_flatfootR_forefootL,
    objective_functions.add(track_sum_contact_forces,
                            grf=grf_ref,
                            custom_type=Objective.Lagrange,
                            instant=Instant.ALL,
                            weight=0.01,
                            quadratic=True,)
    objective_functions.add(track_sum_contact_moments,
                            CoP=CoP,
                            M_ref=M_ref,
                            custom_type=Objective.Lagrange,
                            instant=Instant.ALL,
                            weight=0.01,
                            quadratic=True,)

    # Dynamics
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.TORQUE_DRIVEN_WITH_CONTACT)

    # Constraints
    constraints = ConstraintList()
    constraints.add(
        Constraint.CONTACT_FORCE_INEQUALITY,
        direction="GREATER_THAN",
        instant=Instant.ALL,
        contact_force_idx=(1, 2, 5, 7, 10),
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
    init_x = np.zeros((nb_q + nb_qdot, nb_shooting + 1))
    init_x[:nb_q, :] = q_ref
    init_x[nb_q:nb_q + nb_qdot, :] = qdot_ref
    x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)

    u_init = InitialGuessList()
    init_u = np.zeros((nb_tau, nb_shooting))
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
        nb_threads=nb_threads,
    )

if __name__ == "__main__":
    # Define the problem --- Model path
    model = biorbd.Model("../../ModelesS2M/Marche_saine/2legs/2legs_18dof_0contact.bioMod")
    biorbd_model = (
        biorbd.Model("../../ModelesS2M/Marche_saine/2legs/2legs_18dof_flatfootR.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/2legs/2legs_18dof_forefootR.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/2legs/2legs_18dof_forefootR_HeelL.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/2legs/2legs_18dof_forefootR_flatfootL.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/2legs/2legs_18dof_flatfootL.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/2legs/2legs_18dof_forefootL.bioMod"),
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

    n_p = 7
    ocp = prepare_ocp(
        biorbd_model=biorbd_model[n_p],
        final_time=phase_time[n_p],
        nb_shooting=number_shooting_points[n_p],
        markers_ref=markers_ref[n_p],
        grf_ref=grf_ref[n_p],
        q_ref=q_ref[n_p],
        qdot_ref=qdot_ref[n_p],
        nb_threads=4,
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
    t = np.linspace(0, phase_time[n_p], number_shooting_points[n_p] + 1)

    # --- Plot q --- #
    q_name = []
    for s in range(biorbd_model[n_p].nbSegment()):
        seg_name = biorbd_model[n_p].segment(s).name().to_string()
        for d in range(biorbd_model[n_p].segment(s).nbDof()):
            dof_name = biorbd_model[n_p].segment(s).nameDof(d).to_string()
            q_name.append(seg_name + "_" + dof_name)

    figure, axes = plt.subplots(3, 6)
    axes = axes.flatten()
    for i in range(nb_q):
        axes[i].plot(t, q[i, :], 'r-')
        axes[i].plot(t, q_ref[n_p][i, :], 'b--')
        axes[i].set_title(q_name[i])
    plt.legend(['simulated', 'reference'])

    # --- Plot torques --- #
    figure, axes = plt.subplots(3, 6)
    axes = axes.flatten()
    for i in range(nb_tau):
        axes[i].plot(t, tau[i, :], 'r-')
        axes[i].set_title(q_name[i])
    plt.show()

    # --- Plot grf --- #
    nb_shooting = number_shooting_points[n_p] #total number of shooting points
    labels_forces = ['Heel_r_X', 'Heel_r_Y', 'Heel_r_Z',
                     'Meta_1_r_X', 'Meta_1_r_Y', 'Meta_1_r_Z',
                     'Meta_5_r_X', 'Meta_5_r_Y', 'Meta_5_r_Z',
                     'Heel_l_X', 'Heel_l_Y', 'Heel_l_Z',
                     'Meta_1_l_X', 'Meta_1_l_Y', 'Meta_1_l_Z',
                     'Meta_5_l_X', 'Meta_5_l_Y', 'Meta_5_l_Z',
                     ]
    forces = {}     # dictionary for forces
    for label in labels_forces:
        forces[label] = np.zeros(nb_shooting + 1)

    cn = biorbd_model[n_p].contactNames()
    for n in range(number_shooting_points[n_p] + 1):
        forces_sim = ocp.nlp[0].contact_forces_func(np.concatenate([q[:, n], q_dot[:, n]]), tau[:, n], 0)
        for i, c in enumerate(cn):
            if c.to_string() in forces:
                forces[c.to_string()][n] = forces_sim[i]  # put corresponding forces in dictionnary

    # PLOT EACH FORCES
    figure, axes = plt.subplots(3, 3)
    axes = axes.flatten()
    for i in range(9):
        axes[i].scatter(t, forces[labels_forces[i]], color='r', s=3)
        axes[i].plot(t, forces[labels_forces[i]], 'r-', alpha=0.5)
        if (i == 2) or (i == 5) or (i == 8):
            axes[i].plot([t[0], t[-1]], [0, 0], 'k--')
        axes[i].set_title(labels_forces[i])

    figure, axes = plt.subplots(3, 3)
    axes = axes.flatten()
    for i in range(9):
        axes[i].scatter(t, forces[labels_forces[9+i]], color='r', s=3)
        axes[i].plot(t, forces[labels_forces[9+i]], 'r-', alpha=0.5)
        if (i == 2) or (i == 5) or (i == 8):
            axes[i].plot([t[0], t[-1]], [0, 0], 'k--')
        axes[i].set_title(labels_forces[9+i])

    # PLOT SUM FORCES VS PLATEFORME FORCES
    figure, axes = plt.subplots(1, 3)
    axes = axes.flatten()
    coord_label = ['X', 'Y', 'Z']
    for i in range(3):
        axes[i].plot(t,
                     forces[f"Heel_r_{coord_label[i]}"]
                     + forces[f"Meta_1_r_{coord_label[i]}"]
                     + forces[f"Meta_5_r_{coord_label[i]}"],
                     'r-')
        axes[i].plot(t, grf_ref[n_p][0][i, :], 'b--')
        axes[i].scatter(t, forces[f"Heel_r_{coord_label[i]}"]
                     + forces[f"Meta_1_r_{coord_label[i]}"]
                     + forces[f"Meta_5_r_{coord_label[i]}"],
                        color='r', s=3)
        axes[i].scatter(t, grf_ref[n_p][0][i, :], color='b', s=3)
        axes[i].set_title("Forces in " + coord_label[i] + " R")
    plt.legend(["simulation", "reference"])

    figure, axes = plt.subplots(1, 3)
    axes = axes.flatten()
    coord_label = ['X', 'Y', 'Z']
    for i in range(3):
        axes[i].plot(t,
                     forces[f"Heel_l_{coord_label[i]}"]
                     + forces[f"Meta_1_l_{coord_label[i]}"]
                     + forces[f"Meta_5_l_{coord_label[i]}"],
                     'r-')
        axes[i].plot(t, grf_ref[n_p][1][i, :], 'b--')
        axes[i].scatter(t, forces[f"Heel_l_{coord_label[i]}"]
                     + forces[f"Meta_1_l_{coord_label[i]}"]
                     + forces[f"Meta_5_l_{coord_label[i]}"],
                        color='r', s=3)
        axes[i].scatter(t, grf_ref[n_p][1][i, :], color='b', s=3)
        axes[i].set_title("Forces in " + coord_label[i] + " L")
    plt.legend(["simulation", "reference"])


    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
    result.graphs()