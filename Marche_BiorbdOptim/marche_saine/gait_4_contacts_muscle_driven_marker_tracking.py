import numpy as np
from casadi import dot, Function, vertcat, MX, mtimes, nlpsol, mmax
import biorbd
import bioviz
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Marche_BiorbdOptim.LoadData import Data_to_track
from Marche_BiorbdOptim.marche_saine.Affichage_resultats import Affichage

from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    BoundsList,
    QAndQDotBounds,
    InitialGuessList,
    ShowResult,
    ObjectiveList,
    ObjectiveFcn,
    InterpolationType,
    Data,
    Node,
    ConstraintList,
    ConstraintFcn,
    StateTransitionList,
    StateTransitionFcn,
    Solver,
)

def plot_foot(ax, markers, idx_node, color="black", alpha=1.0):
    ax.plot([markers[0, 19, idx_node], markers[0, 21, idx_node]],
            [markers[1, 19, idx_node], markers[1, 21, idx_node]],
            [markers[2, 19, idx_node], markers[2, 21, idx_node]],
            c=color, alpha=alpha, linestyle='dashed')
    ax.plot([markers[0, 21, idx_node], markers[0, 20, idx_node]],
            [markers[1, 21, idx_node], markers[1, 20, idx_node]],
            [markers[2, 21, idx_node], markers[2, 20, idx_node]],
            c=color, alpha=alpha, linestyle='dashed')
    ax.plot([markers[0, 20, idx_node], markers[0, 22, idx_node]],
            [markers[1, 20, idx_node], markers[1, 22, idx_node]],
            [markers[2, 20, idx_node], markers[2, 22, idx_node]],
            c=color, alpha=alpha, linestyle='dashed')
    ax.plot([markers[0, 22, idx_node], markers[0, 24, idx_node]],
            [markers[1, 22, idx_node], markers[1, 24, idx_node]],
            [markers[2, 22, idx_node], markers[2, 24, idx_node]],
            c=color, alpha=alpha, linestyle='dashed')
    ax.plot([markers[0, 24, idx_node], markers[0, 25, idx_node]],
            [markers[1, 24, idx_node], markers[1, 25, idx_node]],
            [markers[2, 24, idx_node], markers[2, 25, idx_node]],
            c=color, alpha=alpha, linestyle='dashed')
    ax.plot([markers[0, 25, idx_node], markers[0, 19, idx_node]],
            [markers[1, 25, idx_node], markers[1, 19, idx_node]],
            [markers[2, 25, idx_node], markers[2, 19, idx_node]],
            c=color, alpha=alpha, linestyle='dashed')
    ax.plot([markers[0, 19, idx_node], np.mean([markers[0, 17, idx_node], markers[0, 18, idx_node]])],
            [markers[1, 19, idx_node], np.mean([markers[1, 17, idx_node], markers[1, 18, idx_node]])],
            [markers[2, 19, idx_node], 0.2],
            c=color, alpha=alpha, linestyle='dashed')

# modified isometric forces in parameters
def modify_isometric_force(biorbd_model, value, fiso_init):
    n_muscle = 0
    for nGrp in range(biorbd_model.nbMuscleGroups()):
        for nMus in range(biorbd_model.muscleGroup(nGrp).nbMuscles()):
            biorbd_model.muscleGroup(nGrp).muscle(nMus).characteristics().setForceIsoMax(
                value[n_muscle] * fiso_init[n_muscle]
            )
            n_muscle += 1

# --- minimize activation ---
def minimize_activation(ocp, nlp, t, x, u, p, power):
    nb_tau = nlp.model.nbGeneralizedTorque()
    val = []
    for control in u:
        val = vertcat(val, control[nb_tau:]**power)
    return val

# --- minimize max activation ---
def minimize_max_activation(ocp, nlp, t, x, u, p):
    nb_tau = nlp.model.nbGeneralizedTorque()
    val = []
    for control in u:
        val = vertcat(val, mmax(control[nb_tau:]))
    return val

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
        heel  = markers[-4].to_mx() - CoP[:, t[n]]
        meta1 = markers[-3].to_mx() - CoP[:, t[n]]
        meta5 = markers[-2].to_mx() - CoP[:, t[n]]
        toe   = markers[-1].to_mx() - CoP[:, t[n]]

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
    biorbd_model, final_time, nb_shooting, markers_ref, grf_ref, q_ref, qdot_ref, M_ref, CoP, excitations_ref, nb_threads,
):

    # Problem parameters
    nb_phases = len(biorbd_model)
    nb_q = biorbd_model[0].nbQ()
    nb_qdot = biorbd_model[0].nbQdot()
    nb_tau = biorbd_model[0].nbGeneralizedTorque()
    nb_mus = biorbd_model[0].nbMuscleTotal()

    min_bound, max_bound = 0, np.inf
    torque_min, torque_max, torque_init = -1000, 1000, 0
    activation_min, activation_max, activation_init = 1e-3, 1.0, 0.1

    # Add objective functions
    markers_pelvis = [0,1,2,3]
    markers_anat = [4,9,10,11,12,17,18]
    markers_tissus = [5,6,7,8,13,14,15,16]
    markers_pied = [19,20,21,22,23,24,25]
    objective_functions = ObjectiveList()
    for p in range(nb_phases):
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, weight=1, index=range(nb_q), target=q_ref[p], phase=p, quadratic=True)
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_MARKERS, weight=1000, index=markers_anat, target=markers_ref[p][:, markers_anat, :],
                                phase=p, quadratic=True)
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_MARKERS, weight=100000, index=markers_pelvis, target=markers_ref[p][:, markers_pelvis, :],
                                phase=p, quadratic=True)
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_MARKERS, weight=100000, index=markers_pied, target=markers_ref[p][:, markers_pied, :],
                                phase=p, quadratic=True)
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_MARKERS, weight=100, index=markers_tissus, target=markers_ref[p][:, markers_tissus, :],
                                phase=p, quadratic=True)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=0.001, index=(10), phase=p, quadratic=True)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=1, index=(6, 7, 8, 9, 11), phase=p, quadratic=True)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=10, phase=p, quadratic=True)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE_DERIVATIVE, weight=0.1, phase=p, quadratic=True)

    # --- track contact forces for the stance phase ---
    for p in range(nb_phases - 1):
        objective_functions.add(track_sum_contact_forces, # track contact forces
                                grf=grf_ref[p],
                                custom_type=ObjectiveFcn.Lagrange,
                                node=Node.ALL,
                                weight=0.1,
                                quadratic=True,
                                phase=p)

    for p in range(1, nb_phases - 1):
        objective_functions.add(track_sum_contact_moments,
                                CoP=CoP[p],
                                M_ref=M_ref[p],
                                custom_type=ObjectiveFcn.Lagrange,
                                node=Node.ALL,
                                weight=0.01,
                                quadratic=True,
                                phase=p)

    # Dynamics
    dynamics = DynamicsList()
    for p in range(nb_phases - 1):
        dynamics.add(DynamicsFcn.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN_WITH_CONTACT, phase=p)
    dynamics.add(DynamicsFcn.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN, phase=3)

    # Constraints
    constraints = ConstraintList()
    constraints.add( # null speed for the first phase --> non sliding contact point
        ConstraintFcn.TRACK_MARKERS_VELOCITY,
        node=Node.START,
        index=26,
        phase=0,
    )
    # --- phase flatfoot ---
    constraints.add( # positive vertical forces
        ConstraintFcn.CONTACT_FORCE,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.ALL,
        contact_force_idx=(1, 2, 5),
        phase=1,
    )
    constraints.add( # non slipping y
        ConstraintFcn.NON_SLIPPING,
        node=Node.ALL,
        normal_component_idx=(1, 2, 5),
        tangential_component_idx=4,
        static_friction_coefficient=0.2,
        phase=1,
    )
    constraints.add( # non slipping x m5
        ConstraintFcn.NON_SLIPPING,
        node=Node.ALL,
        normal_component_idx=(2, 5),
        tangential_component_idx=3,
        static_friction_coefficient=0.2,
        phase=1,
    )
    constraints.add( # non slipping x heel
        ConstraintFcn.NON_SLIPPING,
        node=Node.ALL,
        normal_component_idx=1,
        tangential_component_idx=0,
        static_friction_coefficient=0.2,
        phase=1,
    )

    constraints.add( # forces heel at zeros at the end of the phase
        get_last_contact_force_null,
        node=Node.ALL,
        contact_name='Heel_r',
        phase=1,
    )

    # --- phase forefoot ---
    constraints.add( # positive vertical forces
        ConstraintFcn.CONTACT_FORCE,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.ALL,
        contact_force_idx=(2, 4, 5),
        phase=2,
    )
    constraints.add(
        ConstraintFcn.NON_SLIPPING,
        node=Node.ALL,
        normal_component_idx=(2, 4, 5),
        tangential_component_idx=1,
        static_friction_coefficient=0.2,
        phase=2,
    )
    constraints.add( # non slipping x m1
        ConstraintFcn.NON_SLIPPING,
        node=Node.ALL,
        normal_component_idx=2,
        tangential_component_idx=0,
        static_friction_coefficient=0.2,
        phase=2,
    )
    # # constraints.add( # non slipping x toes
    # #     Constraint.NON_SLIPPING,
    # #     instant=Instant.ALL,
    # #     normal_component_idx=5,
    # #     tangential_component_idx=4,
    # #     static_friction_coefficient=0.2,
    # #     phase=2,
    # # )
    constraints.add(
        get_last_contact_force_null,
        node=Node.ALL,
        contact_name='all',
        phase=2,
    )

    # State Transitions
    state_transitions = StateTransitionList()
    state_transitions.add(StateTransitionFcn.IMPACT, phase_pre_idx=0)
    state_transitions.add(StateTransitionFcn.IMPACT, phase_pre_idx=1)

    # Path constraint
    x_bounds = BoundsList()
    u_bounds = BoundsList()
    for p in range(nb_phases):
        x_bounds.add(bounds=QAndQDotBounds(biorbd_model[p]))
        u_bounds.add(
            [torque_min] * nb_tau + [activation_min] * nb_mus,
            [torque_max] * nb_tau + [activation_max] * nb_mus,
        )

    # # Initial guess
    # x_init = InitialGuessList()
    # u_init = InitialGuessList()
    # n_shoot=0
    # for p in range(nb_phases):
    #     init_x = np.zeros((nb_q + nb_qdot, nb_shooting[p] + 1))
    #     init_x[:nb_q, :] = q_ref[p]
    #     init_x[nb_q:nb_q + nb_qdot, :] = qdot_ref[p]
    #     x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)
    #
    #     init_u = np.zeros((nb_tau + nb_mus, number_shooting_points[p]))
    #     init_u[nb_tau:, :] = excitations_ref[p][:, :-1]
    #     u_init.add(init_u, interpolation=InterpolationType.EACH_FRAME)
    #     n_shoot += nb_shooting[p]

    # Initial guess
    save_path = './RES/1leg/cycle/muscles/4_contacts/markers_tracking/adjusted_weight/'
    x_init = InitialGuessList()
    u_init = InitialGuessList()
    n_shoot=0
    for p in range(nb_phases):
        init_x = np.zeros((nb_q + nb_qdot, nb_shooting[p] + 1))
        init_x[:nb_q, :] = np.load(save_path + "q.npy")[:, n_shoot:n_shoot + nb_shooting[p] + 1]
        init_x[nb_q:nb_q + nb_qdot, :] = np.load(save_path + "q_dot.npy")[:, n_shoot:n_shoot + nb_shooting[p] + 1]
        x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)

        init_u = np.zeros((nb_tau + nb_mus, number_shooting_points[p]))
        init_u[:nb_tau, :] = np.load(save_path + "tau.npy")[:, n_shoot:n_shoot + nb_shooting[p]]
        init_u[nb_tau:, :] = np.load(save_path + "activation.npy")[:, n_shoot:n_shoot + nb_shooting[p]]
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
        nb_threads=nb_threads,
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
    nb_phases = len(biorbd_model)
    nb_markers = biorbd_model[0].nbMarkers()

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
    EMG_ref = Data_to_track.load_data_emg(number_shooting_points)
    excitations_ref = []
    for p in range(nb_phases):
        excitations_ref.append(Data_to_track.load_muscularExcitation(EMG_ref[p]))

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
        excitations_ref=excitations_ref,
        nb_threads=4,
    )
    #
    # # --- Get Previous Results --- #
    # path_previous = './RES/1leg/cycle/muscles/4_contacts/markers_tracking/adjusted_weightcycle.bo'
    # ocp_previous, sol_previous = ocp.load(path_previous)
    # states_sol, controls_sol = Data.get_data(ocp_previous, sol_previous["x"])
    # q = states_sol["q"]
    # q_dot = states_sol["q_dot"]
    # tau = controls_sol["tau"]
    # activation = controls_sol["muscles"]
    #
    # # --- Plot Results --- #
    # Affichage_resultat = Affichage(ocp_previous, sol_previous, muscles=True, two_leg=False)
    #
    # # plot states and controls
    # Affichage_resultat.plot_q(q_ref=q_ref, R2=False, RMSE=True)
    # # Affichage_resultat.plot_tau()
    # # Affichage_resultat.plot_qdot()
    # Affichage_resultat.plot_activation(excitations_ref=excitations_ref)
    #
    # # plot Forces
    # Affichage_resultat.plot_individual_forces()
    # Affichage_resultat.plot_sum_forces(grf_ref=grf_ref)
    #
    # # plot CoP and moments
    # Affichage_resultat.plot_CoP(CoP_ref=CoP)
    # Affichage_resultat.plot_sum_moments(M_ref=M_ref)
    #
    # # plot foot and CoP
    # CoP_simu = Affichage_resultat.compute_CoP()
    # grf_simu = Affichage_resultat.compute_sum_forces_simu()
    #
    # symbolic_q = MX.sym("q", nb_q, 1)
    # markers_func = []
    # for m in range(nb_markers):
    #     markers_func.append(Function(
    #         "ForwardKin",
    #         [symbolic_q], [biorbd_model[0].marker(symbolic_q, m).to_mx()],
    #         ["q"],
    #         ["markers"],
    #     ).expand())
    # position_markers = np.zeros((3, nb_markers, q.shape[1]))
    # for n in range(q.shape[1]):
    #     Q = q[:, n]
    #     for m in range(nb_markers):
    #         position_markers[:, m, n:n+1]=markers_func[m](Q)
    #
    # complete_markers_ref = np.zeros((3, nb_markers, q.shape[1]))
    # n_shoot=0
    # for p in range(len(biorbd_model)):
    #     complete_markers_ref[:, :, n_shoot:n_shoot+q_ref[p].shape[1]] = markers_ref[p]
    #     n_shoot+=number_shooting_points[p]
    #
    # # plot markers differences
    # RMSE_markers = np.sqrt((complete_markers_ref - position_markers)**2)
    #
    # markers_name = ["LIAS","LIPS","RIPS","RIAS","RFTC", "R Thigh Top", "R Thigh down", "R Thigh Front", "R Thigh Back",
    #                 "RFLE", "RFME", "RFAX", "RTTC", "R Shank Top", "R Shank Down", "R Shank Front", "R Shank Tibia",
    #                 "RFAL", "RFAM", "FCC", "FM1", "FMP1", "FM2", "FMP2", "FM5", "FMP5"]
    # shooting_nodes = range(position_markers.shape[2])
    #
    # fig, ax = plt.subplots()
    # im, cbar = Affichage_resultat.heatmap(data=RMSE_markers[0, :26, :], row_labels=markers_name, col_labels=shooting_nodes)
    # plt.title("x differences markers")
    #
    # fig, ax = plt.subplots()
    # im, cbar = Affichage_resultat.heatmap(data=RMSE_markers[1, :26, :], row_labels=markers_name, col_labels=shooting_nodes)
    # plt.title("y differences markers")
    #
    # fig, ax = plt.subplots()
    # im, cbar = Affichage_resultat.heatmap(data=RMSE_markers[2, :26, :], row_labels=markers_name, col_labels=shooting_nodes)
    # plt.title("z differences markers")
    #
    # # plot movements
    # Affichage_resultat.plot_stance_phase(markers=position_markers[:, :, :markers_ref[0].shape[2]],
    #                                      CoP=CoP_simu[:, :markers_ref[0].shape[2]],
    #                                      grf=grf_simu[:, :markers_ref[0].shape[2]],
    #                                      markers_ref=markers_ref[0],
    #                                      CoP_ref=CoP[0],
    #                                      grf_ref=grf_ref[0])
    # Affichage_resultat.plot_stance_phase(markers=position_markers[:, :, markers_ref[0].shape[2]:markers_ref[0].shape[2] + markers_ref[1].shape[2]],
    #                                      CoP=CoP_simu[:, markers_ref[0].shape[2]:markers_ref[0].shape[2] + markers_ref[1].shape[2]],
    #                                      grf=grf_simu[:, markers_ref[0].shape[2]:markers_ref[0].shape[2] + markers_ref[1].shape[2]],
    #                                      markers_ref=markers_ref[1],
    #                                      CoP_ref=CoP[1],
    #                                      grf_ref=grf_ref[1])
    # Affichage_resultat.plot_stance_phase(markers=position_markers[:, :, markers_ref[0].shape[2] + markers_ref[1].shape[2]:markers_ref[0].shape[2] + markers_ref[1].shape[2] + markers_ref[2].shape[2]],
    #                                      CoP=CoP_simu[:, markers_ref[0].shape[2] + markers_ref[1].shape[2]:markers_ref[0].shape[2] + markers_ref[1].shape[2] + markers_ref[2].shape[2]],
    #                                      grf=grf_simu[:, markers_ref[0].shape[2] + markers_ref[1].shape[2]:markers_ref[0].shape[2] + markers_ref[1].shape[2] + markers_ref[2].shape[2]],
    #                                      markers_ref=markers_ref[2],
    #                                      CoP_ref=CoP[2],
    #                                      grf_ref=grf_ref[2])
    #
    # # --- Show results --- #
    # ShowResult(ocp_previous, sol_previous).animate()

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
    activation = controls_sol["muscles"]

    # --- Save results ---
    save_path = './RES/1leg/cycle/muscles/4_contacts/markers_tracking/adjusted_weight/'
    ocp.save(sol, save_path + 'cycle.bo')
    np.save(save_path + 'activation', activation)
    np.save(save_path + 'q_dot', q_dot)
    np.save(save_path + 'q', q)
    np.save(save_path + 'tau', tau)

    # --- Show results --- #
    ShowResult(ocp, sol).animate()

    # --- Plot Results --- #
    Affichage_resultat = Affichage(ocp, sol, muscles=True, two_leg=False)
    # plot states and controls
    Affichage_resultat.plot_q(q_ref=q_ref, R2=False, RMSE=True)
    Affichage_resultat.plot_tau()
    Affichage_resultat.plot_qdot()
    Affichage_resultat.plot_activation(excitations_ref=excitations_ref)

    # plot Forces
    Affichage_resultat.plot_individual_forces()
    Affichage_resultat.plot_sum_forces(grf_ref=grf_ref)

    # plot CoP and moments
    Affichage_resultat.plot_CoP(CoP_ref=CoP)
    Affichage_resultat.plot_sum_moments(M_ref=M_ref)