import numpy as np
from casadi import dot, Function, vertcat, MX, mtimes, nlpsol, mmax
import biorbd
import bioviz
from matplotlib import pyplot as plt
from Marche_BiorbdOptim.LoadData import Data_to_track
from Marche_BiorbdOptim.contact.Affichage_resultats import Affichage

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
    Node,
    ConstraintList,
    Constraint,
    StateTransitionList,
    StateTransition,
    ParameterList,
    Solver,
)

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
    activation_min, activation_max, activation_init = 1e-3, 0.8, 0.1

    # Add objective functions
    objective_functions = ObjectiveList()
    for p in range(nb_phases):
        objective_functions.add(Objective.Lagrange.TRACK_STATE, weight=10000, index=range(nb_q), target=q_ref[p], phase=p, quadratic=True)
        objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=1, index=range(6, nb_tau), phase=p, quadratic=True)
        # objective_functions.add(Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=10, phase=p, quadratic=True)
        objective_functions.add(minimize_activation,
                                custom_type=Objective.Lagrange,
                                node=Node.ALL,
                                weight=10,
                                power=6,
                                quadratic=False,
                                phase=p)
        # objective_functions.add(minimize_max_activation,
        #                         custom_type=Objective.Lagrange,
        #                         node=Node.ALL,
        #                         weight=10,
        #                         quadratic=True,
        #                         phase=p)
        objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE_DERIVATIVE, weight=0.1, phase=p, quadratic=True)

    # --- track contact forces for the stance phase ---
    for p in range(nb_phases - 1):
        objective_functions.add(track_sum_contact_forces, # track contact forces
                                grf=grf_ref[p],
                                custom_type=Objective.Lagrange,
                                node=Node.ALL,
                                weight=0.1,
                                quadratic=True,
                                phase=p)

    for p in range(1, nb_phases - 1):
        objective_functions.add(track_sum_contact_moments,
                                CoP=CoP[p],
                                M_ref=M_ref[p],
                                custom_type=Objective.Lagrange,
                                node=Node.ALL,
                                weight=0.1,
                                quadratic=True,
                                phase=p)

    # Dynamics
    dynamics = DynamicsTypeList()
    for p in range(nb_phases - 1):
        dynamics.add(DynamicsType.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN_WITH_CONTACT, phase=p)
    dynamics.add(DynamicsType.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN, phase=3)

    # Constraints
    constraints = ConstraintList()
    constraints.add( # null speed for the first phase --> non sliding contact point
        Constraint.TRACK_MARKERS_VELOCITY,
        node=Node.START,
        index=26,
        phase=0,
    )
    # --- phase flatfoot ---
    constraints.add( # positive vertical forces
        Constraint.CONTACT_FORCE,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.ALL,
        contact_force_idx=(1, 2, 5),
        phase=1,
    )
    constraints.add( # non slipping y
        Constraint.NON_SLIPPING,
        node=Node.ALL,
        normal_component_idx=(1, 2, 5),
        tangential_component_idx=4,
        static_friction_coefficient=0.2,
        phase=1,
    )
    constraints.add( # non slipping x m5
        Constraint.NON_SLIPPING,
        node=Node.ALL,
        normal_component_idx=(2, 5),
        tangential_component_idx=3,
        static_friction_coefficient=0.2,
        phase=1,
    )
    constraints.add( # non slipping x heel
        Constraint.NON_SLIPPING,
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
        Constraint.CONTACT_FORCE,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.ALL,
        contact_force_idx=(1, 4),
        phase=2,
    )
    constraints.add(
        Constraint.NON_SLIPPING,
        node=Node.ALL,
        normal_component_idx=(1, 4),
        tangential_component_idx=3,
        static_friction_coefficient=0.2,
        phase=2,
    )
    # constraints.add( # non slipping x m1
    #     Constraint.NON_SLIPPING,
    #     instant=Instant.ALL,
    #     normal_component_idx=2,
    #     tangential_component_idx=0,
    #     static_friction_coefficient=0.2,
    #     phase=2,
    # )
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
    state_transitions.add(StateTransition.IMPACT, phase_pre_idx=0)
    state_transitions.add(StateTransition.IMPACT, phase_pre_idx=1)

    # Path constraint
    x_bounds = BoundsList()
    u_bounds = BoundsList()
    for p in range(nb_phases):
        x_bounds.add(QAndQDotBounds(biorbd_model[p]))
        u_bounds.add([
            [torque_min] * nb_tau + [activation_min] * nb_mus,
            [torque_max] * nb_tau + [activation_max] * nb_mus,
        ])

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
    save_path = './RES/1leg/cycle/muscles/3_contacts/fort/square/'
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

    # path_previous = './RES/1leg/cycle/muscles/3_contacts/cycle.bo'
    # ocp_previous, sol_previous = ocp.load(path_previous)
    # states_sol, controls_sol = Data.get_data(ocp_previous, sol_previous["x"])
    # q = states_sol["q"]
    # q_dot = states_sol["q_dot"]
    # tau = controls_sol["tau"]
    # activation = controls_sol["muscles"]
    # np.save('./RES/1leg/cycle/muscles/3_contacts/activation', activation)
    # np.save('./RES/1leg/cycle/muscles/3_contacts/q', q)


    # --- Solve the program --- #
    sol = ocp.solve(
        solver=Solver.IPOPT,
        solver_options={
            "ipopt.tol": 1e-4,
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
    save_path = './RES/1leg/cycle/muscles/3_contacts/fort/power_6/'
    ocp.save(sol, save_path + 'cycle.bo')
    np.save(save_path + 'activation', activation)
    np.save(save_path + 'q', q)


    Affichage_resultat = Affichage(ocp, sol, muscles=True, two_leg=False)
    # plot states and controls
    Affichage_resultat.plot_q(q_ref=q_ref, R2=True, RMSE=True)
    Affichage_resultat.plot_tau()
    Affichage_resultat.plot_qdot()
    Affichage_resultat.plot_activation(excitations_ref=excitations_ref)

    # plot Forces
    Affichage_resultat.plot_individual_forces()
    Affichage_resultat.plot_sum_forces(grf_ref=grf_ref)

    # plot CoP and moments
    Affichage_resultat.plot_CoP(CoP_ref=CoP)
    Affichage_resultat.plot_sum_moments(M_ref=M_ref)

    # compute differences
    CoP_ref = np.zeros((3, q.shape[1]))
    n_shoot = 0
    for p in range(nb_phases):
        CoP_ref[:, n_shoot:n_shoot + ocp.nlp[p].ns + 1] = CoP[p]
        n_shoot += ocp.nlp[p].ns
    CoP_simu = Affichage_resultat.compute_CoP()
    mean_diff_CoPx = np.mean(np.sqrt((CoP_ref[0, :56] - CoP_simu[0, :56])**2))
    max_diff_CoPx = np.max(np.sqrt((CoP_ref[0, :56] - CoP_simu[0, :56]) ** 2))
    mean_diff_CoPy = np.mean(np.sqrt((CoP_ref[1, :56] - CoP_simu[1, :56]) ** 2))
    max_diff_CoPy = np.max(np.sqrt((CoP_ref[1, :56] - CoP_simu[1, :56]) ** 2))

    moments_simu = Affichage_resultat.compute_moments_at_CoP()
    forces = Affichage_resultat.compute_individual_forces()
    forces_ref = Affichage_resultat.compute_contact_forces_ref(grf_ref=grf_ref)
    moments_ref = Affichage_resultat.compute_contact_forces_ref(grf_ref=M_ref)
    coords_label=['X', 'Y', 'Z']
    mean_diff_moments = []
    mean_diff_forces = []
    max_diff_moments = []
    max_diff_forces = []
    for i in range(3):
        mean_diff_moments.append(np.mean(np.sqrt((moments_ref[f"force_{coords_label[i]}_R"][:56] - moments_simu[f"moments_{coords_label[i]}_R"][:56])**2)))
        max_diff_moments.append(np.max(np.sqrt((moments_ref[f"force_{coords_label[i]}_R"][:56] - moments_simu[f"moments_{coords_label[i]}_R"][:56]) ** 2)))

        F = forces[f"Heel_r_{coords_label[i]}"][:56] + forces[f"Meta_1_r_{coords_label[i]}"][:56] + forces[f"Meta_5_r_{coords_label[i]}"][:56]
        mean_diff_forces.append(np.mean(np.sqrt((forces_ref[f"force_{coords_label[i]}_R"][:56] - F)**2)))
        max_diff_forces.append(np.max(np.sqrt((forces_ref[f"force_{coords_label[i]}_R"][:56] - F) ** 2)))

    # --- Show results --- #
    ShowResult(ocp, sol).animate()
