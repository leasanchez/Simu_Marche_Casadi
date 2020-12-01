import numpy as np
from casadi import dot, Function, vertcat, MX, mtimes, nlpsol
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
    ShowResult,
    ObjectiveList,
    Objective,
    InterpolationType,
    Data,
    Instant,
    ConstraintList,
    Constraint,
    Solver,
    Simulate,
    ParameterList,
    InitialGuess,
)

# --- isometric forces ---
def modify_isometric_force(biorbd_model, value, fiso_init):
    n_muscle = 0
    for nGrp in range(biorbd_model.nbMuscleGroups()):
        for nMus in range(biorbd_model.muscleGroup(nGrp).nbMuscles()):
            biorbd_model.muscleGroup(nGrp).muscle(nMus).characteristics().setForceIsoMax(
                value[n_muscle] * fiso_init[n_muscle]
            )
            n_muscle += 1

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
        heel =  markers[26].to_mx() - CoP[:, t[n]]
        meta1 = markers[27].to_mx() - CoP[:, t[n]]
        meta5 = markers[28].to_mx() - CoP[:, t[n]]
        toe =   markers[29].to_mx() - CoP[:, t[n]]

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
    biorbd_model, final_time, nb_shooting, markers_ref, grf_ref, q_ref, qdot_ref, M_ref, CoP, excitations_ref, fiso_init, nb_threads,
):
    # Problem parameters
    nb_q = biorbd_model.nbQ()
    nb_qdot = biorbd_model.nbQdot()
    nb_tau = biorbd_model.nbGeneralizedTorque()
    nb_mus = biorbd_model.nbMuscleTotal()

    torque_min, torque_max, torque_init = -1000, 1000, 0
    activation_min, activation_max, activation_init = 1e-3, 1, 0.1

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Lagrange.TRACK_STATE, weight=100, states_idx=range(nb_q), target=q_ref)
    objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=0.1, controls_idx=range(6, nb_tau))
    objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE_DERIVATIVE, weight=0.1, controls_idx=range(6, nb_tau))
    objective_functions.add(Objective.Lagrange.TRACK_MUSCLES_CONTROL, weight=1, target=excitations_ref[:, :-1])
    objective_functions.add(track_sum_contact_forces,
                            grf=grf_ref,
                            custom_type=Objective.Lagrange,
                            instant=Instant.ALL,
                            weight=0.1,
                            quadratic=True,)
    # objective_functions.add(track_sum_contact_moments,
    #                         CoP=CoP,
    #                         M_ref=M_ref,
    #                         custom_type=Objective.Lagrange,
    #                         instant=Instant.ALL,
    #                         weight=0.01,
    #                         quadratic=True,)

    # Dynamics
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN_WITH_CONTACT)
    # dynamics.add(DynamicsType.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN)

    # Constraints
    constraints = ConstraintList()
    # constraints.add( # null speed for the first phase --> non sliding contact point
    #     Constraint.TRACK_MARKERS_VELOCITY,
    #     instant=Instant.START,
    #     markers_idx=26,
    # )
    constraints.add( # positive vertical forces
        Constraint.CONTACT_FORCE,
        min_bound=0,
        max_bound=np.inf,
        instant=Instant.ALL,
        contact_force_idx=(1, 2, 5),
    )
    constraints.add( # non slipping y
        Constraint.NON_SLIPPING,
        instant=Instant.ALL,
        normal_component_idx=(1, 2, 5),
        tangential_component_idx=4,
        static_friction_coefficient=0.2,
    )
    # constraints.add( # non slipping x m5
    #     Constraint.NON_SLIPPING,
    #     instant=Instant.ALL,
    #     normal_component_idx=(2, 5),
    #     tangential_component_idx=3,
    #     static_friction_coefficient=0.2,
    # )
    # constraints.add( # non slipping x heel
    #     Constraint.NON_SLIPPING,
    #     instant=Instant.ALL,
    #     normal_component_idx=1,
    #     tangential_component_idx=0,
    #     static_friction_coefficient=0.2,
    # )

    constraints.add( # forces heel at zeros at the end of the phase
        get_last_contact_force_null,
        instant=Instant.ALL,
        contact_name='Heel_r',
    )

    # Path constraint
    x_bounds = BoundsList()
    u_bounds = BoundsList()
    x_bounds.add(QAndQDotBounds(biorbd_model))
    u_bounds.add([
        [torque_min] * nb_tau + [activation_min] * nb_mus,
        [torque_max] * nb_tau + [activation_max] * nb_mus,
    ])

    # Initial guess
    x_init = InitialGuessList()
    init_x = np.zeros((nb_q + nb_qdot, nb_shooting + 1))
    init_x[:nb_q, :] = q_ref
    init_x[nb_q:nb_q + nb_qdot, :] = qdot_ref
    x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)

    u_init = InitialGuessList()
    init_u = np.zeros((nb_tau + nb_mus, nb_shooting))
    init_u[nb_tau:, :] = excitations_ref[:, :-1]
    u_init.add(init_u, interpolation=InterpolationType.EACH_FRAME)

    # Define the parameter to optimize
    # Give the parameter some min and max bounds
    parameters = ParameterList()
    bound_params = Bounds(
        min_bound=np.repeat(0.2, nb_mus), max_bound=np.repeat(5, nb_mus), interpolation=InterpolationType.CONSTANT
    )
    initial_prams = InitialGuess(np.repeat(1, nb_mus))
    parameters.add(
        parameter_name="force_isometric",  # The name of the parameter
        function=modify_isometric_force,  # The function that modifies the biorbd model
        initial_guess=initial_prams,  # The initial guess
        bounds=bound_params,  # The bounds
        size=nb_mus,  # The number of elements this particular parameter vector has
        fiso_init=fiso_init,
    )

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
        parameters=parameters,
    )

if __name__ == "__main__":
    biorbd_model = (
        biorbd.Model("../../ModelesS2M/Marche_saine/Florent_1leg_12dof_heel.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/Florent_1leg_12dof_flatfoot.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/Florent_1leg_12dof_forefoot.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/Florent_1leg_12dof_0contact.bioMod")
    )

    # Problem parameters
    nb_q = biorbd_model[0].nbQ()
    nb_qdot = biorbd_model[0].nbQdot()
    nb_tau = biorbd_model[0].nbGeneralizedTorque()
    nb_mus = biorbd_model[0].nbMuscleTotal()
    nb_phases = len(biorbd_model)

    # Generate data from file
    Data_to_track = Data_to_track("normal01", model=biorbd_model[0], multiple_contact=True, two_leg=False)
    phase_time = Data_to_track.GetTime()
    number_shooting_points = []
    for time in phase_time:
        number_shooting_points.append(int(time / 0.01))
    markers_ref = Data_to_track.load_data_markers(number_shooting_points) # get markers positions
    q_ref = Data_to_track.load_q_kalman(number_shooting_points) # get joint positions
    qdot_ref = Data_to_track.load_qdot_kalman(number_shooting_points) # get joint velocities
    qddot_ref = Data_to_track.load_qdot_kalman(number_shooting_points) # get joint accelerations
    grf_ref = Data_to_track.load_data_GRF(number_shooting_points)  # get ground reaction forces
    CoP = Data_to_track.load_data_CoP(number_shooting_points)
    M_ref = Data_to_track.load_data_Moment_at_CoP(number_shooting_points)
    EMG_ref = Data_to_track.load_data_emg(number_shooting_points)
    excitations_ref = []
    for p in range(nb_phases):
        excitations_ref.append(Data_to_track.load_muscularExcitation(EMG_ref[p]))

    # Get initial isometric forces
    fiso_init = []
    n_muscle = 0
    for nGrp in range(biorbd_model[0].nbMuscleGroups()):
        for nMus in range(biorbd_model[0].muscleGroup(nGrp).nbMuscles()):
            fiso_init.append(biorbd_model[0].muscleGroup(nGrp).muscle(nMus).characteristics().forceIsoMax().to_mx())

    n_p = 1
    ocp = prepare_ocp(
        biorbd_model=biorbd_model[n_p],
        final_time=phase_time[n_p],
        nb_shooting=number_shooting_points[n_p],
        markers_ref=markers_ref[n_p],
        grf_ref=grf_ref[n_p],
        q_ref=q_ref[n_p],
        qdot_ref=qdot_ref[n_p],
        M_ref=M_ref[n_p],
        CoP=CoP[n_p],
        excitations_ref=excitations_ref[n_p],
        fiso_init=fiso_init,
        nb_threads=4,
    )

    # path_previous = './RES/1leg/flatfoot/muscles/cycle.bo'
    # ocp_previous, sol_previous = ocp.load(path_previous)
    # states_previous, controls_previous = Data.get_data(ocp_previous, sol_previous)
    # q_previous, q_dot_previous, tau_previous, muscles_previous = (
    #     states_previous["q"],
    #     states_previous["q_dot"],
    #     controls_previous["tau"],
    #     controls_previous["muscles"],
    # )
    # ShowResult(ocp_previous, sol_previous).graphs()

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
    states, controls, params = Data.get_data(ocp, sol, get_parameters=True)
    q, q_dot, tau, muscles = (
        states["q"],
        states["q_dot"],
        controls["tau"],
        controls["muscles"],
    )
    ShowResult(ocp, sol).graphs()

    # --- Save results ---
    save_path = './RES/1leg/flatfoot/muscles/param/'
    ocp.save(sol, save_path + "cycle.bo")
    np.save(save_path + "q", q)
    np.save(save_path + "q_dot", q_dot)
    np.save(save_path + "tau", tau)
    np.save(save_path + "muscles", muscles)
    np.save(save_path + "params", params)

    Affichage_resultat = Affichage(ocp, sol, muscles=True, two_leg=False)
    Affichage_resultat.plot_sum_forces(grf_ref=grf_ref[n_p])

    # plot states and controls
    Affichage_resultat.plot_q(q_ref=q_ref[n_p])
    Affichage_resultat.plot_tau()
    Affichage_resultat.plot_qdot()
    Affichage_resultat.plot_activation(excitations_ref=excitations_ref[n_p])
    mean_diff_q = Affichage_resultat.compute_mean_difference(x=q, x_ref=q_ref[n_p])
    idx, max_diff_q = Affichage_resultat.compute_max_difference(x=q, x_ref=q_ref[n_p])
    R2_q = Affichage_resultat.compute_R2(x=q, x_ref=q_ref[n_p])

    # plot Forces
    Affichage_resultat.plot_individual_forces()
    Affichage_resultat.plot_sum_forces(grf_ref=grf_ref[n_p])

    # plot CoP and moments
    Affichage_resultat.plot_CoP(CoP_ref=CoP[n_p])
    Affichage_resultat.plot_sum_moments(M_ref=M_ref[n_p])

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
    result.graphs()