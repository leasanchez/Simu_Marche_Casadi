import numpy as np
from casadi import dot, Function, vertcat, MX, mtimes, nlpsol
import biorbd
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
    StateTransitionList,
    StateTransition,
    Solver,
)

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
        val = vertcat(val, (heel[1] * forces[1] + meta1[1] * forces[2] + meta5[1] * forces[5]) - M_ref[0, n])
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
        val = vertcat(val, (meta1[1] * forces[1] + meta5[1] * forces[4]) - M_ref[0, n])
        val = vertcat(val, (-meta1[0]*forces[1] - meta5[0]*forces[4]) - M_ref[1, n])
        val = vertcat(val, (- meta1[1]*forces[0] + meta5[0]*forces[3] - meta5[1]*forces[2]) - M_ref[2, :])
    return val

def prepare_ocp(
    biorbd_model, final_time, nb_shooting, markers_ref, grf_ref, q_ref, qdot_ref, M_ref, CoP,
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
        # objective_functions.add(Objective.Lagrange.TRACK_STATE, weight=200, states_idx=range(nb_q), target=q_ref[p], phase=p)
        objective_functions.add(Objective.Lagrange.TRACK_MARKERS, weight=500, target=markers_ref[p], phase=p)
        objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE_DERIVATIVE, weight=0.01, phase=p)
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
        dynamics.add(DynamicsType.TORQUE_DRIVEN_WITH_CONTACT)
    dynamics.add(DynamicsType.TORQUE_DRIVEN)

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
    constraints.add( # non slipping y
        Constraint.NON_SLIPPING,
        instant=Instant.ALL,
        normal_component_idx=(1,2,5),
        tangential_component_idx=4,
        static_friction_coefficient=0.2,
        phase=1,
    )
    constraints.add( # non slipping x m5
        Constraint.NON_SLIPPING,
        instant=Instant.ALL,
        normal_component_idx=(2,5),
        tangential_component_idx=3,
        static_friction_coefficient=0.2,
        phase=1,
    )
    constraints.add( # non slipping x heel
        Constraint.NON_SLIPPING,
        instant=Instant.ALL,
        normal_component_idx=1,
        tangential_component_idx=0,
        static_friction_coefficient=0.2,
        phase=1,
    )
    constraints.add( # forces heel at zeros at the end of the phase
        get_last_contact_force_null,
        instant=Instant.ALL,
        contact_name='Heel_r',
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
        Constraint.NON_SLIPPING,
        instant=Instant.ALL,
        normal_component_idx=(1,4),
        tangential_component_idx=3,
        static_friction_coefficient=0.2,
        phase=2,
    )
    constraints.add( # non slipping x m5
        Constraint.NON_SLIPPING,
        instant=Instant.ALL,
        normal_component_idx=4,
        tangential_component_idx=2,
        static_friction_coefficient=0.2,
        phase=2,
    )
    constraints.add( # non slipping x m1
        Constraint.NON_SLIPPING,
        instant=Instant.ALL,
        normal_component_idx=1,
        tangential_component_idx=0,
        static_friction_coefficient=0.2,
        phase=2,
    )
    constraints.add(
        get_last_contact_force_null,
        instant=Instant.ALL,
        contact_name=('Meta_1_r', 'Meta_5_r'),
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
        u_bounds.add([[torque_min] * nb_tau,[torque_max] * nb_tau])

    # Initial guess
    x_init = InitialGuessList()
    u_init = InitialGuessList()
    n_shoot=0
    for p in range(nb_phases):
        init_x = np.zeros((nb_q + nb_qdot, nb_shooting[p] + 1))
        init_x[:nb_q, :] = np.load('./RES/1leg/cycle/q.npy')[:, n_shoot:n_shoot + nb_shooting[p] + 1]
        init_x[nb_q:nb_q + nb_qdot, :] = np.load('./RES/1leg/cycle/q_dot.npy')[:,  n_shoot:n_shoot + nb_shooting[p] + 1]
        x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)

        init_u=np.load('./RES/1leg/cycle/tau.npy')[:,  n_shoot:n_shoot + nb_shooting[p]]
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
    # EMG_ref = Data_to_track.load_data_emg(number_shooting_points)
    # excitations_ref = []
    # for p in range(len(EMG_ref)):
    #     excitations_ref.append(Data_to_track.load_muscularExcitation(EMG_ref[p]))

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

    # get previous solution
    # ocp_previous, sol_previous = ocp.load('./RES/1leg/cycle/min_torque/cycle.bo')
    # states_sol, controls_sol = Data.get_data(ocp_previous, sol_previous["x"])
    # q = states_sol["q"]
    # q_dot = states_sol["q_dot"]
    # tau = controls_sol["tau"]

    # # show BiorbdViz
    # # ShowResult(ocp_previous, sol_previous).animate()
    #
    # Affichage_resultat = Affichage(ocp_previous, sol_previous, muscles=False, two_leg=False)
    # # plot states and controls
    # Affichage_resultat.plot_q(q_ref=q_ref)
    # Affichage_resultat.plot_tau()
    # Affichage_resultat.plot_qdot()
    #
    # # plot Forces
    # Affichage_resultat.plot_individual_forces()
    # Affichage_resultat.plot_sum_forces(grf_ref=grf_ref)
    #
    # # plot CoP and moments
    # Affichage_resultat.plot_CoP(CoP_ref=CoP)
    # Affichage_resultat.plot_sum_moments(M_ref=M_ref)
    #
    # # compute differences
    # CoP_ref = np.zeros((3, q.shape[1]))
    # n_shoot = 0
    # for p in range(nb_phases):
    #     CoP_ref[:, n_shoot:n_shoot + ocp_previous.nlp[p].ns + 1] = CoP[p]
    #     n_shoot += ocp_previous.nlp[p].ns
    # CoP_simu = Affichage_resultat.compute_CoP()
    # mean_diff_CoPx = np.mean(np.sqrt((CoP_ref[0, :56] - CoP_simu[0, :56])**2))
    # max_diff_CoPx = np.max(np.sqrt((CoP_ref[0, :56] - CoP_simu[0, :56]) ** 2))
    # mean_diff_CoPy = np.mean(np.sqrt((CoP_ref[1, :56] - CoP_simu[1, :56]) ** 2))
    # max_diff_CoPy = np.max(np.sqrt((CoP_ref[1, :56] - CoP_simu[1, :56]) ** 2))
    #
    # moments_simu = Affichage_resultat.compute_moments_at_CoP()
    # moments_ref = Affichage_resultat.compute_contact_forces_ref(grf_ref=M_ref)
    # coords_label=['X', 'Y', 'Z']
    # mean_diff_moments = []
    # max_diff_moments = []
    # for i in range(3):
    #     mean_diff_moments.append(np.mean(np.sqrt((moments_ref[f"force_{coords_label[i]}_R"][:56] - moments_simu[f"moments_{coords_label[i]}_R"][:56])**2)))
    #     max_diff_moments.append(np.max(np.sqrt((moments_ref[f"force_{coords_label[i]}_R"][:56] - moments_simu[f"moments_{coords_label[i]}_R"][:56]) ** 2)))

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

    # --- Save results ---
    save_path = './RES/1leg/cycle/min_torque/cycle.bo'
    ocp.save(sol, save_path)

    Affichage_resultat = Affichage(ocp, sol, muscles=False, two_leg=False)
    # plot states and controls
    Affichage_resultat.plot_q(q_ref=q_ref)
    Affichage_resultat.plot_tau()
    Affichage_resultat.plot_qdot()

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
        CoP_ref[:, n_shoot:n_shoot + ocp_previous.nlp[p].ns + 1] = CoP[p]
        n_shoot += ocp_previous.nlp[p].ns
    CoP_simu = Affichage_resultat.compute_CoP()
    mean_diff_CoPx = np.mean(np.sqrt((CoP_ref[0, :56] - CoP_simu[0, :56])**2))
    max_diff_CoPx = np.max(np.sqrt((CoP_ref[0, :56] - CoP_simu[0, :56]) ** 2))
    mean_diff_CoPy = np.mean(np.sqrt((CoP_ref[1, :56] - CoP_simu[1, :56]) ** 2))
    max_diff_CoPy = np.max(np.sqrt((CoP_ref[1, :56] - CoP_simu[1, :56]) ** 2))

    moments_simu = Affichage_resultat.compute_moments_at_CoP()
    moments_ref = Affichage_resultat.compute_contact_forces_ref(grf_ref=M_ref)
    coords_label=['X', 'Y', 'Z']
    mean_diff_moments = []
    max_diff_moments = []
    for i in range(3):
        mean_diff_moments.append(np.mean(np.sqrt((moments_ref[f"force_{coords_label[i]}_R"][:56] - moments_simu[f"moments_{coords_label[i]}_R"][:56])**2)))
        max_diff_moments.append(np.max(np.sqrt((moments_ref[f"force_{coords_label[i]}_R"][:56] - moments_simu[f"moments_{coords_label[i]}_R"][:56]) ** 2)))

    # --- Show results --- #
    ShowResult(ocp, sol).animate()
    result.animate()
    result.graphs()
