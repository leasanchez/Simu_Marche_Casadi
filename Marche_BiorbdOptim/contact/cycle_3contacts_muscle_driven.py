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
    biorbd_model, final_time, nb_shooting, markers_ref, grf_ref, q_ref, qdot_ref, M_ref, CoP, excitations_ref,
):

    # Problem parameters
    nb_phases = len(biorbd_model)
    nb_q = biorbd_model[0].nbQ()
    nb_qdot = biorbd_model[0].nbQdot()
    nb_tau = biorbd_model[0].nbGeneralizedTorque()
    nb_mus = biorbd_model[0].nbMuscleTotal()

    torque_min, torque_max, torque_init = -1000, 1000, 0
    activation_min, activation_max, activation_init = 1e-3, 1, 0.1

    # Add objective functions
    objective_functions = ObjectiveList()
    for p in range(nb_phases):
        objective_functions.add(Objective.Lagrange.TRACK_STATE, weight=1000, states_idx=range(nb_q), target=q_ref[p], phase=p)
        objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=0.01, controls_idx=range(6, nb_tau), phase=p)
        # objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE_DERIVATIVE, weight=0.01, phase=p)
        objective_functions.add(Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=0.001, phase=p)

    # track grf
    # --- contact talon ---
    objective_functions.add(track_sum_contact_forces_contact_talon,
                            grf=grf_ref[0],
                            custom_type=Objective.Lagrange,
                            instant=Instant.ALL,
                            weight=0.01,
                            quadratic=True,
                            phase=0)

    # --- flatfoot ---
    objective_functions.add(track_sum_contact_forces_flatfoot,
                            grf=grf_ref[1],
                            custom_type=Objective.Lagrange,
                            instant=Instant.ALL,
                            weight=0.01,
                            quadratic=True,
                            phase=1)
    objective_functions.add(track_sum_moments_flatfoot,
                            CoP=CoP[1],
                            M_ref=M_ref[1],
                            custom_type=Objective.Lagrange,
                            instant=Instant.ALL,
                            weight=0.0001,
                            quadratic=True,
                            phase=1)

    # --- forefoot ---
    objective_functions.add(track_sum_contact_forces_forefoot,
                            grf=grf_ref[2],
                            custom_type=Objective.Lagrange,
                            instant=Instant.ALL,
                            weight=0.01,
                            quadratic=True,
                            phase=2)
    objective_functions.add(track_sum_moments_forefoot,
                            CoP=CoP[2],
                            M_ref=M_ref[2],
                            custom_type=Objective.Lagrange,
                            instant=Instant.ALL,
                            weight=0.0001,
                            quadratic=True,
                            phase=2)

    # Dynamics
    dynamics = DynamicsTypeList()
    for p in range(nb_phases - 1):
        dynamics.add(DynamicsType.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN_WITH_CONTACT, phase=p)
    dynamics.add(DynamicsType.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN, phase=3)

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

    # Initial guess
    x_init = InitialGuessList()
    u_init = InitialGuessList()
    n_shoot=0
    for p in range(nb_phases):
        init_x = np.zeros((nb_q + nb_qdot, nb_shooting[p] + 1))
        init_x[:nb_q, :] = q_ref[p] #np.load('./RES/1leg/3phases/TM1TM2CFI12_100/q.npy')[:, n_shoot:n_shoot + nb_shooting[p] + 1] #
        init_x[nb_q:nb_q + nb_qdot, :] = qdot_ref[p] #np.load('./RES/1leg/3phases/TM1TM2CFI12_100/q_dot.npy')[:,  n_shoot:n_shoot + nb_shooting[p] + 1] #qdot_ref[p]
        x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)

        init_u = np.zeros((nb_tau + nb_mus, number_shooting_points[p])) #np.load('./RES/1leg/3phases/TM1TM2CFI12_100/tau.npy')[:,  n_shoot:n_shoot + nb_shooting[p]] # np.zeros((nb_tau, nb_shooting[p]))
        init_u[nb_tau:, :] = excitations_ref[p][:, :-1]
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
    activation = controls_sol["muscles"]

    # --- Save results ---
    save_path = './RES/1leg/cycle/min_torque/muscles/cycle.bo'
    ocp.save(sol, save_path)

    # --- Affichage ---
    Affichage_resultat = Affichage(ocp, sol, muscles=False, two_leg=False)
    mean_diff_q = Affichage_resultat.compute_mean_difference(q, q_ref)
    idx_max_q, max_diff_q = Affichage_resultat.compute_max_difference(q, q_ref)
    R2 = Affichage_resultat.compute_R2(q, q_ref)
    forces = Affichage_resultat.compute_individual_forces()
    Affichage_resultat.plot_q(q_ref)
    Affichage_resultat.plot_tau()
    Affichage_resultat.plot_individual_forces()
    Affichage_resultat.plot_sum_forces(grf_ref)
    Affichage_resultat.plot_sum_moments(CoP, M_ref)

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
    result.graphs()
