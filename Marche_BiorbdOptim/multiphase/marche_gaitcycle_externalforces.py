import numpy as np
from casadi import dot, Function, MX
import biorbd
from time import time
from matplotlib import pyplot as plt
from Marche_BiorbdOptim.LoadData import Data_to_track

from biorbd_optim import (
    OptimalControlProgram,
    DynamicsTypeOption,
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
    ParameterList,
    Instant,
    ConstraintList,
    Constraint,
    PlotType,
    Solver,
    Simulate,
    InitialConditionsOption,
)


def get_last_contact_forces(ocp, nlp, t, x, u, p, data_to_track=()):
    force = nlp["contact_forces_func"](x[-1], u[-1], p)
    val = force - data_to_track[t[-1], :]
    return dot(val, val)


def get_muscles_first_node(ocp, nlp, t, x, u, p):
    activation = x[0][2 * nlp["nbQ"] :]
    excitation = u[0][nlp["nbQ"] :]
    val = activation - excitation
    return val


def modify_isometric_force(biorbd_model, value, fiso_init):
    n_muscle = 0
    for nGrp in range(biorbd_model.nbMuscleGroups()):
        for nMus in range(biorbd_model.muscleGroup(nGrp).nbMuscles()):
            biorbd_model.muscleGroup(nGrp).muscle(nMus).characteristics().setForceIsoMax(
                value[n_muscle] * fiso_init[n_muscle]
            )
            n_muscle += 1


def prepare_ocp(
    biorbd_model, final_time, nb_shooting, markers_ref, excitation_ref, q_ref, qdot_ref, Fext, Mext, fiso_init, nb_threads,
):
    # Problem parameters
    nb_q = biorbd_model.nbQ()
    nb_qdot = biorbd_model.nbQdot()
    nb_mus = biorbd_model.nbMuscleTotal()
    nb_tau = biorbd_model.nbGeneralizedTorque()

    torque_min, torque_max, torque_init = -1000, 1000, 0
    activation_min, activation_max, activation_init = 0, 1, 0.1

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=0.1, controls_idx=range(6, nb_q), phase=0)
    objective_functions.add(Objective.Lagrange.TRACK_MUSCLES_CONTROL, weight=0.0001, target=excitation_ref, phase=0)
    # objective_functions.add(Objective.Lagrange.TRACK_MARKERS, weight=500, data_to_track=markers_ref, phase=0)
    objective_functions.add(Objective.Lagrange.TRACK_STATE, weight=500, states_idx=range(nb_q), target=q_ref, phase=0)

    # Dynamics
    dynamics = DynamicsTypeOption(DynamicsType.MUSCLE_EXCITATIONS_AND_TORQUE_DRIVEN, phase=0)

    # Constraints
    constraints = ConstraintList()
    constraints.add(get_muscles_first_node, instant=Instant.START)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(QAndQDotBounds(biorbd_model))
    x_bounds[0].concatenate(
        Bounds([activation_min] * nb_mus, [activation_max] * nb_mus)
    )

    u_bounds = BoundsList()
    u_bounds.add([
            [torque_min] * nb_tau + [activation_min] * nb_mus,
            [torque_max] * nb_tau + [activation_max] * nb_mus,
        ])

    # Initial guess
    x_init = InitialConditionsList()
    init_x = np.zeros((nb_q + nb_qdot + nb_mus, nb_shooting + 1))
    init_x[:nb_q, :] = q_ref
    init_x[nb_q: 2*nb_q, :] = qdot_ref
    init_x[-biorbd_model.nbMuscleTotal():, :] = excitation_ref
    x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)

    u_init = InitialConditionsList()
    init_u = np.zeros((nb_tau + nb_mus, nb_shooting))
    init_u[1, :] = np.repeat(-500, nb_shooting)
    init_u[-biorbd_model.nbMuscleTotal():, :] = excitation_ref[:, :-1]
    u_init.add(init_u, interpolation=InterpolationType.EACH_FRAME)

    # Define the parameter to optimize
    parameters = ParameterList()
    bound_length = Bounds(
        min_bound=np.repeat(0.2, nb_mus), max_bound=np.repeat(5, nb_mus), interpolation=InterpolationType.CONSTANT
    )
    parameters.add(
        parameter_name="force_isometric",  # The name of the parameter
        function=modify_isometric_force,  # The function that modifies the biorbd model
        initial_guess=InitialConditions(np.repeat(1, nb_mus)),  # The initial guess
        bounds=bound_length,  # The bounds
        size=nb_mus, # The number of elements this particular parameter vector has
        fiso_init=fiso_init,
   )

    # external forces
    external_forces = []
    fext = np.zeros((6, nb_shooting))
    fext[:3, :]=Mext
    fext[3:, :]=Fext
    external_forces.append(fext)

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
        external_forces=external_forces,
        parameters=parameters,
    )


if __name__ == "__main__":
    # Define the problem
    biorbd_model = biorbd.Model("../../ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact_deGroote_3d_externalforces.bioMod")
    number_shooting_points = [25, 25]

    # Problem parameters
    nb_q = biorbd_model.nbQ()
    nb_qdot = biorbd_model.nbQdot()
    nb_mus = biorbd_model.nbMuscleTotal()
    nb_tau = biorbd_model.nbGeneralizedTorque()
    nb_markers = biorbd_model.nbMarkers()

    # Generate data from file
    markers_ref = np.zeros((3, 26, sum(number_shooting_points) + 1))
    q_ref = np.zeros((nb_q, sum(number_shooting_points) + 1))
    qdot_ref = np.zeros((nb_q, sum(number_shooting_points) + 1))
    excitation_ref = np.zeros((nb_mus, sum(number_shooting_points) + 1))

    Data_to_track = Data_to_track(name_subject="normal01", multiple_contact=False)
    [T, T_stance, T_swing] = Data_to_track.GetTime()
    phase_time = [T_stance, T_swing]
    phase_name = ["stance", "swing"]

    grf_ref = Data_to_track.load_data_GRF(biorbd_model, T_stance, number_shooting_points[0])
    start=0
    for i in range(len(phase_time)):
        markers_ref[:, :, start:start+number_shooting_points[i] + 1]=Data_to_track.load_data_markers(
            biorbd_model, phase_time[i], number_shooting_points[i], phase_name[i])
        q_ref[:, start:start+number_shooting_points[i] + 1]=Data_to_track.load_q_kalman(biorbd_model, phase_time[i], number_shooting_points[i], phase_name[i])
        qdot_ref[:, start:start+number_shooting_points[i] + 1]=Data_to_track.load_qdot_kalman(biorbd_model, phase_time[i], number_shooting_points[i], phase_name[i])
        emg_ref=Data_to_track.load_data_emg(biorbd_model, phase_time[i], number_shooting_points[i], phase_name[i])
        excitation_ref[:, start:start+number_shooting_points[i] + 1]=Data_to_track.load_muscularExcitation(emg_ref)
        start=number_shooting_points[i]

    CoP = Data_to_track.load_data_CoP(biorbd_model, T_stance, number_shooting_points[0])
    M_CoP = Data_to_track.load_data_Moment_at_CoP(biorbd_model, T_stance, number_shooting_points[0])
    M_ref = Data_to_track.load_data_Moment(biorbd_model, T_stance, number_shooting_points[0])

    symbolic_q = MX.sym("x", nb_q, 1)
    markers_func=Function(
        "ForwardKin",
        [symbolic_q],
        [biorbd_model.markers(symbolic_q)],
        ["q"],
        ["marker_pos"],
        ).expand()

    markers_pos = np.zeros((3, nb_markers, number_shooting_points[0] + 1))
    for i in range(number_shooting_points[0] + 1):
        markers_pos[:, :, i]=markers_func(q_ref[:, i])

    Mext = np.zeros((3, sum(number_shooting_points)))
    Fext = np.zeros((3, sum(number_shooting_points)))
    for i in range(number_shooting_points[0]):
        pos = CoP[:, i] - markers_pos[:, 19, i]
        Mext[:, i]=np.cross(pos, grf_ref[:, i]) + M_CoP[:, i]
        Fext[:, i]=grf_ref[:, i]

    # Get initial isometric forces
    fiso_init = []
    n_muscle = 0
    for nGrp in range(biorbd_model.nbMuscleGroups()):
        for nMus in range(biorbd_model.muscleGroup(nGrp).nbMuscles()):
            fiso_init.append(biorbd_model.muscleGroup(nGrp).muscle(nMus).characteristics().forceIsoMax().to_mx())

    # Track these data
    ocp = prepare_ocp(
        biorbd_model=biorbd_model,
        final_time=T,
        nb_shooting=sum(number_shooting_points),
        markers_ref=markers_ref,
        excitation_ref=excitation_ref,
        q_ref=q_ref,
        qdot_ref=qdot_ref,
        Fext=Fext,
        Mext=Mext,
        fiso_init=fiso_init,
        nb_threads=4,
    )

    # U_init_sim = InitialConditionsList()
    # U_init_sim.add([0]*nb_tau + [0]*nb_mus, interpolation=InterpolationType.CONSTANT)
    # sim = Simulate.from_controls_and_initial_states(ocp, ocp.original_values["X_init"][0], U_init_sim[0], single_shoot=True)
    # states_sim, controls_sim = Data.get_data(ocp, sim["x"])
    # ShowResult(ocp, sim).graphs()
    # ShowResult(ocp, sim).animate()

    # --- Solve the program --- #
    tic = time()
    sol = ocp.solve(
        solver=Solver.IPOPT,
        solver_options={
            "ipopt.tol": 1e-3,
            "ipopt.max_iter": 3000,
            "ipopt.hessian_approximation": "exact",
            "ipopt.limited_memory_max_history": 50,
            "ipopt.linear_solver": "ma57",
        },
        show_online_optim=False,
    )

    toc = time() - tic
    print(f"Time to solve : {toc}sec")

    # --- Get Results --- #
    states_sol, controls_sol = Data.get_data(ocp, sol["x"])
    q = states_sol["q"]
    q_dot = states_sol["q_dot"]
    activations = states_sol["muscles"]
    tau = controls_sol["tau"]
    excitations = controls_sol["muscles"]

    # --- Save Results --- #
    np.save("./RES/external_forces/excitations", excitations)
    np.save("./RES/external_forces/activations", activations)
    np.save("./RES/external_forces/tau", tau)
    np.save("./RES/external_forces/q_dot", q_dot)
    np.save("./RES/external_forces/q", q)

    # --- Show results --- #
    ShowResult(ocp, sol).animate()
    ShowResult(ocp, sol).graphs()

