import numpy as np
from casadi import dot, Function, MX
import biorbd
from time import time
from BiorbdViz import BiorbdViz
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
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
    Solver,
    Simulate,
    InitialConditionsOption,
)

def compute_tau_from_muscles(model, states, controls):
    muscles_states = biorbd.VecBiorbdMuscleState(model.nbMuscleTotal())
    muscles_excitation = controls[model.nbQ():]
    muscles_activations = states[model.nbQ() + model.nbQdot():]

    for k in range(model.nbMuscleTotal()):
        muscles_states[k].setExcitation(muscles_excitation[k])
        muscles_states[k].setActivation(muscles_activations[k])

    muscles_tau = model.muscularJointTorque(muscles_states, states[:model.nbQ()], states[model.nbQ():model.nbQ() + model.nbQdot()]).to_mx()
    tau = muscles_tau + controls[:model.nbQ()]
    return tau

def inverse_dynamics(model, Q, Qdot, Qddot, Fext):
    sv = biorbd.VecBiorbdSpatialVector()
    sv.append(biorbd.SpatialVector(Fext))
    tau = model.InverseDynamics(Q, Qdot, Qddot, sv).to_mx()
    return tau


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

def forward_dynamique(model, q, qdot, tau, fext):
    sv = biorbd.VecBiorbdSpatialVector()
    sv.append(biorbd.SpatialVector(fext))
    qddot=model.ForwardDynamics(q, qdot, tau, sv).to_mx()
    return qddot

def prepare_ocp(
    biorbd_model, final_time, nb_shooting, markers_ref, excitation_ref, q_ref, qdot_ref, Fext, Mext, fiso_init, nb_threads
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
    objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=1, controls_idx=range(6, nb_q), phase=0)
    # objective_functions.add(Objective.Lagrange.TRACK_MUSCLES_CONTROL, weight=0.01, target=excitation_ref, phase=0)
    # objective_functions.add(Objective.Lagrange.TRACK_MARKERS, weight=500, markers_idx=26, data_to_track=cop_pos, phase=0)
    objective_functions.add(Objective.Lagrange.TRACK_STATE, weight=100, states_idx=range(nb_q), target=q_ref, phase=0)

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
    init_x[nb_q:(nb_q + nb_qdot), :] = qdot_ref
    init_x[-biorbd_model.nbMuscleTotal() :, :] = excitation_ref
    x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)

    u_init = InitialConditionsList()
    init_u = np.zeros((nb_tau + nb_mus, nb_shooting))
    init_u[1, :] = np.zeros(nb_shooting) + 500
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
    fext[:3, :]=Mext[:, :-1]
    fext[3:, :]=Fext[:, :-1]
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
        # external_forces=external_forces,
        # parameters=parameters,
    )


if __name__ == "__main__":
    # Define the problem
    biorbd_model = biorbd.Model("../../ModelesS2M/ANsWER_externalforces_simple.bioMod")
    n_shooting_points = 25
    Gaitphase = "stance"

    # Problem parameters
    nb_q = biorbd_model.nbQ()
    nb_qdot = biorbd_model.nbQdot()
    nb_mus = biorbd_model.nbMuscleTotal()
    nb_tau = biorbd_model.nbGeneralizedTorque()
    nb_markers = biorbd_model.nbMarkers()

    # Generate data from file
    Data_to_track = Data_to_track(name_subject="normal01")
    [T, T_stance, T_swing] = Data_to_track.GetTime()
    final_time = T_stance

    grf_ref = Data_to_track.load_data_GRF(biorbd_model, T_stance, n_shooting_points)  # get ground reaction forces
    markers_ref = Data_to_track.load_data_markers(
        biorbd_model, T_stance, n_shooting_points, "stance"
    )  # get markers position
    q_ref = Data_to_track.load_q_kalman(biorbd_model, T_stance, n_shooting_points, "stance")  # get q from kalman
    qdot_ref = Data_to_track.load_qdot_kalman(biorbd_model, T_stance, n_shooting_points, "stance")
    qddot_ref = Data_to_track.load_qddot_kalman(biorbd_model, T_stance, n_shooting_points, "stance")
    emg_ref = Data_to_track.load_data_emg(biorbd_model, T_stance, n_shooting_points, "stance")  # get emg
    excitation_ref = Data_to_track.load_muscularExcitation(emg_ref)
    CoP = Data_to_track.load_data_CoP(biorbd_model, T_stance, n_shooting_points)
    M_CoP = Data_to_track.load_data_Moment_at_CoP(biorbd_model, T_stance, n_shooting_points)
    M_ref = Data_to_track.load_data_Moment(biorbd_model, T_stance, n_shooting_points)

    symbolic_q = MX.sym("x", nb_q, 1)
    markers_func=Function(
        "ForwardKin",
        [symbolic_q],
        [biorbd_model.markers(symbolic_q)],
        ["q"],
        ["marker_pos"],
        ).expand()

    markers_pos = np.zeros((3, nb_markers, n_shooting_points + 1))
    for i in range(n_shooting_points + 1):
        markers_pos[:, :, i]=markers_func(q_ref[:, i])

    Mext = np.zeros((3, n_shooting_points + 1))
    for i in range(n_shooting_points):
        pos = CoP[:, i] - markers_pos[:, 19, i]
        Mext[:, i]=np.cross(pos, grf_ref[:, i]) + M_CoP[:, i]

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter3D(CoP[0, :], CoP[1, :], CoP[2, :], color='red')
    ax.scatter3D(markers_pos[0, 19, :], markers_pos[1, 19, :], markers_pos[2, 19, :], color='green')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.figure("Forces")
    plt.plot(grf_ref.T)
    plt.xlabel("shooting points")
    plt.ylabel("force (N)")
    plt.title("Forces")
    plt.legend(["x", "y", "z"])

    plt.figure("Moments")
    plt.plot(Mext.T)
    plt.xlabel("shooting points")
    plt.ylabel("moments (Nm)")
    plt.title("Moments")
    plt.legend(["x", "y", "z"])
    plt.show()

    # b = BiorbdViz(loaded_model=biorbd_model)
    # b.load_movement(q_ref)

    # Get initial isometric forces
    fiso_init = []
    n_muscle = 0
    for nGrp in range(biorbd_model.nbMuscleGroups()):
        for nMus in range(biorbd_model.muscleGroup(nGrp).nbMuscles()):
            fiso_init.append(biorbd_model.muscleGroup(nGrp).muscle(nMus).characteristics().forceIsoMax().to_mx())
    #
    # symbolic_states = MX.sym("x", nb_q + nb_q + nb_mus, 1)
    # symbolic_controls = MX.sym("u", nb_tau + nb_mus, 1)
    # muscles_torque_func = Function(
    #     "muscles_torque",
    #     [symbolic_states, symbolic_controls],
    #     [compute_tau_from_muscles(biorbd_model, symbolic_states, symbolic_controls)],
    #     ["x", "u"],
    #     ["tau"],
    # ).expand()
    #
    # Tau = np.zeros((nb_tau, n_shooting_points + 1))
    # for k in range(n_shooting_points + 1):
    #     state=np.concatenate([q_ref[:, k], qdot_ref[:, k], excitation_ref[:, k]])
    #     control=np.concatenate([np.zeros(nb_tau), excitation_ref[:, k]])
    #     Tau[:, k] = np.array(muscles_torque_func(state, control)).squeeze()
    #
    # q_name = []
    # for s in range(biorbd_model.nbSegment()):
    #     seg_name = biorbd_model.segment(s).name().to_string()
    #     for d in range(biorbd_model.segment(s).nbDof()):
    #         dof_name = biorbd_model.segment(s).nameDof(d).to_string()
    #         q_name.append(seg_name + "_" + dof_name)
    #
    # figure, axes = plt.subplots(5, 6)
    # axes = axes.flatten()
    # for i in range(nb_tau):
    #     axes[i].plot(Tau[i, :], color="tab:red")
    #     axes[i].set_title("Tau " + q_name[i])
    # plt.show()

    # Inverse dynamics
    symbolic_q = MX.sym("q", nb_q, 1)
    symbolic_dq = MX.sym("dq", nb_q, 1)
    symbolic_ddq = MX.sym("ddq", nb_q, 1)
    symbolic_fext = MX.sym("fext", 6, 1)
    inverse_dynamics_func=Function(
        "InverseDynamics",
        [symbolic_q, symbolic_dq, symbolic_ddq, symbolic_fext],
        [inverse_dynamics(biorbd_model, symbolic_q, symbolic_dq, symbolic_ddq, symbolic_fext)],
        ["q", "dq", "ddq", "fext"],
        ["tau"],
        ).expand()

    tau_iv = np.zeros((nb_q, n_shooting_points + 1))
    for i in range(n_shooting_points + 1):
        Fext = np.concatenate((Mext[:, i], grf_ref[:, i]))
        tau_iv[:, i]=inverse_dynamics_func(q_ref[:, i], qdot_ref[:, i], qddot_ref[:, i], Fext)


    # Track these data
    ocp = prepare_ocp(
        biorbd_model,
        final_time,
        n_shooting_points,
        markers_ref,
        excitation_ref=excitation_ref,
        q_ref=q_ref,
        qdot_ref=qdot_ref,
        Fext=grf_ref,
        Mext=M_CoP,
        fiso_init=fiso_init,
        nb_threads=4,
    )

    sim = Simulate.from_controls_and_initial_states(ocp, ocp.original_values["X_init"][0], ocp.original_values["U_init"][0], single_shoot=True)
    states_sim, controls_sim = Data.get_data(ocp, sim["x"])
    ShowResult(ocp, sim).graphs()
    ShowResult(ocp, sim).animate()

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

