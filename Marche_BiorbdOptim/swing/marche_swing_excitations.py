import numpy as np
from casadi import MX, Function, vertcat
from matplotlib import pyplot as plt
import biorbd
from time import time
from Marche_BiorbdOptim.LoadData import Data_to_track

from biorbd_optim import (
    OptimalControlProgram,
    DynamicsTypeList,
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
    ParametersList,
    Instant,
    ConstraintList,
    Constraint,
    PlotType,
)


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
    biorbd_model, final_time, nb_shooting, markers_ref, q_ref, qdot_ref, excitations_ref, fiso_init, nb_threads,
):
    # Problem parameters
    nb_q = biorbd_model.nbQ()
    nb_qdot = biorbd_model.nbQdot()
    nb_mus = biorbd_model.nbMuscleTotal()
    nb_tau = biorbd_model.nbGeneralizedTorque()

    torque_min, torque_max, torque_init = -5000, 5000, 0
    activation_min, activation_max, activation_init = 0, 1, 0.1

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=1, controls_idx=range(6, nb_q), phase=0)
    objective_functions.add(Objective.Lagrange.TRACK_MUSCLES_CONTROL, weight=0.0001, data_to_track=excitation_ref.T, phase=0)
    objective_functions.add(Objective.Lagrange.TRACK_MARKERS, weight=500, data_to_track=markers_ref, phase=0)

    # Dynamics
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.MUSCLE_EXCITATIONS_AND_TORQUE_DRIVEN, phase=0)

    # Constraints
    constraints = ConstraintList()
    constraints.add(Constraint.CUSTOM, custom_function=get_muscles_first_node, instant=Instant.START)

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
    init_x[nb_q:nb_q + nb_qdot, :] = qdot_ref
    init_x[-biorbd_model.nbMuscleTotal() :, :] = excitation_ref
    x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)

    u_init = InitialConditionsList()
    init_u = np.zeros((nb_tau + nb_mus, nb_shooting))
    init_u[1, :] = np.repeat(-500, n_shooting_points)
    init_u[-biorbd_model.nbMuscleTotal():, :] = excitation_ref[:, :-1]
    u_init.add(init_u, interpolation=InterpolationType.EACH_FRAME)

    # Define the parameter to optimize
    parameters = ParametersList()
    bound_length = Bounds(
        min_bound=np.repeat(0.2, nb_mus), max_bound=np.repeat(5, nb_mus), interpolation=InterpolationType.CONSTANT
    )
    parameters.add(
        parameter_name="force_isometric",  # The name of the parameter
        function=modify_isometric_force,  # The function that modifies the biorbd model
        initial_guess=InitialConditions(np.repeat(1, nb_mus)),  # The initial guess
        bounds=bound_length,  # The bounds
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
    # Define the problem
    biorbd_model = biorbd.Model("../../ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact_deGroote_3d.bioMod")
    model_q = biorbd.Model("../../ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact_deGroote.bioMod")
    n_shooting_points = 25
    Gaitphase = "swing"

    # Generate data from file
    Data_to_track = Data_to_track(name_subject="equincocont01")
    [T, T_stance, T_swing] = Data_to_track.GetTime()
    final_time = T_swing

    markers_ref = Data_to_track.load_data_markers(
        biorbd_model, T_stance, n_shooting_points, "swing"
    )  # get markers position
    q_ref = Data_to_track.load_q_kalman(biorbd_model, T_stance, n_shooting_points, "swing")  # get q from kalman
    qdot_ref = Data_to_track.load_qdot_kalman(biorbd_model, T_stance, n_shooting_points, "swing")
    emg_ref = Data_to_track.load_data_emg(biorbd_model, T_stance, n_shooting_points, "swing")  # get emg
    excitation_ref = Data_to_track.load_muscularExcitation(emg_ref)

    # Get initial isometric forces
    fiso_init = []
    n_muscle = 0
    for nGrp in range(biorbd_model.nbMuscleGroups()):
        for nMus in range(biorbd_model.muscleGroup(nGrp).nbMuscles()):
            fiso_init.append(biorbd_model.muscleGroup(nGrp).muscle(nMus).characteristics().forceIsoMax().to_mx())

    # Track these data
    ocp = prepare_ocp(
        biorbd_model, final_time, n_shooting_points, markers_ref, q_ref, qdot_ref, excitation_ref, fiso_init, nb_threads=4,
    )
    # --- Add plot kalman --- #
    ocp.add_plot("q", lambda x, u: q_ref, PlotType.STEP, axes_idx=[0, 1, 5, 8, 9, 11])

    # --- Solve the program --- #
    tic = time()
    sol = ocp.solve(
        solver="ipopt",
        solver_options={
            "ipopt.tol": 1e-3,
            "ipopt.max_iter": 5000,
            "ipopt.hessian_approximation": "exact",
            "ipopt.limited_memory_max_history": 50,
            "ipopt.linear_solver": "ma57",
        },
        show_online_optim=False,
    )
    toc = time() - tic
    print(f"Time to solve : {toc}sec")

    # --- Get Results --- #
    states_sol, controls_sol, params_sol = Data.get_data(ocp, sol["x"], get_parameters=True)
    q = states_sol["q"]
    q_dot = states_sol["q_dot"]
    activations = states_sol["muscles"]
    tau = controls_sol["tau"]
    excitations = controls_sol["muscles"]
    params = params_sol[ocp.nlp[0]["p"].name()]

    # --- Save Results --- #
    np.save("./RES/equincocont03/excitations", excitations)
    np.save("./RES/equincocont03/activations", activations)
    np.save("./RES/equincocont03/tau", tau)
    np.save("./RES/equincocont03/q_dot", q_dot)
    np.save("./RES/equincocont03/q", q)
    np.save("./RES/equincocont03/params", params)

    # --- Show results --- #
    ShowResult(ocp, sol).animate()
t = np.linspace(0, final_time, n_shooting_points + 1)
q_name = []
for s in range(biorbd_model.nbSegment()):
    seg_name = biorbd_model.segment(s).name().to_string()
    for d in range(biorbd_model.segment(s).nbDof()):
        dof_name = biorbd_model.segment(s).nameDof(d).to_string()
        q_name.append(seg_name + "_" + dof_name)

figure, axes = plt.subplots(4, 3, sharex=True)
axes = axes.flatten()
for i in range(biorbd_model.nbQ()):
    axes[i].plot(t, q[i, :], color="tab:red", linestyle="-", linewidth=1)
    axes[i].plot(t, Q_ref[i, :], color="k", linestyle="--", linewidth=0.7)
    axes[i].set_title(q_name[i])
    # axes[i].set_ylim([np.max(q[i, :]), np.min(q[i, :])])
    axes[i].grid(color="k", linestyle="--", linewidth=0.5)
