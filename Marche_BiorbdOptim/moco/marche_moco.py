import numpy as np
from casadi import dot
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
    Solver,
)


def prepare_ocp(
    biorbd_model, final_time, nb_shooting, excitation_ref, q_ref, qdot_ref, nb_threads,
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
    objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=1, phase=0)
    objective_functions.add(Objective.Lagrange.TRACK_STATE, weight=100, states_idx=range(nb_q), data_to_track=q_ref, phase=0)

    # Dynamics
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.MUSCLE_EXCITATIONS_AND_TORQUE_DRIVEN, phase=0)

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
        constraints=(),
        nb_threads=nb_threads,
    )


if __name__ == "__main__":
    # Define the problem
    biorbd_model = biorbd.Model("../../ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact_deGroote_3d.bioMod")
    n_shooting_points = 25
    Gaitphase = "stance"

    # Generate data from file
    Data_to_track = Data_to_track(name_subject="normal01")
    [T, T_stance, T_swing] = Data_to_track.GetTime()
    final_time = T_stance

    Q_ref = np.zeros((biorbd_model.nbQ(), 51))
    Q_ref[:, :26] = Data_to_track.load_q_kalman(biorbd_model, T_stance, n_shooting_points, "stance")
    Q_ref[:, 25:] = Data_to_track.load_q_kalman(biorbd_model, T_stance, n_shooting_points, "swing")

    Qdot_ref = np.zeros((biorbd_model.nbQ(), 51))
    Qdot_ref[:, :26] = Data_to_track.load_qdot_kalman(biorbd_model, T_stance, n_shooting_points, "stance")
    Qdot_ref[:, 25:] = Data_to_track.load_qdot_kalman(biorbd_model, T_stance, n_shooting_points, "swing")

    EMG_ref = np.zeros((biorbd_model.nbMuscleTotal(), 51))
    emg_ref = Data_to_track.load_data_emg(biorbd_model, T_stance, n_shooting_points, "stance")  # get emg
    EMG_ref[:, :26] = Data_to_track.load_muscularExcitation(emg_ref)
    emg_ref = Data_to_track.load_data_emg(biorbd_model, T_stance, n_shooting_points, "swing")  # get emg
    EMG_ref[:, 25:] = Data_to_track.load_muscularExcitation(emg_ref)

    # Get initial isometric forces
    fiso_init = []
    n_muscle = 0
    for nGrp in range(biorbd_model.nbMuscleGroups()):
        for nMus in range(biorbd_model.muscleGroup(nGrp).nbMuscles()):
            fiso_init.append(biorbd_model.muscleGroup(nGrp).muscle(nMus).characteristics().forceIsoMax().to_mx())

    # Track these data
    ocp = prepare_ocp(
        biorbd_model,
        final_time=T,
        nb_shooting=50,
        excitation_ref=EMG_ref,
        q_ref=Q_ref,
        qdot_ref=Qdot_ref,
        nb_threads=4,
    )

    # --- Solve the program --- #
    tic = time()
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
