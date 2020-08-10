import numpy as np
import biorbd
from time import time
from .Load_OpenSim_data import get_q, get_grf

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
    biorbd_model, final_time, nb_shooting, q_ref, qdot_ref, grf_ref, nb_threads,
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

    # Constraints
    x = np.array(
        [0, 0.174533, 0.349066, 0.523599, 0.698132, 0.872665, 1.0472, 1.22173, 1.39626, 1.5708, 1.74533, 1.91986,
         2.0944])
    constraints = ()

    # External forces
    external_forces = np.zeros((6, 2, nb_shooting)) # 1 torseur par jambe
    external_forces[:, 0, :] = grf_ref[:6, :]  # right leg
    external_forces[:, 1, :] = grf_ref[6:, :]  # left leg

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
    init_x[-nb_mus :, :] = np.zeros((nb_mus, nb_shooting + 1)) + 0.1
    x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)

    u_init = InitialConditionsList()
    init_u = np.zeros((nb_tau + nb_mus, nb_shooting))
    init_u[1, :] = np.repeat(-500, nb_shooting)
    init_u[-biorbd_model.nbMuscleTotal():, :] = np.zeros((nb_mus, nb_shooting)) + 0.1
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
        constraints=constraints,
        nb_threads=nb_threads,
        external_forces=external_forces,
    )


if __name__ == "__main__":
    # Define the problem
    biorbd_model = biorbd.Model("../../ModelesS2M/Open_Sim/subject_walk_armless_test_no_muscle_fext.bioMod")
    t_init = 0.81
    t_end = 1.65
    final_time = t_end - t_init
    nb_shooting = 50

    # Generate data from file OpenSim
    [Q_ref, Qdot_ref, Qddot_ref] = get_q(t_init, t_end, final_time, biorbd_model.nbQ(), nb_shooting)
    GRF_ref = get_grf(t_init, t_end, final_time, nb_shooting)

    # Track these data
    ocp = prepare_ocp(
        biorbd_model,
        final_time=final_time,
        nb_shooting=nb_shooting,
        q_ref=Q_ref,
        qdot_ref=Qdot_ref,
        grf_ref=GRF_ref,
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
