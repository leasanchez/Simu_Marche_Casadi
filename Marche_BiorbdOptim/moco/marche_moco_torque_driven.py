import numpy as np
import biorbd
from time import time
from casadi import vertcat, MX, Function, interp1d
from Marche_BiorbdOptim.moco.Load_OpemSim_data import get_q, get_grf
from Marche_BiorbdOptim.moco.constraints_dof import Constraints

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
    ConstraintList,
    Instant,
)


def tibia_r_Ty(ocp, nlp, t, x, u, p, x_interpol, q_ref_idx, q_target_idx):
    y = np.array([0, 0.000479, 0.000835, 0.001086, 0.001251, 0.001346, 0.001391, 0.001403, 0.0014, 0.0014, 0.001421,
                  0.001481, 0.001599])
    nb_q = nlp["nbQ"]
    # f = interp1d(y, q[q_ref_idx], x_interpol)
    val = []
    for v in x:
        q = v[:nb_q]
        f = interp1d(x_interpol, y)
        val = vertcat(val, q[q_target_idx] - 0.95799999999999996 * f(q[q_ref_idx]))
    return val

def prepare_ocp(
    biorbd_model, final_time, nb_shooting, q_ref, qdot_ref, grf_ref, nb_threads,
):
    # Problem parameters
    nb_q = biorbd_model.nbQ()
    nb_qdot = biorbd_model.nbQdot()
    nb_tau = biorbd_model.nbGeneralizedTorque()

    torque_min, torque_max, torque_init = -1000, 1000, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=1, phase=0)
    objective_functions.add(Objective.Lagrange.TRACK_STATE, weight=100, states_idx=range(nb_q), data_to_track=q_ref, phase=0)

    # Dynamics
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.TORQUE_DRIVEN, phase=0)

    # Constraints
    x = np.array(
        [0, 0.174533, 0.349066, 0.523599, 0.698132, 0.872665, 1.0472, 1.22173, 1.39626, 1.5708, 1.74533, 1.91986,
         2.0944])
    constraints = ConstraintList()
    constraints.add(tibia_r_Ty, instant=Instant.ALL, x_interpol=x, q_ref_idx=11, q_target_idx=9)

    # # External forces
    # external_forces = np.zeros((6, 2, nb_shooting)) # 1 torseur par jambe
    # external_forces[:, 0, :] = grf_ref[:6, :]  # right leg
    # external_forces[:, 1, :] = grf_ref[6:, :]  # left leg

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(QAndQDotBounds(biorbd_model))

    u_bounds = BoundsList()
    u_bounds.add([
            [torque_min] * nb_tau,
            [torque_max] * nb_tau,
        ])

    # Initial guess
    x_init = InitialConditionsList()
    init_x = np.zeros((nb_q + nb_qdot, nb_shooting + 1))
    init_x[:nb_q, :] = q_ref
    init_x[nb_q:nb_q + nb_qdot, :] = qdot_ref
    x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)

    u_init = InitialConditionsList()
    init_u = np.zeros((nb_tau, nb_shooting))
    init_u[1, :] = np.repeat(-500, nb_shooting)
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
    )


if __name__ == "__main__":
    # Define the problem
    biorbd_model = biorbd.Model("../../ModelesS2M/Open_Sim/subject_walk_armless_test_no_muscle.bioMod")
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
    tau = controls_sol["tau"]

    # --- Show results --- #
    ShowResult(ocp, sol).animate()
