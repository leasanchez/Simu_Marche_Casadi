import numpy as np
from casadi import dot, Function, vertcat, MX
from matplotlib import pyplot as plt
import biorbd
from time import time
from Marche_BiorbdOptim.moco.Load_OpemSim_data import get_q, get_grf
import Marche_BiorbdOptim.moco.constraints_dof as Constraints

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
    Instant,
    ConstraintList,
    Constraint,
    StateTransitionList,
    StateTransition,
    Solver,
)



def prepare_ocp(
    biorbd_model, phase_time, number_shooting_points, grf_ref, q_ref, qdot_ref,
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
        objective_functions.add(Objective.Lagrange.TRACK_STATE, weight=100, data_to_track=q_ref[p].T, states_idx=range(nb_q), phase=p)
        objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=10, phase=p)

    # Dynamics
    dynamics = DynamicsTypeList()
    for p in range(nb_phases):
        dynamics.add(DynamicsType.TORQUE_DRIVEN_WITH_CONTACT, phase=p)

    # Constraints
    constraints = ConstraintList()

    # Path constraint
    x_bounds = BoundsList()
    for p in range(nb_phases):
        x_bounds.add(QAndQDotBounds(biorbd_model[p]))

    u_bounds = BoundsList()
    for p in range(nb_phases):
        u_bounds.add([
                [torque_min] * nb_tau,
                [torque_max] * nb_tau,
            ])

    x_init = InitialConditionsList()
    for p in range(nb_phases):
        init_x = np.zeros((nb_q + nb_qdot, number_shooting_points[p] + 1))
        init_x[:nb_q, :] = q_ref[p]
        init_x[nb_q : nb_q + nb_qdot, :] = qdot_ref[p]
        x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)

    u_init = InitialConditionsList()
    for p in range(nb_phases):
        init_u = np.zeros((nb_tau, number_shooting_points[p]))
        u_init.add(init_u, interpolation=InterpolationType.EACH_FRAME)


    # State Transitions
    state_transitions = StateTransitionList()
    state_transitions.add(StateTransition.IMPACT, phase_pre_idx=0)
    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        number_shooting_points,
        phase_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        # state_transitions=state_transitions,
    )


if __name__ == "__main__":
    # Define the problem
    # Model path
    biorbd_model = (
        biorbd.Model("../../ModelesS2M/Open_Sim/Multiphase/subject_walk_armless_test_1.bioMod"),
        biorbd.Model("../../ModelesS2M/Open_Sim/Multiphase/subject_walk_armless_test_2.bioMod"),
        biorbd.Model("../../ModelesS2M/Open_Sim/Multiphase/subject_walk_armless_test_3.bioMod"),
        biorbd.Model("../../ModelesS2M/Open_Sim/Multiphase/subject_walk_armless_test_4.bioMod"),
    )

    # Problem parameters
    number_shooting_points = [6, 9, 9, 10]
    # phase_time = [0.1008, 0.252, 0.4032, 0.5712]
    phase_time = [0.1008, 0.1512, 0.1512, 0.168]
    nb_q = biorbd_model[0].nbQ()
    nb_qdot = biorbd_model[0].nbQdot()
    nb_tau = biorbd_model[0].nbGeneralizedTorque()

    # Generate data from file
    t_init = 0.81
    t_end = 1.65
    final_time = t_end - t_init
    nb_shooting = 50

    [Q_ref, Qdot_ref, Qddot_ref] = get_q(t_init, t_end, final_time, nb_q, nb_shooting)
    GRF_ref = get_grf(t_init, t_end, final_time, nb_shooting)
    t = np.linspace(0, final_time, nb_shooting + 1)

    # q_ref = (Q_ref[:, :7], Q_ref[:, 6:16], Q_ref[:, 15:25], Q_ref[:, 24:35])
    # qdot_ref = (Qdot_ref[:, :7], Qdot_ref[:, 6:16], Qdot_ref[:, 15:25], Qdot_ref[:, 24:35])

    q_ref = (Q_ref[:, :7])
    qdot_ref = (Qdot_ref[:, :7])
    number_shooting_points = [6]
    phase_time = [0.1008]

    # plt.plot(t, GRF_ref[1, :])
    # plt.plot(t, GRF_ref[7, :])
    # plt.legend(["right", "left"])

    # Track these data
    biorbd_model = (
        biorbd.Model("../../ModelesS2M/Open_Sim/Multiphase/subject_walk_armless_test_1.bioMod"),
        # biorbd.Model("../../ModelesS2M/Open_Sim/Multiphase/subject_walk_armless_test_2.bioMod"),
        # biorbd.Model("../../ModelesS2M/Open_Sim/Multiphase/subject_walk_armless_test_3.bioMod"),
        # biorbd.Model("../../ModelesS2M/Open_Sim/Multiphase/subject_walk_armless_test_4.bioMod"),
    )

    ocp = prepare_ocp(
        biorbd_model=biorbd_model,
        phase_time=phase_time,
        number_shooting_points=number_shooting_points,
        grf_ref=GRF_ref,
        q_ref=q_ref,
        qdot_ref=qdot_ref,
    )

    # --- Solve the program --- #
    tic = time()
    sol = ocp.solve(
        solver=Solver.IPOPT,
        solver_options={
            "ipopt.tol": 1e-2,
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
    params = params_sol[ocp.nlp[0]["p"].name()]

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
    result.graphs()
