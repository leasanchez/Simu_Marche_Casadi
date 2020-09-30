import numpy as np
from casadi import dot, Function, vertcat, MX
from matplotlib import pyplot as plt
import biorbd
from time import time
import Marche_BiorbdOptim.moco.Load_OpenSim_data as Moco
import Marche_BiorbdOptim.moco.constraints_dof as Constraints
from BiorbdViz import BiorbdViz

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
    Simulate,
)



def prepare_ocp(
    biorbd_model, phase_time, number_shooting_points, q_ref, qdot_ref, tau_ref, nb_thread,
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
        q_tracked = np.zeros((2*nb_q, number_shooting_points[p] + 1))
        q_tracked[:nb_q, :] = q_ref[p]
        q_tracked[nb_q:, :] = qdot_ref[p]
        # objective_functions.add(Objective.Lagrange.TRACK_STATE, weight=100, target=q_tracked,  phase=p)
        objective_functions.add(Objective.Lagrange.MINIMIZE_ALL_CONTROLS, weight=1, phase=p)

    # Dynamics
    dynamics = DynamicsTypeList()
    for p in range(nb_phases):
        dynamics.add(DynamicsType.TORQUE_DRIVEN_WITH_CONTACT, phase=p)

    # Constraints
    x = np.array(
        [0, 0.174533, 0.349066, 0.523599, 0.698132, 0.872665, 1.0472, 1.22173, 1.39626, 1.5708, 1.74533, 1.91986,
         2.0944])
    constraints = ConstraintList()
    for p in range(nb_phases):
        # # -- tibia_r --
        # constraints.add(Constraints.tibia_r_Ty, instant=Instant.ALL, x_interpol=x, q_ref_idx=11, q_target_idx=9)
        # constraints.add(Constraints.tibia_r_Tz, instant=Instant.ALL, x_interpol=x, q_ref_idx=11, q_target_idx=10)
        # constraints.add(Constraints.tibia_r_Rz, instant=Instant.ALL, x_interpol=x, q_ref_idx=11, q_target_idx=12)
        # constraints.add(Constraints.tibia_r_Ry, instant=Instant.ALL, x_interpol=x, q_ref_idx=11, q_target_idx=13)
        #
        # # -- patella_r --
        # constraints.add(Constraints.patella_r_Tx, instant=Instant.ALL, x_interpol=x, q_ref_idx=11, q_target_idx=14)
        # constraints.add(Constraints.patella_r_Ty, instant=Instant.ALL, x_interpol=x, q_ref_idx=11, q_target_idx=15)
        # constraints.add(Constraints.patella_r_Rz, instant=Instant.ALL, x_interpol=x, q_ref_idx=11, q_target_idx=16)
        #
        # # -- tibia_l --
        # constraints.add(Constraints.tibia_l_Ty, instant=Instant.ALL, x_interpol=x, q_ref_idx=23, q_target_idx=21)
        # constraints.add(Constraints.tibia_l_Tz, instant=Instant.ALL, x_interpol=x, q_ref_idx=23, q_target_idx=22)
        # constraints.add(Constraints.tibia_l_Rz, instant=Instant.ALL, x_interpol=x, q_ref_idx=23, q_target_idx=24)
        # constraints.add(Constraints.tibia_l_Ry, instant=Instant.ALL, x_interpol=x, q_ref_idx=23, q_target_idx=25)
        #
        # # -- patella_l --
        # constraints.add(Constraints.patella_l_Tx, instant=Instant.ALL, x_interpol=x, q_ref_idx=23, q_target_idx=26)
        # constraints.add(Constraints.patella_l_Ty, instant=Instant.ALL, x_interpol=x, q_ref_idx=23, q_target_idx=27)
        # constraints.add(Constraints.patella_l_Rz, instant=Instant.ALL, x_interpol=x, q_ref_idx=23, q_target_idx=28)

        # -- track state --
        constraints.add(
            Constraint.TRACK_STATE,
            instant=Instant.ALL,
            states_idx=range(nb_q),
            target=q_ref[p],
        )

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
        # if (p==0):
        #     init_x = np.zeros((nb_q + nb_qdot, number_shooting_points[p] + 1))
        #     init_x[:nb_q, :] = np.load("./RES/part0/q.npy")
        #     init_x[nb_q: nb_q + nb_qdot, :] = np.load("./RES/part0/q_dot.npy")
        #     x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)
        # else:
        init_x = np.zeros((nb_q + nb_qdot, number_shooting_points[p] + 1))
        init_x[:nb_q, :] = np.load("./RES/part0/q.npy")
        init_x[nb_q: nb_q + nb_qdot, :] = np.load("./RES/part0/q_dot.npy")
        # init_x[:nb_q, :] = q_ref[p]
        # init_x[nb_q: nb_q + nb_qdot, :] = qdot_ref[p]
        x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)

    u_init = InitialConditionsList()
    for p in range(nb_phases):
        # if (p==0):
        #     init_u = np.load("./RES/part0/tau.npy")[:, :-1]
        #     u_init.add(init_u, interpolation=InterpolationType.EACH_FRAME)
        # else:
        # init_u = tau_ref[p]
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

    # Generate data from file OpenSim
    [Q_ref, Qdot_ref, Qddot_ref] = Moco.get_state_tracked(t_init, t_end, final_time, nb_q, nb_shooting)
    [Q_sol, Qdot_sol, Activation_sol] = Moco.get_state_from_solution(t_init, t_end, final_time, nb_q, nb_shooting)
    [Tau_sol, Excitation_sol] = Moco.get_control_from_solution(t_init, t_end, final_time, nb_q, nb_shooting)
    Tau_ref = Moco.get_tau_from_inverse_dynamics(t_init, t_end, final_time, nb_q, nb_shooting, Qddot_ref)

    # q_ref = (Q_ref[:, :7], Q_ref[:, 6:16], Q_ref[:, 15:25], Q_ref[:, 24:35])
    # qdot_ref = (Qdot_ref[:, :7], Qdot_ref[:, 6:16], Qdot_ref[:, 15:25], Qdot_ref[:, 24:35])
    #
    # biorbd_model = [biorbd.Model("../../ModelesS2M/Open_Sim/Multiphase/subject_walk_armless_test_1.bioMod"),
    #                 biorbd.Model("../../ModelesS2M/Open_Sim/Multiphase/subject_walk_armless_test_2.bioMod"),]
    # q_ref = [Q_ref[:, :7], Q_ref[:, 6:16],]
    # qdot_ref = [Qdot_ref[:, :7], Qdot_ref[:, 6:16],]
    # tau_ref = [Tau_ref[:, :6], Tau_ref[:, 6:15]]
    # number_shooting_points = [6, 9]
    # phase_time = [0.1008, 0.1512]

    biorbd_model = [biorbd.Model("../../ModelesS2M/Open_Sim/Multiphase/subject_walk_armless_test_1.bioMod"),]
    q_ref = [Q_ref[:, :7],]
    qdot_ref = [Qdot_ref[:, :7],]
    tau_ref = [Tau_ref[:, :7],]
    number_shooting_points = [6,]
    phase_time = [0.1008,]

    # plt.plot(t, GRF_ref[1, :])
    # plt.plot(t, GRF_ref[7, :])
    # plt.legend(["right", "left"])

    ocp = prepare_ocp(
        biorbd_model=biorbd_model,
        phase_time=phase_time,
        number_shooting_points=number_shooting_points,
        q_ref=q_ref,
        qdot_ref=qdot_ref,
        tau_ref=tau_ref,
        nb_thread=4,
    )

    # sim = Simulate.from_controls_and_initial_states(ocp, ocp.original_values["X_init"][0], ocp.original_values["U_init"][0], single_shoot=True)
    # states_sim, controls_sim = Data.get_data(ocp, sim["x"])
    # ShowResult(ocp, sim).graphs()
    # ShowResult(ocp, sim).animate()


    # --- Solve the program --- #
    tic = time()
    sol = ocp.solve(
        solver=Solver.IPOPT,
        solver_options={
            "ipopt.tol": 1e-4,
            "ipopt.max_iter": 100,
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
    tau = controls_sol["tau"]

    # --- Save Results --- #
    np.save("./RES/part0/tau", tau)
    np.save("./RES/part0/q_dot", q_dot)
    np.save("./RES/part0/q", q)

    q_name = []
    for s in range(biorbd_model[0].nbSegment()):
        seg_name = biorbd_model[0].segment(s).name().to_string()
        for d in range(biorbd_model[0].segment(s).nbDof()):
            dof_name = biorbd_model[0].segment(s).nameDof(d).to_string()
            q_name.append(seg_name + "_" + dof_name)

    figure, axes = plt.subplots(5, 6)
    axes = axes.flatten()
    for i in range(nb_q):
        axes[i].plot(q_ref[0][i, :], "b")
        axes[i].plot(q[i, :], "r")
        # axes[i].set_ylim([np.min(q_ref[0][i, :]) - 0.05, np.max(q_ref[0][i, :]) + 0.05])
        axes[i].set_title(q_name[i])


    # --- Show results --- #
    b = BiorbdViz(loaded_model=biorbd_model[0])
    # b.load_movement(q_ref[0])
    b.load_movement(q)

    result = ShowResult(ocp, sol)
    result.animate()
    result.graphs()
