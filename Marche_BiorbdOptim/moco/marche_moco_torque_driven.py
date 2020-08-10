import numpy as np
import biorbd
from time import time
from BiorbdViz import BiorbdViz
from casadi import vertcat, MX, Function, interp1d, interpolant
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from Marche_BiorbdOptim.moco.Load_OpenSim_data import get_q, get_grf, get_position
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
    Solver,
    ConstraintList,
    Instant,
    Simulate,
)


def prepare_ocp(
    biorbd_model, final_time, nb_shooting, q_ref, qdot_ref, grf_ref, moment_ref, nb_threads,
):
    # Problem parameters
    nb_q = biorbd_model.nbQ()
    nb_qdot = biorbd_model.nbQdot()
    nb_tau = biorbd_model.nbGeneralizedTorque()

    torque_min, torque_max, torque_init = -1000, 1000, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=1, phase=0)
    objective_functions.add(Objective.Lagrange.TRACK_STATE, weight=100, states_idx=range(nb_q), data_to_track=q_ref.T, phase=0)

    # Dynamics
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.TORQUE_DRIVEN, phase=0)

    # Constraints
    x = np.array(
        [0, 0.174533, 0.349066, 0.523599, 0.698132, 0.872665, 1.0472, 1.22173, 1.39626, 1.5708, 1.74533, 1.91986,
         2.0944])
    constraints = ConstraintList()
    # -- tibia_r --
    constraints.add(Constraints.tibia_r_Ty, instant=Instant.ALL, x_interpol=x, q_ref_idx=11, q_target_idx=9)
    constraints.add(Constraints.tibia_r_Tz, instant=Instant.ALL, x_interpol=x, q_ref_idx=11, q_target_idx=10)
    constraints.add(Constraints.tibia_r_Rz, instant=Instant.ALL, x_interpol=x, q_ref_idx=11, q_target_idx=12)
    constraints.add(Constraints.tibia_r_Ry, instant=Instant.ALL, x_interpol=x, q_ref_idx=11, q_target_idx=13)

    # -- patella_r --
    constraints.add(Constraints.patella_r_Tx, instant=Instant.ALL, x_interpol=x, q_ref_idx=11, q_target_idx=14)
    constraints.add(Constraints.patella_r_Ty, instant=Instant.ALL, x_interpol=x, q_ref_idx=11, q_target_idx=15)
    constraints.add(Constraints.patella_r_Rz, instant=Instant.ALL, x_interpol=x, q_ref_idx=11, q_target_idx=16)

    # -- tibia_l --
    constraints.add(Constraints.tibia_l_Ty, instant=Instant.ALL, x_interpol=x, q_ref_idx=23, q_target_idx=21)
    constraints.add(Constraints.tibia_l_Tz, instant=Instant.ALL, x_interpol=x, q_ref_idx=23, q_target_idx=22)
    constraints.add(Constraints.tibia_l_Rz, instant=Instant.ALL, x_interpol=x, q_ref_idx=23, q_target_idx=24)
    constraints.add(Constraints.tibia_l_Ry, instant=Instant.ALL, x_interpol=x, q_ref_idx=23, q_target_idx=25)

    # -- patella_l --
    constraints.add(Constraints.patella_l_Tx, instant=Instant.ALL, x_interpol=x, q_ref_idx=23, q_target_idx=26)
    constraints.add(Constraints.patella_l_Ty, instant=Instant.ALL, x_interpol=x, q_ref_idx=23, q_target_idx=27)
    constraints.add(Constraints.patella_l_Rz, instant=Instant.ALL, x_interpol=x, q_ref_idx=23, q_target_idx=28)

    # External forces
    external_forces = [np.zeros((6, 2, nb_shooting))] # 1 torseur par jambe
    for i in range(len(grf_ref)):
        external_forces[0][:3, i, :] = grf_ref[i][:, :-1]
        external_forces[0][3:, i, :] = moment_ref[i][:, :-1]

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
        external_forces=external_forces,
    )


if __name__ == "__main__":
    # Define the problem
    biorbd_model = biorbd.Model("../../ModelesS2M/Open_Sim/subject_walk_armless_test.bioMod")
    t_init = 0.81
    t_end = 1.65
    final_time = t_end - t_init
    nb_shooting = 50

    # model parameters
    nb_q = biorbd_model.nbQ()
    nb_qdot = biorbd_model.nbQdot()
    nb_tau = biorbd_model.nbGeneralizedTorque()
    nb_markers = biorbd_model.nbMarkers()
    node_t = np.linspace(0, final_time, nb_shooting + 1)

    # Generate data from file OpenSim
    [Q_ref, Qdot_ref, Qddot_ref] = get_q(t_init, t_end, final_time, biorbd_model.nbQ(), nb_shooting)
    [Force_ref, Moment_ref] = get_grf(t_init, t_end, final_time, nb_shooting)
    position = get_position(t_init, t_end, final_time, nb_shooting)

    symbolic_q = MX.sym("x", nb_q, 1)
    markers_func=Function(
        "ForwardKin",
        [symbolic_q],
        [biorbd_model.markers(symbolic_q)],
        ["q"],
        ["marker_pos"],
        ).expand()

    markers_pos = np.zeros((3, nb_markers, nb_shooting + 1))
    for i in range(nb_shooting + 1):
        markers_pos[:, :, i]=markers_func(Q_ref[:, i])

    # b = BiorbdViz(loaded_model=biorbd_model)
    # b.load_movement(Q_ref)

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter3D(position[0][0, :], position[0][1, :], position[0][2, :], color='blue')
    ax.scatter3D(position[1][0, :], position[1][1, :], position[1][2, :], color='red')
    ax.scatter3D(markers_pos[0, 0, :], markers_pos[1, 0, :], markers_pos[2, 0, :], color='green')
    ax.scatter3D(markers_pos[0, 1, :], markers_pos[1, 1, :], markers_pos[2, 1, :], color='magenta')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    Mext = []
    for leg in range(len(Force_ref)):
        pos = position[leg]
        force = Force_ref[leg]
        moment = Moment_ref[leg]
        marker = markers_pos[:, leg, :]
        M = np.zeros((3, nb_shooting + 1))
        for i in range(nb_shooting + 1):
            p = pos[:, i] - marker[:, i]
            M[:, i]=np.cross(p, force[:, i]) + moment[:, i]
        Mext.append(M)

    figure, axes = plt.subplots(1, 3)
    axes[0].plot(node_t, Force_ref[0].T)
    axes[0].set_title('Forces right')
    axes[1].plot(node_t, Moment_ref[0].T)
    axes[1].set_title('Moments init right')
    axes[2].plot(node_t, Mext[0].T)
    axes[2].set_title('Moments ankle right')
    plt.legend(['x', 'y', 'z'])

    figure, axes = plt.subplots(1, 3)
    axes[0].plot(node_t, Force_ref[1].T)
    axes[0].set_title('Forces left')
    axes[1].plot(node_t, Moment_ref[1].T)
    axes[1].set_title('Moments init left')
    axes[2].plot(node_t, Mext[1].T)
    axes[2].set_title('Moments ankle left')
    plt.legend(['x', 'y', 'z'])

    plt.show()
    # Track these data
    ocp = prepare_ocp(
        biorbd_model,
        final_time=final_time,
        nb_shooting=nb_shooting,
        q_ref=Q_ref,
        qdot_ref=Qdot_ref,
        grf_ref=Force_ref,
        moment_ref=Mext,
        nb_threads=1,
    )

    # U_init_sim = InitialConditionsList()
    # U_init_sim.add([0]*nb_tau, interpolation=InterpolationType.CONSTANT)
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
