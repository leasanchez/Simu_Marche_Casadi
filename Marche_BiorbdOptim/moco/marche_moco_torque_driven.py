import numpy as np
import biorbd
from time import time
from BiorbdViz import BiorbdViz
from casadi import vertcat, MX, Function, jacobian
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import Marche_BiorbdOptim.moco.Load_OpenSim_data as Moco
import Marche_BiorbdOptim.moco.constraints_dof as Constraints

from biorbd_optim import (
    OptimalControlProgram,
    BoundsList,
    Constraint,
    ConstraintList,
    DynamicsType,
    DynamicsTypeList,
    QAndQDotBounds,
    InitialConditionsList,
    ShowResult,
    Data,
    Objective,
    ObjectiveList,
    InterpolationType,
    Solver,
    Simulate,
    OdeSolver,
)

def plot_control(ax, t, x, color="k", linestyle="--", linewidth=0.7):
    nbPoints = len(np.array(x))
    for n in range(nbPoints - 1):
        ax.plot([t[n], t[n + 1], t[n + 1]], [x[n], x[n], x[n + 1]], color, linestyle, linewidth)

def prepare_ocp(
    biorbd_model, final_time, nb_shooting, q_ref, qdot_ref, tau_ref, activation_ref, grf_ref, moment_ref, nb_threads,
):
    # Problem parameters
    nb_q = biorbd_model.nbQ()
    nb_qdot = biorbd_model.nbQdot()
    nb_tau = biorbd_model.nbGeneralizedTorque()
    torque_min, torque_max = -1500, 1500

    # Add objective functions
    objective_functions = ObjectiveList()
    #objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, controls_idx=range(6, nb_q), weight=0.01, phase=0)
    objective_functions.add(Objective.Lagrange.TRACK_STATE, weight=100, states_idx=range(nb_q), target=q_ref, phase=0)

    # Dynamics
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.TORQUE_DRIVEN, phase=0)

    # Constraints
    # x = np.array(
    #     [0, 0.174533, 0.349066, 0.523599, 0.698132, 0.872665, 1.0472, 1.22173, 1.39626, 1.5708, 1.74533, 1.91986,
    #      2.0944])
    constraints = ConstraintList()
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

    # External forces
    external_forces = [np.zeros((6, 2, nb_shooting))] # 1 torseur par jambe
    for i in range(len(grf_ref)):
        external_forces[0][:3, i, :] = moment_ref[i][:, :-1]
        external_forces[0][3:, i, :] = grf_ref[i][:, :-1]

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
    init_u[:nb_tau] = tau_ref
    u_init.add(init_u, interpolation=InterpolationType.EACH_FRAME)

    # ------------- #

    return OptimalControlProgram(
        biorbd_model=biorbd_model,
        dynamics_type=dynamics,
        number_shooting_points=nb_shooting,
        phase_time=final_time,
        X_init=x_init,
        U_init=u_init,
        X_bounds=x_bounds,
        U_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        nb_threads=nb_threads,
        external_forces=external_forces,
        ode_solver=OdeSolver.RK,
    )


if __name__ == "__main__":
    # Define the problem
    biorbd_model = biorbd.Model("../../ModelesS2M/Open_Sim/subject_walk_armless_test.bioMod")
    t_init = 0.81
    t_end = 1.7
    dt = 0.01
    final_time = np.round(t_end - t_init, 2)
    nb_shooting = int(final_time/dt)

    # model parameters
    nb_q = biorbd_model.nbQ()
    nb_qdot = biorbd_model.nbQdot()
    nb_tau = biorbd_model.nbGeneralizedTorque()
    nb_mus = biorbd_model.nbMuscleTotal()
    nb_markers = biorbd_model.nbMarkers()
    node_t = np.linspace(0, final_time, nb_shooting + 1)

    # Generate data from file OpenSim
    [Q_ref, Qdot_ref, Qddot_ref] = Moco.get_state_tracked(t_init, t_end, final_time, nb_q, nb_shooting)
    [Q_MI, Qdot_MI, Activation_MI] = Moco.get_state_from_solution_MI(t_init, t_end, final_time, nb_q, nb_shooting)
    [Tau_MI, Excitation_MI] = Moco.get_control_from_solution_MI(t_init, t_end, final_time, nb_q, nb_shooting)
    [Force_ref, Moment_ref] = Moco.get_grf(t_init, t_end, final_time, nb_shooting)
    position = Moco.get_position(t_init, t_end, final_time, nb_shooting)
    Tau_OpenSim_ID = Moco.get_tau_from_inverse_dynamics("inverse_dynamics_fext.xlsx", t_init, t_end, final_time, nb_q, nb_shooting)

    # Compute ankle moments
    markers_pos = np.zeros((3, nb_markers, nb_shooting + 1))
    for i in range(nb_shooting + 1):
        for m in range(nb_markers):
            # --- define casadi function ---
            q_iv = MX.sym("Q", nb_q, 1)
            marker_func = biorbd.to_casadi_func("marker", biorbd_model.marker, q_iv, m)
            markers_pos[:, m, i:i + 1] = marker_func(Q_ref[:, i])

    Mext = []
    for leg in range(len(Force_ref)):
        pos = position[leg]
        force = Force_ref[leg]
        moment = Moment_ref[leg]
        marker = markers_pos[:, leg, :]
        M = np.zeros((3, nb_shooting + 1))
        for i in range(nb_shooting + 1):
            p = marker[:, i] - pos[:, i]
            M[:, i] = np.cross(p, force[:, i]) + moment[:, i]
        Mext.append(M)

    # Use ID to compute initial guess
    Tau_ID = np.zeros((nb_q, nb_shooting + 1))
    for n in range(nb_shooting + 1):
        # --- define external forces ---
        forces = biorbd.VecBiorbdSpatialVector()
        forces.append(biorbd.SpatialVector(MX(
            (Mext[0][0, n], Mext[0][1, n], Mext[0][2, n], Force_ref[0][0, n], Force_ref[0][1, n], Force_ref[0][2, n]))))
        forces.append(biorbd.SpatialVector(MX(
            (Mext[1][0, n], Mext[1][1, n], Mext[1][2, n], Force_ref[1][0, n], Force_ref[1][1, n], Force_ref[1][2, n]))))
        # --- define casadi function ---
        q_iv = MX.sym("Q", nb_q, 1)
        dq_iv = MX.sym("Qdot", nb_q, 1)
        ddq_iv = MX.sym("Qddot", nb_q, 1)
        # func = biorbd.to_casadi_func("ID", biorbd_model.InverseDynamics, q_iv, dq_iv, ddq_iv, forces)
        func = biorbd.to_casadi_func("ID", biorbd_model.InverseDynamics, q_iv, dq_iv, ddq_iv)
        # --- compute torques from inverse dynamics ---
        q_iv = Q_ref[:, n]
        dq_iv = Qdot_ref[:, n]
        ddq_iv = Qddot_ref[:, n]
        Tau_ID[:, n:n + 1] = func(q_iv, dq_iv, ddq_iv)
    #
    # # --- Show results --- #
    # b = BiorbdViz(loaded_model=biorbd_model)
    # b.load_movement(Q_ref)
    # b.exec()

    # Track these data
    ocp = prepare_ocp(
        biorbd_model=biorbd_model,
        final_time=final_time,
        nb_shooting=nb_shooting,
        q_ref=Q_ref,
        qdot_ref=Qdot_ref,
        tau_ref=Tau_ID[:, :-1],
        activation_ref=Activation_MI[:, :-1],
        grf_ref=Force_ref,
        moment_ref=Mext,
        nb_threads=1,
    )

    # # --- Nan in initial guess ? --- #
    # # compute constraint value + jacobian
    # g = Function("g", [ocp.V], ocp.g[0])
    # Jg_init = []
    # for i in range(nb_shooting):
    #     Jg = Function("Jg", [ocp.V], [jacobian(ocp.g[0][i], ocp.V)])
    #     Jg_init.append(Jg(ocp.V_init.init))
    # g_init = g(ocp.V_init.init)
    #
    # Nan_g = []
    # Nan_Jg = []
    # for i in range(nb_shooting):
    #     Nan_Jg.append(np.isnan(np.array(Jg_init[0])).sum())
    #     Nan_g.append(np.isnan(np.array(g_init[0])).sum())

    sim = Simulate.from_controls_and_initial_states(ocp, ocp.original_values["X_init"][0], ocp.original_values["U_init"][0], single_shoot=True)
    states_zeros, controls_zeros = Data.get_data(ocp, sim["x"])
    ShowResult(ocp, sim).graphs()

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
    states_sol, controls_sol = Data.get_data(ocp, sol["x"])
    q = states_sol["q"]
    q_dot = states_sol["q_dot"]
    tau = controls_sol["tau"]

    # --- Show results --- #
    ShowResult(ocp, sol).animate()
    ShowResult(ocp, sol).graphs()
