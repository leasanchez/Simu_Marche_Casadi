import numpy as np
import biorbd
from time import time
from BiorbdViz import BiorbdViz
from casadi import MX, Function, vertcat
from matplotlib import pyplot as plt
import Marche_BiorbdOptim.moco.Load_OpenSim_data as Moco
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
    biorbd_model, final_time, nb_shooting, q_ref, qdot_ref, tau_ref, grf_ref, moment_ref, nb_threads,
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
    constraints = ConstraintList()

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
    init_u = tau_ref
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
    [Q_ref, Qdot_ref, Qddot_ref] = Moco.get_state_tracked(t_init, t_end, final_time, biorbd_model.nbQ(), nb_shooting)
    [Q_sol, Qdot_sol, Activation_sol] = Moco.get_state_from_solution(t_init, t_end, final_time, biorbd_model.nbQ(), nb_shooting)
    [Tau_sol, Excitation_sol] = Moco.get_control_from_solution(t_init, t_end, final_time, biorbd_model.nbQ(), nb_shooting)
    Tau_inverse=Moco.get_tau_from_inverse_dynamics(t_init, t_end, final_time, nb_q, nb_shooting, Qddot_ref)
    Tau_muscles=Moco.get_tau_from_muscles(biorbd_model, t_init, t_end, final_time, nb_shooting)
    [Force_ref, Moment_ref] = Moco.get_grf(t_init, t_end, final_time, nb_shooting)
    position = Moco.get_position(t_init, t_end, final_time, nb_shooting)

    # # Animate motion
    # b = BiorbdViz(loaded_model=biorbd_model)
    # b.load_movement(Q_ref)

    # Plot tau
    q_name = []
    for s in range(biorbd_model.nbSegment()):
        seg_name = biorbd_model.segment(s).name().to_string()
        for d in range(biorbd_model.segment(s).nbDof()):
            dof_name = biorbd_model.segment(s).nameDof(d).to_string()
            q_name.append(seg_name + "_" + dof_name)

    figure, axes = plt.subplots(5, 6)
    axes = axes.flatten()
    for i in range(nb_tau):
        # axes[i].plot(node_t, np.zeros(nb_shooting + 1), color="black", linestyle="--")
        axes[i].plot(node_t, Tau_inverse[i, :], color="tab:green")
        # axes[i].plot(node_t, Tau_muscles[i, :], color="tab:red")
        axes[i].set_title("Tau " + q_name[i])
    # plt.legend(["zeros", "inverse dynamics", "muscles"])
    plt.show()

    # Compute moment at ankle
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

    # ---- tau=zeros ----
    ocp_zeros = prepare_ocp(
        biorbd_model= biorbd.Model("../../ModelesS2M/Open_Sim/subject_walk_armless_test.bioMod"),
        final_time=final_time,
        nb_shooting=nb_shooting,
        q_ref=Q_ref,
        qdot_ref=Qdot_ref,
        tau_ref=np.zeros((nb_tau, nb_shooting)),
        grf_ref=Force_ref,
        moment_ref=Mext,
        nb_threads=1,
    )

    sim = Simulate.from_controls_and_initial_states(ocp_zeros, ocp_zeros.original_values["X_init"][0], ocp_zeros.original_values["U_init"][0], single_shoot=True)
    states_zeros, controls_zeros = Data.get_data(ocp_zeros, sim["x"])
    ShowResult(ocp_zeros, sim).graphs()
    # ShowResult(ocp, sim).animate()

    # ---- tau=tau from inverse dynamics ----
    ocp_inverse = prepare_ocp(
        biorbd_model=biorbd.Model("../../ModelesS2M/Open_Sim/subject_walk_armless_test.bioMod"),
        final_time=final_time,
        nb_shooting=nb_shooting,
        q_ref=Q_ref,
        qdot_ref=Qdot_ref,
        tau_ref=Tau_inverse[:, :-1],
        grf_ref=Force_ref,
        moment_ref=Mext,
        nb_threads=1,
    )

    sim = Simulate.from_controls_and_initial_states(ocp_inverse, ocp_inverse.original_values["X_init"][0], ocp_inverse.original_values["U_init"][0], single_shoot=True)
    states_inverse, controls_inverse = Data.get_data(ocp_inverse, sim["x"])
    ShowResult(ocp_inverse, sim).graphs()
    # ShowResult(ocp_inverse, sim).animate()

    # ---- tau=tau from muscles excitations/activations ----
    ocp_muscles = prepare_ocp(
        biorbd_model=biorbd.Model("../../ModelesS2M/Open_Sim/subject_walk_armless_test.bioMod"),
        final_time=final_time,
        nb_shooting=nb_shooting,
        q_ref=Q_ref,
        qdot_ref=Qdot_ref,
        tau_ref=Tau_muscles[:, :-1],
        grf_ref=Force_ref,
        moment_ref=Mext,
        nb_threads=1,
    )

    sim = Simulate.from_controls_and_initial_states(ocp_muscles, ocp_muscles.original_values["X_init"][0], ocp_muscles.original_values["U_init"][0], single_shoot=True)
    states_sim, controls_sim = Data.get_data(ocp_muscles, sim["x"])
    ShowResult(ocp_muscles, sim).graphs()
    # ShowResult(ocp_muscles, sim).animate()