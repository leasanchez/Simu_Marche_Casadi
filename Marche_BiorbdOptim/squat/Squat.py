import numpy as np
from casadi import dot, Function, vertcat, MX, mtimes, nlpsol, mmax
import biorbd
import bioviz
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    BoundsList,
    Bounds,
    QAndQDotBounds,
    InitialGuessList,
    InitialGuess,
    ShowResult,
    ObjectiveList,
    ObjectiveFcn,
    InterpolationType,
    Data,
    Node,
    ConstraintList,
    ConstraintFcn,
    Solver,
    PenaltyNodes,
)


def custom_CoM_low(pn: PenaltyNodes) -> MX:
    nq = pn.nlp.shape["q"]
    compute_CoM = biorbd.to_casadi_func("CoM", pn.nlp.model.CoM, pn.nlp.q)
    com = compute_CoM(pn.x[31][:nq])
    return com[2] + 0.25

def custom_CoM_variation(pn: PenaltyNodes) -> MX:
    nq = pn.nlp.shape["q"]
    compute_CoM = biorbd.to_casadi_func("CoM", pn.nlp.model.CoM, pn.nlp.q)
    com_init = compute_CoM(pn.x[0][:nq])
    val = []
    for n in range(1, pn.nlp.ns):
        val = vertcat(val, com_init[0] - compute_CoM(pn.x[pn.t[n]][:nq])[0])
        val = vertcat(val, com_init[1] - compute_CoM(pn.x[pn.t[n]][:nq])[1])
    return val

def custom_CoM_high(pn: PenaltyNodes) -> MX:
    nq = pn.nlp.shape["q"]
    compute_CoM = biorbd.to_casadi_func("CoM", pn.nlp.model.CoM, pn.nlp.q)
    com = compute_CoM(pn.x[0][:nq])
    return com[2]

# OPTIMAL CONTROL PROBLEM
model = biorbd.Model("Modeles_S2M/2legs_18dof_flatfootR.bioMod")
# c = model.contactNames()
# for (i, name) in enumerate(c):
#     print(f"{i} : {name.to_string()}")

# q_name = []
# for s in range(model.nbSegment()):
#     seg_name = model.segment(s).name().to_string()
#     for d in range(model.segment(s).nbDof()):
#         dof_name = model.segment(s).nameDof(d).to_string()
#         q_name.append(seg_name + "_" + dof_name)
# for (i, q) in enumerate(q_name):
#     print(f"{i} : {q}")

# --- Problem parameters --- #
nb_q = model.nbQ()
nb_qdot = model.nbQdot()
nb_tau = model.nbGeneralizedTorque()
nb_mus = model.nbMuscleTotal()
nb_shooting = 62
final_time=1.0
min_bound, max_bound = 0, np.inf
torque_min, torque_max, torque_init = -1000, 1000, 0
activation_min, activation_max, activation_init = 1e-3, 1.0, 0.1

    # --- Objective function --- #
    objective_functions = ObjectiveList()
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=1)
    objective_functions.add(custom_compute_CoM,
                            custom_type=ObjectiveFcn.Mayer,
                            node=Node.END,
                            weight=10)

    # --- Dynamics --- #
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN_WITH_CONTACT)

    # --- Constraints --- #
    constraints = ConstraintList()
    # contact forces constraint
    contact_z_axes = (1, 2, 5, 7, 8, 11)
    for c in contact_z_axes:
        constraints.add( # positive vertical forces
            ConstraintFcn.CONTACT_FORCE,
            min_bound=0,
            max_bound=np.inf,
            node=Node.ALL,
            contact_force_idx=c,
        )


    # --- Path constraints --- #
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))
    x_bounds[0][:nb_q, 0] = 0
    # x_bounds[0][[11, 12, 17,18], -1] = np.pi/2

    u_bounds = BoundsList()
    # u_bounds.add(
    #             [torque_min] * nb_tau + [activation_min] * nb_mus,
    #             [torque_max] * nb_tau + [activation_max] * nb_mus,
    # )
    u_bounds.add(
                [torque_min] * nb_tau,
                [torque_max] * nb_tau,
    )

    # --- Initial guess --- #
    x_init = InitialGuessList()
    init_x = np.zeros((nb_q + nb_qdot, nb_shooting + 1))
    init_x[:nb_q, :] = q_init
    init_x[nb_q:, :] = qdot_init
    x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)

    u_init = InitialGuessList()
    # u_init.add([torque_init]*nb_tau + [activation_init]*nb_mus)
    u_init.add([torque_init] * nb_tau)

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
        n_threads=nb_threads,
    )

if __name__ == "__main__":
    model = biorbd.Model("Modeles_S2M/2legs_18dof_flatfootR.bioMod")

    # c = model.contactNames()
    # for (i, name) in enumerate(c):
    #     print(f"{i} : {name.to_string()}")
    #
    q_name = []
    for s in range(model.nbSegment()):
        seg_name = model.segment(s).name().to_string()
        for d in range(model.segment(s).nbDof()):
            dof_name = model.segment(s).nameDof(d).to_string()
            q_name.append(seg_name + "_" + dof_name)
    for (i, q) in enumerate(q_name):
        print(f"{i} : {q}")

    nb_q = model.nbQ()
    nb_shooting = 31
    position_high = [0]*nb_q
    position_low = [-0.06, -0.36, 0, 0, 0, -0.8,
                    0, 0, 0.2,
                    0, 0, 1.53, -1.55, 0, 0.68,
                    0, 0, 1.53, -1.55, 0, 0.68]
    q_init = np.zeros((nb_q, nb_shooting + 1))
    for i in range(nb_q):
        # q_init[i, :int(nb_shooting/2)] = np.linspace(position_high[i], position_low[i], int(nb_shooting/2))
        # q_init[i, int(nb_shooting/2):] = np.linspace(position_low[i], position_high[i], int(nb_shooting/2) + 1)
        q_init[i, :] = np.linspace(position_high[i], position_low[i], nb_shooting + 1)
    qdot_init = np.gradient(q_init)[1]
    qddot_init = np.gradient(qdot_init)[1]

    symbolic_q = MX.sym("q", nb_q, 1)
    compute_CoM = Function(
        "ComputeCoM",
        [symbolic_q],
        [model.CoM(symbolic_q).to_mx()],
        ["q"],
        ["CoM"],
    ).expand()

    CoM_high = compute_CoM(position_high)
    CoM_low = compute_CoM(position_low)
    # b = bioviz.Viz(loaded_model=model)
    # # b.set_q(np.array(position_low))
    # # b.set_q(np.array(position_high))
    # b.load_movement(q_init)
    # b.exec()

    ocp = prepare_ocp(biorbd_model=model,
                      nb_shooting=31,
                      final_time=0.4,
                      q_init=q_init,
                      qdot_init=qdot_init,
                      nb_threads=4,)

    # --- Solve the program --- #
    sol = ocp.solve(
        solver=Solver.IPOPT,
        solver_options={
            "ipopt.tol": 1e-4,
            "ipopt.max_iter": 5000,
            "ipopt.hessian_approximation": "exact",
            "ipopt.limited_memory_max_history": 50,
            "ipopt.linear_solver": "ma57",
        },
        show_online_optim=False,
    )

    # --- Get Results --- #
    states_sol, controls_sol = Data.get_data(ocp, sol["x"])
    q = states_sol["q"]
    q_dot = states_sol["q_dot"]
    tau = controls_sol["tau"]
    activation = controls_sol["muscles"]

    # --- Show results --- #
    ShowResult(ocp, sol).animate()


