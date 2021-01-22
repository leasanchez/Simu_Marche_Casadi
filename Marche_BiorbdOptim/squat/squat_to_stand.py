from time import time
import biorbd

from biorbd_optim import (
    BidirectionalMapping,
    Mapping,
    ObjectiveOption,
    Objective,
    ObjectiveList,
    DynamicsTypeOption,
    DynamicsType,
    ConstraintList,
    Constraint,
    Instant,
    BoundsOption,
    QAndQDotBounds,
    InitialConditionsOption,
    InterpolationType,
    OptimalControlProgram,
    ShowResult,
    Data,
    Solver,
)


def prepare_ocp(biorbd_model_path, number_shooting_points, final_time, use_actuators=False, use_symmetry_by_constraints=False, use_SX=False):
    # --- Options --- #

    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)

    # Problem parameters
    if use_actuators:
        tau_min, tau_max = -1, 1
        if use_symmetry_by_constraints:
            tau_init = [0, 0, 0, 0, 1, 0, 1, -1, -1, -1, -1, -1, -1]
        else:
            tau_init = [1, -1, -1, -1]
    else:
        tau_min, tau_max = -1000, 1000
        if use_symmetry_by_constraints:
            # tau_init = [3.33841359e-14, -1.60018112e-13, 4.03183495e+01, -4.35790988e-01, 5.12200905, 4.35790988e-01, 5.12200905, -1.64587104, 3.10483864e+01, -1.75036755, -1.64587104, 3.10483864e+01, -1.75036755]
            tau_init = [0, 0, 0, 0, 1000, 0, 1000, -1000, -1000, -1000, -1000, -1000, -1000]
        else:
            tau_init = [1000, -1000, -1000, -1000]

    # Symmetry
    if not use_symmetry_by_constraints:        # i.e. symmetry by construction (mapping) is chosen
        q_mapping = BidirectionalMapping(
            Mapping([0, 1, 2, -1, 3, -1, 3, 4, 5, 6, 4, 5, 6]), Mapping([0, 1, 2, 4, 7, 8, 9])         # Here I remove the rotation of arm on himself (around z-axis), if used, do not forget to add [5] in sign_to_oppose arg of the first Mapping
        )
        tau_mapping = BidirectionalMapping(
            Mapping([-1, -1, -1, -1, 0, -1, 0, 1, 2, 3, 1, 2, 3]), Mapping([4, 7, 8, 9])
        )

    # --- Objective function --- #
    # objective_functions = ObjectiveOption(Objective.Lagrange.MINIMIZE_TORQUE)
    # objective_functions = ObjectiveOption(Objective.Mayer.MINIMIZE_TIME)
    objective_functions = ObjectiveList()
    # objective_functions.add(Objective.Mayer.MINIMIZE_TIME, weight=100)
    objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=1)

    # --- Dynamics --- #
    if use_actuators:
        dynamics = DynamicsTypeOption(DynamicsType.TORQUE_ACTIVATIONS_DRIVEN_WITH_CONTACT)
    else:
        dynamics = DynamicsTypeOption(DynamicsType.TORQUE_DRIVEN_WITH_CONTACT)

    # --- Constraints --- #
    constraints = ConstraintList()

    # Positivity constraints of the normal component of the reaction forces
    contact_axes = (1, 2, 4, 5)
    for i in contact_axes:
        constraints.add(Constraint.CONTACT_FORCE_INEQUALITY, direction="GREATER_THAN", instant=Instant.ALL, contact_force_idx=i, boundary=0)

    # Non-slipping constraints
    # N.B.: Application on only one of the two feet is sufficient, as the slippage cannot occurs on only one foot here
    constraints.add(Constraint.NON_SLIPPING, instant=Instant.ALL, normal_component_idx=(1, 2), tangential_component_idx=0, static_friction_coefficient=0.5)

    # Constraints for symmetry
    # if use_symmetry_by_constraints:
    #     first_dof = (3, 4, 7, 8, 9)
    #     second_dof = (5, 6, 10, 11, 12)
    #     coef = (-1, 1, 1, 1, 1)
    #     for i in range(len(first_dof)):
    #         constraints.add(Constraint.PROPORTIONAL_STATE, instant=Instant.ALL, first_dof=first_dof[i], second_dof=second_dof[i], coef=coef[i])

    # Time constraint #TODO: See if necessary to bound the free-time that will be minimized
    # constraints.add(Constraint.TIME_CONSTRAINT, minimum=0.1, maximum=1)

    # --- Path constraints --- #
    if use_symmetry_by_constraints:
        nb_q = biorbd_model.nbDof()
        nb_qdot = nb_q
        nb_tau = nb_q
        # pose_at_first_node = [-0.12, -0.23, -1.10, 0, 1.85, 0, 1.85, 2.06, -1.67, 0.55, 2.06, -1.67, 0.55]
        # pose_at_first_node = [-0.13, -0.38, -1.21, 0, 2.67, 0, 2.67, 2.48, -2.08, 0.63, 2.48, -2.08, 0.63]
        pose_at_first_node = [-0.06, -0.16, -0.83, 0, 0.84, 0, 0.84, 1.53, -1.55, 0.68, 1.53, -1.55, 0.68]      # Without angles of 90° to prevent
        pose_at_last_node = [0, 0.07, -0.52, 0, 1.3, 0, 1.3, 0.37, -0.13, 0.11, 0.37, -0.13, 0.11]
        # pose_at_last_node = [0, 0, -0.5336, 0, 1.4, 0, 1.4, 0.8, -0.9, 0.47, 0.8, -0.9, 0.47]
    else:
        nb_q = q_mapping.reduce.len
        nb_qdot = nb_q
        nb_tau = tau_mapping.reduce.len
        # pose_at_first_node = [-0.12, -0.23, -1.10, 1.85, 2.06, -1.67, 0.55]
        # pose_at_first_node = [-0.13, -0.38, -1.21, 2.67, 2.48, -2.08, 0.63]
        pose_at_first_node = [-0.06, -0.16, -0.83, 0.84, 1.53, -1.55, 0.68]         # Without angles of 90° kind of gimbal lock
        pose_at_last_node = [0, 0.07, -0.52, 1.3, 0.37, -0.13, 0.11]
        # pose_at_last_node = [0, 0, -0.5336, 1.4, 0.8, -0.9, 0.47]

    # Bounds on states (Interpolation type is CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)
    if use_symmetry_by_constraints:
        x_bounds = BoundsOption(QAndQDotBounds(biorbd_model))
    else:
        x_bounds = BoundsOption(QAndQDotBounds(biorbd_model, all_generalized_mapping=q_mapping))
    x_bounds.min[:, 0] = pose_at_first_node + [0] * nb_qdot
    x_bounds.max[:, 0] = pose_at_first_node + [0] * nb_qdot
    x_bounds.min[:, -1] = pose_at_last_node + [0] * nb_qdot
    x_bounds.max[:, -1] = pose_at_last_node + [0] * nb_qdot

    # Initial guess for states (Interpolation type is CONSTANT)
    x_init = InitialConditionsOption(pose_at_first_node + [0] * nb_qdot)

    # Bounds on controls (Interpolation type is CONSTANT)
    u_bounds = BoundsOption([[tau_min] * nb_tau, [tau_max] * nb_tau], interpolation=InterpolationType.CONSTANT)

    # Initial guess for controls (Interpolation type is CONSTANT)
    u_init = InitialConditionsOption(tau_init)
    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        number_shooting_points,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        constraints,
        q_mapping=q_mapping if not use_symmetry_by_constraints else None,
        q_dot_mapping=q_mapping if not use_symmetry_by_constraints else None,
        tau_mapping=tau_mapping if not use_symmetry_by_constraints else None,
        nb_integration_steps=2,
        use_SX=use_SX,
    )


if __name__ == "__main__":
    tic = time()
    ocp = prepare_ocp("/home/iornaith/Documents/GitKraken/JumperOCP/models/jumper2contacts.bioMod", number_shooting_points=31, final_time=0.4, use_actuators=False, use_symmetry_by_constraints=True, use_SX=False)
    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True, solver_options={"hessian_approximation": "exact", "linear_solver": "ma57", "print_info_string": "yes"})
    # sol = ocp.solve(show_online_optim=False, solver=Solver.ACADOS)
    OptimalControlProgram.save(ocp, sol, "../squat2stand/v1_18_08_squat2stand.bo")
    toc = time() - tic
    print(f"Time to solve : {toc}sec")
    # --- Show results --- #
    # param = Data.get_data(ocp, sol["x"], get_states=False, get_controls=False, get_parameters=True)
    # print(f"The optimized phase time are: {param['time'][0, 0]}s .")
    result = ShowResult(ocp, sol)
    result.animate()
    result.graphs(adapt_graph_size_to_bounds=True)
    print("Simulation finished")
