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
    StateTransitionList,
    StateTransitionFcn,
    Solver,
)

def prepare_ocp(biorbd_model, nb_shooting, final_time):

    # Problem parameters
    nb_phases = len(biorbd_model)
    nb_q = biorbd_model.nbQ()
    nb_qdot = biorbd_model.nbQdot()
    nb_tau = biorbd_model.nbGeneralizedTorque()
    nb_mus = biorbd_model.nbMuscleTotal()

    min_bound, max_bound = 0, np.inf
    torque_min, torque_max, torque_init = -1000, 1000, 0
    activation_min, activation_max, activation_init = 1e-3, 1.0, 0.1

    # --- Objective function --- #
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=1, quadratic=True)
    # objective_functions = ObjectiveOption(Objective.Lagrange.MINIMIZE_TORQUE)
    # objective_functions = ObjectiveOption(Objective.Mayer.MINIMIZE_TIME)

    # --- Dynamics --- #
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN_WITH_CONTACT)

    # --- Constraints --- #
    constraints = ConstraintList()
    constraints.add( # positive vertical forces
        ConstraintFcn.CONTACT_FORCE,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.ALL,
        contact_force_idx=(1, 2, 5),
    )
    constraints.add( # non slipping y
        ConstraintFcn.NON_SLIPPING,
        node=Node.ALL,
        normal_component_idx=(1, 2, 5),
        tangential_component_idx=4,
        static_friction_coefficient=0.2,
    )
    constraints.add( # non slipping x m5
        ConstraintFcn.NON_SLIPPING,
        node=Node.ALL,
        normal_component_idx=(2, 5),
        tangential_component_idx=3,
        static_friction_coefficient=0.2,
    )
    constraints.add( # non slipping x heel
        ConstraintFcn.NON_SLIPPING,
        node=Node.ALL,
        normal_component_idx=1,
        tangential_component_idx=0,
        static_friction_coefficient=0.2,
    )

    # --- Path constraints --- #
    # x_bounds = BoundsList()
    # u_bounds = BoundsList()
    x_bounds=Bounds(bounds=QAndQDotBounds(biorbd_model))
    u_bounds=Bounds(
            [torque_min] * nb_tau + [activation_min] * nb_mus,
            [torque_max] * nb_tau + [activation_max] * nb_mus,
        )

    # --- Initial guess --- #
    # pose_at_first_node = [-0.12, -0.23, -1.10, 0, 1.85, 0, 1.85, 2.06, -1.67, 0.55, 2.06, -1.67, 0.55]
    # pose_at_first_node = [-0.13, -0.38, -1.21, 0, 2.67, 0, 2.67, 2.48, -2.08, 0.63, 2.48, -2.08, 0.63]
    pose_at_first_node = [-0.06, -0.16, -0.83, 0, 0.84, 0, 0.84, 1.53, -1.55, 0.68, 1.53, -1.55, 0.68]      # Without angles of 90° to prevent
    pose_at_last_node = [0, 0.07, -0.52, 0, 1.3, 0, 1.3, 0.37, -0.13, 0.11, 0.37, -0.13, 0.11]
    # pose_at_last_node = [0, 0, -0.5336, 0, 1.4, 0, 1.4, 0.8, -0.9, 0.47, 0.8, -0.9, 0.47]

    x_init = InitialGuess(pose_at_first_node + [0] * nb_qdot)
    u_init = InitialGuess(torque_init*nb_tau + activation_init*nb_mus)


    # Bounds on states (Interpolation type is CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)
    x_bounds.min[:, 0] = pose_at_first_node + [0] * nb_qdot
    x_bounds.max[:, 0] = pose_at_first_node + [0] * nb_qdot
    x_bounds.min[:, -1] = pose_at_last_node + [0] * nb_qdot
    x_bounds.max[:, -1] = pose_at_last_node + [0] * nb_qdot

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
    )

model = biorbd.Model("Modeles_S2M/2legs_18dof_flatfootR.bioMod")
nb_q = model.nbQ()
pose_at_first_node = np.array([-0.06, -0.16, -0.83, 0, 0.84, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
b = bioviz.Viz(loaded_model=model)
b.set_q(pose_at_first_node)
b.exec()


# pose_at_first_node = [-0.06, -0.16, -0.83, 0, 0.84, 0, 0.84, 1.53, -1.55, 0.68, 1.53, -1.55, 0.68]
pose_at_last_node = [0]*nb_q