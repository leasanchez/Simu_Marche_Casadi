from scipy.integrate import solve_ivp
import numpy as np
from casadi import MX, Function, vertcat
from matplotlib import pyplot as plt
import sys

sys.path.append('/home/leasanchez/programmation/BiorbdOptim')
import biorbd

from biorbd_optim import (
    Instant,
    OptimalControlProgram,
    ProblemType,
    Objective,
    Constraint,
    Bounds,
    QAndQDotBounds,
    InitialConditions,
    ShowResult,
    OdeSolver,
)


def prepare_ocp(
    biorbd_model,
    final_time,
    nb_shooting,
    markers_ref,
    activation_ref,
    show_online_optim,
):
    # Problem parameters
    torque_min, torque_max, torque_init = -1000, 1000, 0
    activation_min, activation_max, activation_init = 0, 1, 0.1

    # Add objective functions
    objective_functions = (
        {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 1, "controls_idx":[3, 4, 5]},
        {"type": Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, "weight": 1, "data_to_track":activation_ref.T},
        {"type": Objective.Lagrange.TRACK_MARKERS, "weight": 100, "data_to_track": markers_ref}
    )

    # Dynamics
    variable_type = ProblemType.muscles_and_torque_driven_with_contact

    # Constraints
    constraints = ()

    # Path constraint
    X_bounds = QAndQDotBounds(biorbd_model)

    # Initial guess
    X_init = InitialConditions([0] * (biorbd_model.nbQ() + biorbd_model.nbQdot()))

    # Define control path constraint
    U_bounds = Bounds(
        [torque_min] * biorbd_model.nbGeneralizedTorque() + [activation_min] * biorbd_model.nbMuscleTotal(),
        [torque_max] * biorbd_model.nbGeneralizedTorque() + [activation_max] * biorbd_model.nbMuscleTotal(),
    )
    U_init = InitialConditions(
        [torque_init] * biorbd_model.nbGeneralizedTorque() + [activation_init] * biorbd_model.nbMuscleTotal()
    )

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        variable_type,
        nb_shooting,
        final_time,
        objective_functions,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        constraints,
        show_online_optim=show_online_optim,
    )


if __name__ == "__main__":
    # Define the problem
    biorbd_model = biorbd.Model("ANsWER_Rleg_6dof_17muscle_1contact.bioMod")
    n_shooting_points = 25
    Gaitphase = 'stance'

    # Generate data from file
    from Marche_BiorbdOptim.LoadData import load_data_markers, load_data_q, load_data_emg, load_data_GRF

    name_subject = "equincocont01"
    grf_ref, T, T_stance, T_swing = load_data_GRF(name_subject, biorbd_model, n_shooting_points)
    final_time = T_stance

    t, markers_ref = load_data_markers(name_subject, biorbd_model, final_time, n_shooting_points, Gaitphase)
    q_ref = load_data_q(name_subject, biorbd_model, final_time, n_shooting_points, Gaitphase)
    emg_ref = load_data_emg(name_subject, biorbd_model, final_time, n_shooting_points, Gaitphase)
    activation_ref = np.zeros((biorbd_model.nbMuscleTotal(), n_shooting_points))
    idx_emg = 0
    for i in range(biorbd_model.nbMuscleTotal()):
        if (i!=1) and (i!=2) and (i!=3) and (i!=5) and (i!=6) and (i!=11) and (i!=12):
            activation_ref[i, :] = emg_ref[idx_emg, :-1]
            idx_emg += 1

    # Track these data
    biorbd_model = biorbd.Model("ANsWER_Rleg_6dof_17muscle_1contact.bioMod")
    ocp = prepare_ocp(
        biorbd_model,
        final_time,
        n_shooting_points,
        markers_ref,
        activation_ref,
        show_online_optim=False,
    )

    # --- Solve the program --- #
    sol = ocp.solve()

    # # --- Show the results --- #
    q, qdot, tau, mus = ProblemType.get_data_from_V(ocp, sol["x"])
    n_q = ocp.nlp[0]["model"].nbQ()
    n_mark = ocp.nlp[0]["model"].nbMarkers()
    n_frames = q.shape[1]

    markers = np.ndarray((3, n_mark, q.shape[1]))
    markers_func = []
    for i in range(n_mark):
        markers_func.append(
            Function(
                "ForwardKin",
                [ocp.symbolic_states],
                [biorbd_model.marker(ocp.symbolic_states[:n_q], i).to_mx()],
                ["q"],
                ["marker_" + str(i)],
            ).expand()
        )
    for i in range(n_frames):
        for j, mark_func in enumerate(markers_func):
            markers[:, j, i] = np.array(mark_func(np.append(q[:, i], qdot[:, i]))).squeeze()

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate(show_meshes=False)
    result.graphs()
