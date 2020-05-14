import numpy as np
from casadi import MX, Function
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
    Data,
    InterpolationType,
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
        {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 1, "controls_idx": [3, 4, 5]},
        {"type": Objective.Lagrange.TRACK_MUSCLES_CONTROL, "weight": 1, "data_to_track": activation_ref.T},
        {"type": Objective.Lagrange.TRACK_MARKERS, "weight": 30, "data_to_track": markers_ref},
    )

    # Dynamics
    variable_type = ProblemType.muscle_excitations_and_torque_driven

    # Constraints
    constraints = ()

    # Path constraint
    X_bounds = QAndQDotBounds(biorbd_model)
    X_bounds.concatenate(
        Bounds([activation_min] * biorbd_model.nbMuscles(), [activation_max] * biorbd_model.nbMuscles())
    )

    # Initial guess
    init_x = np.zeros((3, (biorbd_model.nbQ() + biorbd_model.nbQdot() + biorbd_model.nbMuscleTotal())))

    init_x_start = [0] * (biorbd_model.nbQ() + biorbd_model.nbQdot()) + [0.1] * biorbd_model.nbMuscleTotal()
    init_x_start[:biorbd_model.nbQ()] = q_ref[:, 0]
    init_x[0, :] = init_x_start

    init_x_end = [0] * (biorbd_model.nbQ() + biorbd_model.nbQdot()) + [0.1] * biorbd_model.nbMuscleTotal()
    init_x_end[:biorbd_model.nbQ()] = q_ref[:, -1]
    init_x[2, :] = init_x_end

    init_x_inter = [0] * (biorbd_model.nbQ() + biorbd_model.nbQdot()) + [0.1] * biorbd_model.nbMuscleTotal()
    for i in range(biorbd_model.nbQ()):
        init_x_inter[i] = np.mean(q_ref[i, :])
    init_x[1, :] = init_x_inter

    X_init = InitialConditions(init_x.T, interpolation_type=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)

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
    biorbd_model = biorbd.Model("ANsWER_Rleg_6dof_17muscle_0contact.bioMod")
    final_time = 0.37
    n_shooting_points = 50
    Gaitphase = 'swing'

    # Generate data from file
    from Marche_BiorbdOptim.LoadData import load_data_markers, load_data_q, load_data_emg
    name_subject = "equincocont01"
    t, markers_ref = load_data_markers(name_subject, biorbd_model, final_time, n_shooting_points, Gaitphase)
    q_ref = load_data_q(name_subject, biorbd_model, final_time, n_shooting_points, Gaitphase)
    emg_ref = load_data_emg(name_subject, biorbd_model, final_time, n_shooting_points, Gaitphase)
    excitations_ref = np.zeros((biorbd_model.nbMuscleTotal(), n_shooting_points))
    idx_emg = 0
    for i in range(biorbd_model.nbMuscleTotal()):
        if (i!=1) and (i!=2) and (i!=3) and (i!=5) and (i!=6) and (i!=11) and (i!=12):
            excitations_ref[i, :] = emg_ref[idx_emg, :-1]
            idx_emg += 1

    # Track these data
    biorbd_model = biorbd.Model("ANsWER_Rleg_6dof_17muscle_0contact.bioMod")  # To allow for non free variable, the model must be reloaded
    ocp = prepare_ocp(
        biorbd_model,
        final_time,
        n_shooting_points,
        markers_ref,
        excitations_ref,
        show_online_optim=True,
    )

    # --- Solve the program --- #
    sol = ocp.solve()

    # --- Get Results --- #
    states_sol, controls_sol = Data.get_data(ocp, sol["x"])
    q = states_sol["q"]
    q_dot = states_sol["q_dot"]
    activations = states_sol["muscles"]
    tau = controls_sol["tau"]
    excitations = controls_sol["muscles"]

    n_q = ocp.nlp[0]["model"].nbQ()
    n_qdot = ocp.nlp[0]["model"].nbQdot()
    n_mark = ocp.nlp[0]["model"].nbMarkers()
    n_frames = q.shape[1]

    # --- Plot --- #
    def plot_control(ax, t, x, color='b'):
        nbPoints = len(np.array(x))
        for n in range(nbPoints - 1):
            ax.plot([t[n], t[n + 1], t[n + 1]], [x[n], x[n], x[n + 1]], color)

    figure, axes = plt.subplots(2,3)
    axes = axes.flatten()
    for i in range(biorbd_model.nbQ()):
        name_dof = ocp.nlp[0]["model"].nameDof()[i].to_string()
        axes[i].set_title(name_dof)
        if (i > 1) :
            axes[i].plot(t, q[i, :]*180/np.pi)
            axes[i].plot(t, q_ref[i, :]*180/np.pi, 'r')
        else:
            axes[i].plot(t, q[i, :])
            axes[i].plot(t, q_ref[i, :], 'r')


    figure2, axes2 = plt.subplots(4, 5, sharex=True)
    axes2 = axes2.flatten()
    for i in range(biorbd_model.nbMuscleTotal()):
        name_mus = ocp.nlp[0]["model"].muscleNames()[i].to_string()
        plot_control(axes2[i], t[:-1], excitations_ref[i, :], color='r')
        plot_control(axes2[i], t[:-1], excitations[i, :-1])
        axes2[i].plot(t[:-1], activations[i, :-1])
        axes2[i].set_title(name_mus)

    plt.show()

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
    result.graphs()
