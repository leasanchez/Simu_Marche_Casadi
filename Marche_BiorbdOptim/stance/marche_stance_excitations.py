import numpy as np
from casadi import dot
import biorbd
from time import time
from Marche_BiorbdOptim.LoadData import Data_to_track

from biorbd_optim import (
    Instant,
    OptimalControlProgram,
    ProblemType,
    Objective,
    Bounds,
    QAndQDotBounds,
    InitialConditions,
    ShowResult,
    Data,
    InterpolationType,
    PlotType,
    Constraint,
)


def get_last_contact_forces(ocp, nlp, t, x, u, p, data_to_track=()):
    force = nlp["contact_forces_func"](x[-1], u[-1], p)
    val = force - data_to_track[t[-1], :]
    return dot(val, val)


def get_muscles_first_node(ocp, nlp, t, x, u, p):
    activation = x[0][2 * nlp["nbQ"] :]
    excitation = u[0][nlp["nbQ"] :]
    val = activation - excitation
    return val

def modify_isometric_force(biorbd_model, value):
    n_muscle = 0
    for nGrp in range(biorbd_model.nbMuscleGroups()):
        for nMus in range(biorbd_model.muscleGroup(nGrp).nbMuscles()):
            fiso_init = biorbd_model.muscleGroup(nGrp).muscle(nMus).characteristics().forceIsoMax().to_mx()
            biorbd_model.muscleGroup(nGrp).muscle(nMus).characteristics().setForceIsoMax(value[n_muscle] * fiso_init)
            n_muscle += 1

def prepare_ocp(
    biorbd_model, final_time, nb_shooting, markers_ref, excitation_ref, q_ref, grf_ref, nb_threads,
):
    # Problem parameters
    nb_q = biorbd_model.nbQ()
    nb_qdot = biorbd_model.nbQdot()
    nb_mus = biorbd_model.nbMuscleTotal()

    torque_min, torque_max, torque_init = -1000, 1000, 0
    activation_min, activation_max, activation_init = 0, 1, 0.1

    # Add objective functions
    objective_functions = (
        {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 1, "controls_idx": range(6, nb_q)},
        {"type": Objective.Lagrange.TRACK_MUSCLES_CONTROL, "weight": 1, "data_to_track": excitation_ref[:, :-1].T},
        {"type": Objective.Lagrange.TRACK_MARKERS, "weight": 100, "data_to_track": markers_ref},
        # {"type": Objective.Lagrange.TRACK_STATE, "weight": 0.01, "states_idx": [0, 1, 5, 8, 9, 10], "data_to_track": q_ref.T},
        {"type": Objective.Lagrange.TRACK_CONTACT_FORCES, "weight": 0.00005, "data_to_track": grf_ref.T},
        {
            "type": Objective.Mayer.CUSTOM,
            "weight": 0.00005,
            "function": get_last_contact_forces,
            "data_to_track": grf_ref.T,
            "instant": Instant.ALL,
        },
    )

    # Dynamics
    variable_type = ProblemType.muscle_excitations_and_torque_driven_with_contact

    # Constraints
    constraints = {"type": Constraint.CUSTOM, "function": get_muscles_first_node, "instant": Instant.START}

    # Path constraint
    X_bounds = QAndQDotBounds(biorbd_model)
    X_bounds.concatenate(
        Bounds([activation_min] * biorbd_model.nbMuscles(), [activation_max] * biorbd_model.nbMuscles())
    )

    # Initial guess
    init_x = np.zeros((biorbd_model.nbQ() + biorbd_model.nbQdot() + biorbd_model.nbMuscleTotal(), nb_shooting + 1))
    for i in range(nb_shooting + 1):
        init_x[[0, 1, 5, 8, 9, 10], i] = q_ref[:, i]
        init_x[-biorbd_model.nbMuscleTotal() :, i] = excitation_ref[:, i]
    X_init = InitialConditions(init_x, interpolation_type=InterpolationType.EACH_FRAME)

    # Define control path constraint
    U_bounds = Bounds(
        [torque_min] * biorbd_model.nbGeneralizedTorque() + [activation_min] * biorbd_model.nbMuscleTotal(),
        [torque_max] * biorbd_model.nbGeneralizedTorque() + [activation_max] * biorbd_model.nbMuscleTotal(),
    )
    # Initial guess
    init_u = np.zeros((biorbd_model.nbGeneralizedTorque() + biorbd_model.nbMuscleTotal(), nb_shooting))
    for i in range(nb_shooting):
        init_u[1, i] = -500
        init_u[-biorbd_model.nbMuscleTotal() :, i] = excitation_ref[:, i]
    U_init = InitialConditions(init_u, interpolation_type=InterpolationType.EACH_FRAME)

    # Define the parameter to optimize
    # Give the parameter some min and max bounds
    bound_length = Bounds(min_bound=np.repeat(0.2, nb_mus), max_bound=np.repeat(5, nb_mus), interpolation_type=InterpolationType.CONSTANT)
    parameters = {
        "name": "force_isometric",  # The name of the parameter
        "function": modify_isometric_force,  # The function that modifies the biorbd model
        "bounds": bound_length,  # The bounds
        "initial_guess": InitialConditions(np.repeat(1, nb_mus)),  # The initial guess
        "size": nb_mus,  # The number of elements this particular parameter vector has
    }

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        variable_type,
        nb_shooting,
        final_time,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        objective_functions,
        constraints,
        nb_threads=nb_threads,
        parameters=parameters,
    )


if __name__ == "__main__":
    # Define the problem
    biorbd_model = biorbd.Model("../../ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact_deGroote_3d.bioMod")
    model_q = biorbd.Model("../../ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact_deGroote.bioMod")
    n_shooting_points = 25
    Gaitphase = "stance"

    # Generate data from file
    Data_to_track = Data_to_track(name_subject="equincocont01")
    [T, T_stance, T_swing] = Data_to_track.GetTime()
    final_time = T_stance

    grf_ref = Data_to_track.load_data_GRF(biorbd_model, T_stance, n_shooting_points)  # get ground reaction forces
    markers_ref = Data_to_track.load_data_markers(
        biorbd_model, T_stance, n_shooting_points, "stance"
    )  # get markers position
    q_ref = Data_to_track.load_data_q(biorbd_model, T_stance, n_shooting_points, "stance")  # get q from kalman
    emg_ref = Data_to_track.load_data_emg(biorbd_model, T_stance, n_shooting_points, "stance")  # get emg
    excitation_ref = Data_to_track.load_muscularExcitation(emg_ref)

    # Track these data
    ocp = prepare_ocp(
        biorbd_model,
        final_time,
        n_shooting_points,
        markers_ref,
        excitation_ref=excitation_ref,
        q_ref=q_ref,
        grf_ref=grf_ref,
        nb_threads=4,
    )
    ocp.add_plot("q", lambda x, u: q_ref, PlotType.STEP, axes_idx=[0, 1, 5, 8, 9, 10])

    # --- Solve the program --- #
    tic = time()
    sol = ocp.solve(
        solver="ipopt",
        options_ipopt={
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
    states_sol, controls_sol, params_sol = Data.get_data(ocp, sol, get_parameters=True)
    q = states_sol["q"]
    q_dot = states_sol["q_dot"]
    activations = states_sol["muscles"]
    tau = controls_sol["tau"]
    excitations = controls_sol["muscles"]

    n_q = ocp.nlp[0]["model"].nbQ()
    n_qdot = ocp.nlp[0]["model"].nbQdot()
    nb_marker = biorbd_model.nbMarkers()
    n_mus = ocp.nlp[0]["model"].nbMuscleTotal()
    n_frames = q.shape[1]

    # --- Save the optimal control program and the solution --- #
    ocp.save(sol, "marche_stance_excitation")

    # # --- Load the optimal control program and the solution --- #
    # ocp_load, sol_load = OptimalControlProgram.load("marche_stance_excitation.bo")
    # result = ShowResult(ocp_load, sol_load)

    # --- Show results --- #
    ShowResult(ocp, sol).animate(nb_frames=200)
