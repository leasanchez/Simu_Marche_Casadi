import numpy as np
from casadi import MX, Function, vertcat
from matplotlib import pyplot as plt
import biorbd
from time import time
from Marche_BiorbdOptim.LoadData import Data_to_track

from biorbd_optim import (
    OptimalControlProgram,
    ProblemType,
    Objective,
    Bounds,
    QAndQDotBounds,
    InitialConditions,
    ShowResult,
    Data,
    InterpolationType,
    Axe,
    Constraint,
    PlotType,
    Instant,
)


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
    biorbd_model, final_time, nb_shooting, markers_ref, q_ref, excitations_ref, nb_threads,
):
    # Problem parameters
    nb_q = biorbd_model.nbQ()
    nb_qdot = biorbd_model.nbQdot()
    nb_mus = biorbd_model.nbMuscleTotal()
    nb_tau = biorbd_model.nbGeneralizedTorque()

    torque_min, torque_max, torque_init = -5000, 5000, 0
    activation_min, activation_max, activation_init = 0, 1, 0.1

    # Add objective functions
    objective_functions = (
        {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 1, "controls_idx": range(6, 11)},
        {"type": Objective.Lagrange.TRACK_MUSCLES_CONTROL, "weight": 0.1, "data_to_track": excitations_ref[:, :-1].T},
        # {"type": Objective.Lagrange.TRACK_STATE, "weight": 5, "states_idx": [0, 1, 5, 8, 9, 10], "data_to_track": q_ref.T},
        {"type": Objective.Lagrange.TRACK_MARKERS, "weight": 100, "data_to_track": markers_ref},
    )

    # Dynamics
    variable_type = ProblemType.muscle_excitations_and_torque_driven

    # Constraints
    constraints = {"type": Constraint.CUSTOM, "function": get_muscles_first_node, "instant": Instant.START}

    # Path constraint
    X_bounds = QAndQDotBounds(biorbd_model)
    X_bounds.concatenate(
        Bounds([activation_min] * nb_mus, [activation_max] * nb_mus)
    )

    # Initial guess
    init_x = np.zeros((nb_q + nb_qdot + nb_mus, nb_shooting + 1))
    for i in range(nb_shooting + 1):
        init_x[[0, 1, 5, 8, 9, 10], i] = q_ref[:, i]
        init_x[-nb_mus:, i] = excitations_ref[:, i]
    X_init = InitialConditions(init_x, interpolation_type=InterpolationType.EACH_FRAME)

    # Define control path constraint
    U_bounds = Bounds(
        [torque_min] * nb_tau + [activation_min] * nb_mus,
        [torque_max] * nb_tau + [activation_max] * nb_mus,
    )
    # Initial guess
    init_u = np.zeros((nb_tau + nb_mus, nb_shooting))
    for i in range(nb_shooting):
        init_u[-nb_mus :, i] = excitations_ref[:, i]
    U_init = InitialConditions(init_u, interpolation_type=InterpolationType.EACH_FRAME)

    # Define the parameter to optimize
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
    final_time = 0.37
    n_shooting_points = 25
    Gaitphase = "swing"

    # Generate data from file
    Data_to_track = Data_to_track(name_subject="equincocont01")
    [T, T_stance, T_swing] = Data_to_track.GetTime()
    final_time = T_swing

    markers_ref = Data_to_track.load_data_markers(
        biorbd_model, T_stance, n_shooting_points, "swing"
    )  # get markers position
    q_ref = Data_to_track.load_data_q(biorbd_model, T_stance, n_shooting_points, "swing")  # get q from kalman
    emg_ref = Data_to_track.load_data_emg(biorbd_model, T_stance, n_shooting_points, "swing")  # get emg
    excitation_ref = Data_to_track.load_muscularExcitation(emg_ref)

    # Track these data
    ocp = prepare_ocp(biorbd_model, final_time, n_shooting_points, markers_ref, q_ref, excitation_ref, nb_threads=4,)
    # --- Add plot kalman --- #
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
        show_online_optim=True,
    )
    toc = time() - tic
    print(f"Time to solve : {toc}sec")

    # --- Save the optimal control program and the solution --- #
    ocp.save(sol, "marche_swing_excitation")

    # --- Show results --- #
    ShowResult(ocp, sol).animate()

