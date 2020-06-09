import numpy as np
from scipy.integrate import solve_ivp
from casadi import dot, Function, vertcat, MX, tanh
from Marche_BiorbdOptim.LoadData import Data_to_track
import biorbd

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
)


def get_last_contact_forces(ocp, nlp, t, x, u, p, data_to_track=()):
    force = nlp["contact_forces_func"](x[-1], u[-1], p)
    val = force - data_to_track[t[-1], :]
    return dot(val, val)

def modify_isometric_force(biorbd_model, value):
    n_muscle = 0
    for nGrp in range(biorbd_model.nbMuscleGroups()):
        for nMus in range(biorbd_model.muscleGroup(nGrp).nbMuscles()):
            fiso_init = biorbd_model.muscleGroup(nGrp).muscle(nMus).characteristics().forceIsoMax().to_mx()
            biorbd_model.muscleGroup(nGrp).muscle(nMus).characteristics().setForceIsoMax(value[n_muscle] * fiso_init)
            n_muscle += 1

def generate_activation(biorbd_model, final_time, nb_shooting, emg_ref):
    # Aliases
    nb_mus = biorbd_model.nbMuscleTotal()
    nb_musclegrp = biorbd_model.nbMuscleGroups()
    dt = final_time / nb_shooting

    # init
    ta = td = []
    activation_ref = np.ndarray((nb_mus, nb_shooting + 1))

    for n_grp in range(nb_musclegrp):
        for n_muscle in range(biorbd_model.muscleGroup(n_grp).nbMuscles()):
            ta.append(biorbd_model.muscleGroup(n_grp).muscle(n_muscle).characteristics().torqueActivation().to_mx())
            td.append(biorbd_model.muscleGroup(n_grp).muscle(n_muscle).characteristics().torqueDeactivation().to_mx())

    def compute_activationDot(a, e, ta, td):
        activationDot = []
        for i in range(nb_mus):
            f = 0.5 * tanh(0.1 * (e[i] - a[i]))
            da = (f + 0.5) / (ta[i] * (0.5 + 1.5 * a[i]))
            dd = (-f + 0.5) * (0.5 + 1.5 * a[i]) / td[i]
            activationDot.append((da + dd) * (e[i] - a[i]))
        return vertcat(*activationDot)

    # casadi
    symbolic_states = MX.sym("a", nb_mus, 1)
    symbolic_controls = MX.sym("e", nb_mus, 1)
    dynamics_func = Function(
        "ActivationDyn",
        [symbolic_states, symbolic_controls],
        [compute_activationDot(symbolic_states, symbolic_controls, ta, td)],
        ["a", "e"],
        ["adot"],
    ).expand()

    def dyn_interface(t, a, e):
        return np.array(dynamics_func(a, e)).squeeze()

    # Integrate and collect the position of the markers accordingly
    activation_init = emg_ref[:, 0]
    activation_ref[:, 0] = activation_init
    sol_act = []
    for i in range(nb_shooting):
        e = emg_ref[:, i]
        sol = solve_ivp(dyn_interface, (0, dt), activation_init, method="RK45", args=(e,))
        sol_act.append(sol["y"])
        activation_init = sol["y"][:, -1]
        activation_ref[:, i + 1] = activation_init

    return activation_ref


def prepare_ocp(
    biorbd_model, final_time, nb_shooting, markers_ref, activation_ref, grf_ref, q_ref,
):
    # Problem parameters
    nb_phases = len(biorbd_model)
    nb_q = biorbd_model[0].nbQ()
    nb_mus = biorbd_model[0].nbMuscleTotal()
    torque_min, torque_max, torque_init = -5000, 5000, 0
    activation_min, activation_max, activation_init = 0, 1, 0.1

    # Add objective functions
    objective_functions = (
        (
            {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 1, "controls_idx": range(6, nb_q)},
            {"type": Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, "weight": 0.1, "data_to_track": activation_ref[0].T},
            {"type": Objective.Lagrange.TRACK_MARKERS, "weight": 100, "data_to_track": markers_ref[0]},
            {"type": Objective.Lagrange.TRACK_CONTACT_FORCES, "weight": 0.00005, "data_to_track": grf_ref[:, :-1].T},
            {"type": Objective.Mayer.CUSTOM, "function": get_last_contact_forces, "data_to_track": grf_ref.T,
             "weight": 0.00005, "instant": Instant.ALL}
        ),
        (
            {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 1, "controls_idx": range(6, nb_q)},
            {"type": Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, "weight": 0.1, "data_to_track": activation_ref[1].T},
            {"type": Objective.Lagrange.TRACK_MARKERS, "weight": 100, "data_to_track": markers_ref[1]},
        ),
    )

    # Dynamics
    problem_type = (
        ProblemType.muscles_activations_and_torque_driven_with_contact,
        ProblemType.muscle_activations_and_torque_driven,
    )

    # Constraints
    constraints = ()

    # Path constraint
    X_bounds = [QAndQDotBounds(biorbd_model[i]) for i in range(nb_phases)]

    # Initial guess
    init_x = []
    for i in range(nb_phases):
        init_x_s = np.zeros((biorbd_model[0].nbQ() + biorbd_model[0].nbQdot(), nb_shooting[0] + 1))
        init_x_s[[0, 1, 5, 8, 9, 11], :] = q_ref[i]
        init_x.append(init_x_s)

    X_init = [
        InitialConditions(init_x[0], interpolation_type=InterpolationType.EACH_FRAME),
        InitialConditions(init_x[1], interpolation_type=InterpolationType.EACH_FRAME),
    ]

    # Define control path constraint
    U_bounds = [
        Bounds(
            min_bound=[torque_min] * biorbd_model[i].nbGeneralizedTorque()
            + [activation_min] * biorbd_model[i].nbMuscleTotal(),
            max_bound=[torque_max] * biorbd_model[i].nbGeneralizedTorque()
            + [activation_max] * biorbd_model[i].nbMuscleTotal(),
        )
        for i in range(nb_phases)
    ]

    init_u = []
    for i in range(nb_phases):
        init_u_s = np.zeros((nb_q + nb_mus, nb_shooting[i]))
        if (i == 0) :
            init_u_s[1, i] = -500
        init_u_s[-nb_mus:, :] = activation_ref[i][:, :-1]
        init_u.append(init_u_s)

    U_init = [
        InitialConditions(init_u[0], interpolation_type=InterpolationType.EACH_FRAME),
        InitialConditions(init_u[1], interpolation_type=InterpolationType.EACH_FRAME),
    ]

    # Define the parameter to optimize
    bound_length = Bounds(min_bound=np.repeat(0.2, nb_mus), max_bound=np.repeat(5, nb_mus), interpolation_type=InterpolationType.CONSTANT)
    parameter = ({
        "name": "force_isometric",  # The name of the parameter
        "function": modify_isometric_force,  # The function that modifies the biorbd model
        "bounds": bound_length,  # The bounds
        "initial_guess": InitialConditions(np.repeat(1, nb_mus)),  # The initial guess
        "size": nb_mus,  # The number of elements this particular parameter vector has
    }, )
    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        problem_type,
        nb_shooting,
        final_time,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        objective_functions,
        constraints,
        parameters=(parameter, parameter),
    )


if __name__ == "__main__":
    # Model path
    biorbd_model = (
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_1contact_deGroote_3d_Forefoot.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_0contact_deGroote_3d.bioMod"),
    )

    # Problem parameters
    number_shooting_points = [25, 25]

    Data_to_track = Data_to_track("equincocont01", multiple_contact=False)
    [T, T_stance, T_swing] = Data_to_track.GetTime()
    phase_time = [T_stance, T_swing]  # get time for each phase

    grf_ref = Data_to_track.load_data_GRF(
        biorbd_model[0], T_stance, number_shooting_points[0]
    )  # get ground reaction forces

    markers_ref = []
    markers_ref.append(Data_to_track.load_data_markers(biorbd_model[0], T_stance, number_shooting_points[0], "stance"))
    markers_ref.append(
        Data_to_track.load_data_markers(biorbd_model[-1], phase_time[-1], number_shooting_points[-1], "swing")
    )  # get markers position

    q_ref = []
    model_q = biorbd.Model("../../ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact_deGroote.bioMod")
    q_ref.append(Data_to_track.load_data_q(model_q, T_stance, number_shooting_points[0], "stance"))
    q_ref.append(
        Data_to_track.load_data_q(model_q, phase_time[-1], number_shooting_points[-1], "swing")
    )  # get q from kalman

    emg_ref = []
    emg_ref.append(Data_to_track.load_data_emg(biorbd_model[0], T_stance, number_shooting_points[0], "stance"))
    emg_ref.append(
        Data_to_track.load_data_emg(biorbd_model[-1], phase_time[-1], number_shooting_points[-1], "swing")
    )  # get emg

    activation_ref = []
    excitation_ref = []
    for i in range(len(phase_time)):
        excitation_ref.append(Data_to_track.load_muscularExcitation(emg_ref[i]))
        activation_ref.append(generate_activation(biorbd_model[i], final_time=phase_time[i], nb_shooting=number_shooting_points[i], emg_ref=excitation_ref[i]))

    biorbd_model = (
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_1contact_deGroote_3d_Forefoot.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_0contact_deGroote_3d.bioMod"),
    )

    ocp = prepare_ocp(
        biorbd_model,
        phase_time,
        number_shooting_points,
        markers_ref=markers_ref,
        activation_ref=activation_ref,
        grf_ref=grf_ref,
        q_ref=q_ref,
    )

    # --- Solve the program --- #
    sol = ocp.solve(
        solver="ipopt",
        options_ipopt={
            "ipopt.tol": 1e-3,
            "ipopt.max_iter": 5000,
            "ipopt.hessian_approximation": "limited-memory",
            "ipopt.limited_memory_max_history": 50,
            "ipopt.linear_solver": "ma57",
        },
        show_online_optim=False,
    )

    ocp.save(sol, "marche_gait_equin_activation")
    # --- Show results --- #
    ShowResult(ocp, sol).animate()
    # result.graphs()
