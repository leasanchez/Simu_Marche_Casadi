import numpy as np
from casadi import dot, Function, vertcat, MX, mtimes, nlpsol
import biorbd
from matplotlib import pyplot as plt
from Marche_BiorbdOptim.LoadData import Data_to_track

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
    Data,
    InterpolationType,
    StateTransition,
    OdeSolver,
)

def get_dispatch_contact_forces(grf_ref, M_ref, coord, nb_shooting):
    p = 0.5  # repartition entre les 2 points

    F_Meta1 = MX.sym("F_Meta1", 3 * (nb_shooting + 1), 1)
    F_Meta5 = MX.sym("F_Meta1", 3 * (nb_shooting + 1), 1)
    M_Meta1 = MX.sym("M_Meta1", 3 * (nb_shooting + 1), 1)
    M_Meta5 = MX.sym("M_Meta1", 3 * (nb_shooting + 1), 1)

    objective = 0
    lbg = []
    ubg = []
    constraint = []
    for i in range(nb_shooting + 1):
        # Aliases
        fm1 = F_Meta1[3 * i: 3 * (i + 1)]
        fm5 = F_Meta5[3 * i: 3 * (i + 1)]
        mm1 = M_Meta1[3 * i: 3 * (i + 1)]
        mm5 = M_Meta5[3 * i: 3 * (i + 1)]

        # sum forces = 0 --> Fp1 + Fp2 = Ftrack
        sf = fm1 + fm5
        jf = sf - grf_ref[:, i]
        objective += 100 * mtimes(jf.T, jf)

        # sum moments = 0 --> Mp1_P1 + CP1xFp1 + Mp2_P2 + CP2xFp2 = Mtrack
        sm = mm1 + dot(coord[0], fm1) + mm5 + dot(coord[1], fm5)
        jm = sm - M_ref[:, i]
        objective += 100 * mtimes(jm.T, jm)

        # use of p to dispatch forces --> p*Fp1 - (1-p)*Fp2 = 0
        jf2 = p * fm1 - (1 - p) * fm5
        objective += mtimes(jf2.T, jf2)

        # use of p to dispatch moments
        jm2 = p * (mm1 + dot(coord[0], fm1)) - (1 - p) * (mm5 + dot(coord[1], fm5))
        objective += mtimes(jm2.T, jm2)

        # positive vertical force
        constraint += (fm1[2], fm5[2])
        lbg += [0] * 2
        ubg += [1000] * 2

        # non slipping --> -0.4*Fz < Fx < 0.4*Fz
        constraint += ((-0.4 * fm1[2] - fm1[0]), (-0.4 * fm5[2] - fm5[0]))
        lbg += [-1000] * 2
        ubg += [0] * 2

        constraint += ((0.4 * fm1[2] - fm1[0]), (0.4 * fm5[2] - fm5[0]))
        lbg += [0] * 2
        ubg += [1000] * 2

    w = [F_Meta1, F_Meta5, M_Meta1, M_Meta5]
    nlp = {'x': vertcat(*w), 'f': objective, 'g': vertcat(*constraint)}
    opts = {"ipopt.tol": 1e-8, "ipopt.hessian_approximation": "exact"}
    solver = nlpsol("solver", "ipopt", nlp, opts)
    res = solver(x0=np.zeros(6 * 2 * (number_shooting_points[2] + 1)),
                 lbx=-1000,
                 ubx=1000,
                 lbg=lbg,
                 ubg=ubg)

    FM1 = res['x'][:3 * (nb_shooting + 1)]
    FM5 = res['x'][3 * (nb_shooting + 1): 6 * (nb_shooting + 1)]

    grf_dispatch_ref = np.zeros((3 * 2, nb_shooting + 1))
    for i in range(3):
        grf_dispatch_ref[i, :] = np.array(FM1[i::3]).squeeze()
        grf_dispatch_ref[i + 3, :] = np.array(FM5[i::3]).squeeze()
    return grf_dispatch_ref

def get_last_contact_forces(ocp, nlp, t, x, u, p, data_to_track=()):
    force = nlp["contact_forces_func"](x[-1], u[-1], p)
    val = force - data_to_track[t[-1], :]
    return dot(val, val)


def get_muscles_first_node(ocp, nlp, t, x, u, p):
    activation = x[0][2 * nlp["nbQ"] :]
    excitation = u[0][nlp["nbQ"] :]
    val = activation - excitation
    return val


def prepare_ocp(
    biorbd_model, final_time, nb_shooting, markers_ref, excitation_ref, grf_ref, q_ref,nb_threads,
):

    # Problem parameters
    nb_q = biorbd_model.nbQ()
    nb_qdot = biorbd_model.nbQdot()
    nb_tau = biorbd_model.nbGeneralizedTorque()
    nb_mus = biorbd_model.nbMuscleTotal()

    torque_min, torque_max, torque_init = -1000, 1000, 0
    activation_min, activation_max, activation_init = 0, 1, 0.1

    # Add objective functions
    objective_functions = (
            {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 1, "controls_idx": range(6, nb_tau)},
            {"type": Objective.Lagrange.TRACK_MUSCLES_CONTROL, "weight": 0.1, "data_to_track": excitation_ref[:, :-1].T,},
            {"type": Objective.Lagrange.TRACK_MARKERS, "weight": 100, "data_to_track": markers_ref},
            {"type": Objective.Lagrange.TRACK_STATE, "weight": 0.01, "states_idx": [0, 1, 5, 8, 9, 11], "data_to_track": q_ref.T},
            {"type": Objective.Lagrange.TRACK_CONTACT_FORCES, "weight": 0.00005, "data_to_track": grf_ref.T},
        )

    # Dynamics
    problem_type = (
        ProblemType.muscle_excitations_and_torque_driven_with_contact,
    )

    # Constraints
    constraints = ()

    # Path constraint
    X_bounds = QAndQDotBounds(biorbd_model)
    X_bounds.concatenate(
            Bounds([activation_min] * biorbd_model.nbMuscles(), [activation_max] * biorbd_model.nbMuscles())
        )

    # Initial guess
    init_x = np.zeros((nb_q + nb_qdot + nb_mus, nb_shooting + 1,))
    # init_x[[0, 1, 5, 8, 9, 11], :] = q_ref
    init_x[:nb_q, :] = q_ref
    init_x[-nb_mus:, :] = excitation_ref
    X_init = InitialConditions(init_x, interpolation_type=InterpolationType.EACH_FRAME)

    # Define control path constraint
    U_bounds = Bounds(
            min_bound=[torque_min] * nb_tau + [activation_min] * nb_mus,
            max_bound=[torque_max] * nb_tau + [activation_max] * nb_mus,
        )

    # Initial guess
    init_u = np.zeros((nb_tau + nb_mus, nb_shooting))
    init_u[1, :] = np.repeat(-500, nb_shooting)
    init_u[-nb_mus :, :] = excitation_ref[:, :-1]
    U_init = InitialConditions(init_u, interpolation_type=InterpolationType.EACH_FRAME)

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
        nb_threads=nb_threads,
    )


if __name__ == "__main__":
    # Define the problem
    # Model path
    biorbd_model = (
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_1contact_deGroote_3d_Heel.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_3contacts_deGroote_3d.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_contact_deGroote_3d_Forefoot.bioMod"),
  )

    # Problem parameters
    number_shooting_points = [5, 10, 15]

    # Generate data from file
    Data_to_track = Data_to_track("normal01", multiple_contact=True)
    [T, T_stance, T_swing] = Data_to_track.GetTime()
    phase_time = [T_stance[0], T_stance[1], T_stance[2]]  # get time for each phase

    grf_ref = Data_to_track.load_data_GRF(biorbd_model[0], T_stance, number_shooting_points)  # get ground reaction forces
    M_ref = Data_to_track.load_data_Moment(biorbd_model[0], T_stance, number_shooting_points)
    markers_ref = Data_to_track.load_data_markers(biorbd_model[0], T_stance, number_shooting_points, "stance")
    q_ref = Data_to_track.load_data_q(biorbd_model[0], T_stance, number_shooting_points, "stance")
    emg_ref = Data_to_track.load_data_emg(biorbd_model[0], T_stance, number_shooting_points, "stance")
    excitation_ref = []
    for i in range(len(phase_time)):
        excitation_ref.append(Data_to_track.load_muscularExcitation(emg_ref[i]))

    Meta1 = np.array([np.mean(markers_ref[2][0, 21, :]), np.mean(markers_ref[2][1, 21, :]), 0])
    Meta5 = np.array([np.mean(markers_ref[2][0, 24, :]), np.mean(markers_ref[2][1, 24, :]), 0])
    grf_dispatch_ref = get_dispatch_contact_forces(grf_ref[2], M_ref[2], [Meta1, Meta5], number_shooting_points[2])

    Q_ref = np.zeros((biorbd_model[2].nbQ(), number_shooting_points[2] + 1))
    Q_ref[[0, 1, 5, 8, 9, 11],:] = q_ref[2]

    ocp = prepare_ocp(
        biorbd_model=biorbd_model[2],
        final_time=phase_time[2],
        nb_shooting=number_shooting_points[2],
        markers_ref=markers_ref[2],
        excitation_ref=excitation_ref[2],
        grf_ref=grf_dispatch_ref[[0, 2, 3, 5], :],
        q_ref=Q_ref,
        nb_threads=4,
    )

    # --- Solve the program --- #
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

    # --- Get Results --- #
    states_sol, controls_sol = Data.get_data(ocp, sol["x"])
    q = states_sol["q"]
    q_dot = states_sol["q_dot"]
    activations = states_sol["muscles"]
    tau = controls_sol["tau"]
    excitations = controls_sol["muscles"]

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
    result.graphs()
