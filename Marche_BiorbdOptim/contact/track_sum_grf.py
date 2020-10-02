import numpy as np
from casadi import dot, Function, vertcat, MX, mtimes, nlpsol
import biorbd
import BiorbdViz as BiorbdViz
from matplotlib import pyplot as plt
from Marche_BiorbdOptim.LoadData import Data_to_track

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
    ParameterList,
    Instant,
    ConstraintList,
    Constraint,
    Solver,
)

# --- fcn contact talon ---
def get_last_contact_forces_contact_talon(ocp, nlp, t, x, u, p, grf):
    force = nlp["contact_forces_func"](x[-1], u[-1], p)
    val = grf[0, t[-1]] - (force[0])
    val = vertcat(val, grf[1, t[-1]] - force[1])
    val = vertcat(val, grf[2, t[-1]] - (force[2]))
    return dot(val, val)

def track_sum_contact_forces_contact_talon(ocp, nlp, t, x, u, p, grf, target=()):
    ns = nlp["ns"]
    val = []
    for n in range(ns):
        force = nlp["contact_forces_func"](x[n], u[n], p)
        val = vertcat(val, grf[0, t[n]] - (force[0]))
        val = vertcat(val, grf[1, t[n]] - force[1])
        val = vertcat(val, grf[2, t[n]] - (force[2]))
    return dot(val, val)

# --- fcn flatfoot ---
def get_last_contact_forces_flatfoot(ocp, nlp, t, x, u, p, grf):
    force = nlp["contact_forces_func"](x[-1], u[-1], p)
    val = grf[0, t[-1]] - (force[0] + force[4])
    val = vertcat(val, grf[1, t[-1]] - force[1])
    val = vertcat(val, grf[2, t[-1]] - (force[2] + force[3] + force[5]))
    val = vertcat(val, force[2])
    val = vertcat(val, force[0]) # minimise contact talon ?
    return dot(val, val)

def track_sum_contact_forces_flatfoot(ocp, nlp, t, x, u, p, grf, target=()):
    ns = nlp["ns"]
    val = []
    for n in range(ns):
        force = nlp["contact_forces_func"](x[n], u[n], p)
        val = vertcat(val, grf[0, t[n]] - (force[0] + force[4]))
        val = vertcat(val, grf[1, t[n]] - force[1])
        val = vertcat(val, grf[2, t[n]] - (force[2] + force[3] + force[5]))
    return dot(val, val)

# --- fcn forefoot ---
def get_last_contact_forces_forefoot(ocp, nlp, t, x, u, p, grf):
    force = nlp["contact_forces_func"](x[-1], u[-1], p)
    return dot(force, force)

def track_sum_contact_forces_forefoot(ocp, nlp, t, x, u, p, grf, target=()):
    ns = nlp["ns"]
    val = []
    for n in range(ns):
        force = nlp["contact_forces_func"](x[n], u[n], p)
        val = vertcat(val, grf[0, t[n]] - (force[0] + force[2]))
        val = vertcat(val, grf[1, t[n]] - force[3])
        val = vertcat(val, grf[2, t[n]] - (force[1] + force[4]))
    return dot(val, val)



def plot_control(ax, t, x, color="b"):
    nbPoints = len(np.array(x))
    for n in range(nbPoints - 1):
        ax.plot([t[n], t[n + 1], t[n + 1]], [x[n], x[n], x[n + 1]], color)


def prepare_ocp(
    biorbd_model, final_time, nb_shooting, markers_ref, grf_ref, q_ref, qdot_ref, nb_threads,
):

    # Problem parameters
    nb_q = biorbd_model.nbQ()
    nb_qdot = biorbd_model.nbQdot()
    nb_tau = biorbd_model.nbGeneralizedTorque()

    torque_min, torque_max, torque_init = -1000, 1000, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    # objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=1, controls_idx=range(6, nb_q), phase=0)
    objective_functions.add(Objective.Lagrange.TRACK_STATE, weight=200, states_idx=range(nb_q), target=q_ref, phase=0)
    objective_functions.add(track_sum_contact_forces_flatfoot, grf=grf_ref, custom_type=Objective.Mayer, instant=Instant.ALL, weight=0.001)
    objective_functions.add(get_last_contact_forces_flatfoot, custom_type=Objective.Mayer, instant=Instant.ALL, grf=grf_ref, weight=0.001)

    # Dynamics
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.TORQUE_DRIVEN_WITH_CONTACT, phase=0)

    # Constraints
    constraints = ConstraintList()
    constraints.add(
        Constraint.CONTACT_FORCE_INEQUALITY,
        direction="GREATER_THAN",
        instant=Instant.ALL,
        contact_force_idx=(2, 3, 5),
        boundary=50,
    )
    # constraints.add(
    #     get_last_contact_forces,
    #     instant=Instant.ALL,
    #     grf=grf_ref,
    # )
    # constraints.add(
    #     Constraint.NON_SLIPPING,
    #     instant=Instant.ALL,
    #     normal_component_idx=5,
    #     tangential_component_idx=4,
    #     static_friction_coefficient=0.2,
    # )

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
    init_x[nb_q:nb_q + nb_qdot, :] = qdot_ref # np.load("./RES/1leg/flatfoot/q_dot.npy")
    x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)

    u_init = InitialConditionsList()
    init_u = np.zeros((nb_q, nb_shooting)) # np.load("./RES/1leg/flatfoot/tau.npy")[:, :-1]
    u_init.add(init_u, interpolation=InterpolationType.EACH_FRAME)

    # ------------- #

    return OptimalControlProgram(
        biorbd_model=biorbd_model,
        dynamics_type=dynamics,
        number_shooting_points=nb_shooting,
        phase_time=final_time,
        X_init=x_init,
        U_init=u_init,
        X_bounds=x_bounds,
        U_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        nb_threads=nb_threads,
    )


if __name__ == "__main__":
    # Define the problem --- Model path
    biorbd_model = (
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_1contact_deGroote_3d_Heel.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_3contacts_deGroote_3d.bioMod"),
        biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_contact_deGroote_3d_Forefoot.bioMod"),
    )

    # Problem parameters
    dt = 0.01
    nb_q = biorbd_model[0].nbQ()
    nb_qdot = biorbd_model[0].nbQdot()
    nb_tau = biorbd_model[0].nbGeneralizedTorque()

    # Generate data from file
    Data_to_track = Data_to_track("normal02", model=biorbd_model[2],multiple_contact=True, two_leg=False)
    phase_time = Data_to_track.GetTime()
    number_shooting_points = []
    for time in phase_time:
        number_shooting_points.append(int(time/0.01))
    grf_ref = Data_to_track.load_data_GRF(number_shooting_points)  # get ground reaction forces
    markers_ref = Data_to_track.load_data_markers(number_shooting_points)
    q_ref = Data_to_track.load_q_kalman(number_shooting_points)
    qdot_ref = Data_to_track.load_qdot_kalman(number_shooting_points)

    ocp = prepare_ocp(
        biorbd_model=biorbd_model[1],
        final_time=phase_time[1],
        nb_shooting=number_shooting_points[1],
        markers_ref=markers_ref[1],
        grf_ref=grf_ref[1],
        q_ref=q_ref[1],
        qdot_ref=qdot_ref[1],
        nb_threads=4,
    )

    tau = np.load("./RES/1leg/flatfoot/tau.npy")
    q_dot = np.load("./RES/1leg/flatfoot/q_dot.npy")
    q = np.load("./RES/1leg/flatfoot/q.npy")

    # --- Solve the program --- #
    sol = ocp.solve(
        solver=Solver.IPOPT,
        solver_options={
            "ipopt.tol": 1e-3,
            "ipopt.max_iter": 5000,
            "ipopt.hessian_approximation": "exact",
            "ipopt.limited_memory_max_history": 50,
            "ipopt.linear_solver": "ma57",
        },
        show_online_optim=True,
    )

    # --- Get Results --- #
    states, controls = Data.get_data(ocp, sol)
    q, q_dot, tau = (
        states["q"],
        states["q_dot"],
        controls["tau"],
    )

    # # --- Save Results --- #
    # np.save("./RES/1leg/flatfoot/tau", tau)
    # np.save("./RES/1leg/flatfoot/q_dot", q_dot)
    # np.save("./RES/1leg/flatfoot/q", q)

    # --- plot grf ---
    forces_sim = np.zeros((biorbd_model[1].nbContacts(), number_shooting_points[1] + 1))
    for n in range(number_shooting_points[1] + 1):
        forces_sim[:, n:n+1] = ocp.nlp[0]['contact_forces_func'](np.concatenate([q[:, n], q_dot[:, n]]), tau[:, n], 0)

    figure, axes = plt.subplots(1, 3)
    axes = axes.flatten()
    axes[0].plot(forces_sim[0, :] + forces_sim[2, :])
    axes[0].plot(grf_ref[1][0, :], 'k--')
    axes[0].set_title('forces in x (N)')

    axes[1].plot(forces_sim[3, :])
    axes[1].plot(grf_ref[1][1, :], 'k--')
    axes[1].set_title('forces in y (N)')

    axes[2].plot(forces_sim[1, :] + forces_sim[4, :])
    axes[2].plot(grf_ref[1][2, :], 'k--')
    axes[2].set_title('forces in z (N)')

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
    result.graphs()