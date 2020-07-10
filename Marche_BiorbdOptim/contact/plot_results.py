import numpy as np
from casadi import dot, Function, vertcat, MX, mtimes, nlpsol
from matplotlib import pyplot as plt
import biorbd
from BiorbdViz import BiorbdViz
from pathlib import Path
from Marche_BiorbdOptim.LoadData import Data_to_track


def plot_control(ax, t, x, color="k", linestyle="--", linewidth=0.7):
    nbPoints = len(np.array(x))
    for n in range(nbPoints - 1):
        ax.plot([t[n], t[n + 1], t[n + 1]], [x[n], x[n], x[n + 1]], color, linestyle, linewidth)


def get_forces(biorbd_model, states, controls, parameters):
    nb_q = biorbd_model.nbQ()
    nb_qdot = biorbd_model.nbQdot()
    nb_tau = biorbd_model.nbGeneralizedTorque()
    nb_mus = biorbd_model.nbMuscleTotal()

    n_muscle = 0
    for nGrp in range(biorbd_model.nbMuscleGroups()):
        for nMus in range(biorbd_model.muscleGroup(nGrp).nbMuscles()):
            fiso_init = biorbd_model.muscleGroup(nGrp).muscle(nMus).characteristics().forceIsoMax().to_mx()
            biorbd_model.muscleGroup(nGrp).muscle(nMus).characteristics().setForceIsoMax(
                parameters[n_muscle] * fiso_init
            )
            n_muscle += 1

    muscles_states = biorbd.VecBiorbdMuscleState(nb_mus)
    muscles_excitation = controls[nb_tau:]
    muscles_activations = states[nb_q + nb_qdot :]

    for k in range(nb_mus):
        muscles_states[k].setExcitation(muscles_excitation[k])
        muscles_states[k].setActivation(muscles_activations[k])

    muscles_tau = biorbd_model.muscularJointTorque(muscles_states, states[:nb_q], states[nb_q : nb_q + nb_qdot]).to_mx()
    tau = muscles_tau + controls[:nb_tau]
    cs = biorbd_model.getConstraints()
    biorbd.Model.ForwardDynamicsConstraintsDirect(biorbd_model, states[:nb_q], states[nb_q : nb_q + nb_qdot], tau, cs)
    return cs.getForce().to_mx()


def get_dispatch_forefoot_contact_forces(grf_ref, M_ref, coord, nb_shooting):
    p = 0.5  # repartition entre les 2 points
    px = np.repeat(0.5, number_shooting_points[2] + 1)
    px[:5] = np.linspace(0, 0.5, 5)

    F_Meta1 = MX.sym("F_Meta1", 3 * (nb_shooting + 1), 1)
    F_Meta5 = MX.sym("F_Meta1", 3 * (nb_shooting + 1), 1)

    objective = 0
    lbg = []
    ubg = []
    constraint = []
    for i in range(nb_shooting + 1):
        # Aliases
        fm1 = F_Meta1[3 * i : 3 * (i + 1)]
        fm5 = F_Meta5[3 * i : 3 * (i + 1)]

        # sum forces = 0 --> Fp1 + Fp2 = Ftrack
        sf = fm1 + fm5
        jf = sf - grf_ref[:, i]
        objective += 100 * mtimes(jf.T, jf)

        # sum moments = 0 --> Mp1_P1 + CP1xFp1 + Mp2_P2 + CP2xFp2 = Mtrack
        sm = dot(coord[0], fm1) + dot(coord[1], fm5)
        jm = sm - M_ref[:, i]
        objective += 100 * mtimes(jm.T, jm)

        # use of p to dispatch forces --> p*Fp1 - (1-p)*Fp2 = 0
        jf2 = p * fm1[2] - (1 - p) * fm5[2]
        jf3 = px[i] * p * fm1[0] - (1 - px[i]) * (1 - p) * fm5[0]
        jf4 = p * fm1[1] - (1 - p) * fm5[1]
        objective += dot(jf2, jf2) + dot(jf3, jf3) + dot(jf4, jf4)

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

    w = [F_Meta1, F_Meta5]
    nlp = {"x": vertcat(*w), "f": objective, "g": vertcat(*constraint)}
    opts = {"ipopt.tol": 1e-8, "ipopt.hessian_approximation": "exact"}
    solver = nlpsol("solver", "ipopt", nlp, opts)
    res = solver(x0=np.zeros(6 * (number_shooting_points[2] + 1)), lbx=-1000, ubx=1000, lbg=lbg, ubg=ubg)

    FM1 = res["x"][: 3 * (nb_shooting + 1)]
    FM5 = res["x"][3 * (nb_shooting + 1) : 6 * (nb_shooting + 1)]

    grf_dispatch_ref = np.zeros((3 * 2, nb_shooting + 1))
    for i in range(3):
        grf_dispatch_ref[i, :] = np.array(FM1[i::3]).squeeze()
        grf_dispatch_ref[i + 3, :] = np.array(FM5[i::3]).squeeze()
    return grf_dispatch_ref


def get_dispatch_flatfoot_contact_forces(grf_ref, M_CoP, coord, CoP, nb_shooting):
    # init
    grf_dispatch_ref = np.zeros((6, nb_shooting + 1))
    Heel_pos = coord[0]
    Meta1_pos = coord[1]
    Meta5_pos = coord[2]

    for i in range(number_shooting_points[1] + 1):
        Heel = CoP[:, i] - Heel_pos
        Meta1 = CoP[:, i] - Meta1_pos
        Meta5 = CoP[:, i] - Meta5_pos
        A = np.array([[1, 0, 0, 0, 1, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 1, 0, 1],
                      [0, -Heel[2], Heel[1], Meta1[1], 0, Meta5[1]],
                      [Heel[2], 0, -Heel[0], -Meta1[0], Meta5[2], -Meta5[0]],
                      [-Heel[1], Heel[0], 0, 0, -Meta5[1], 0]])
        grf_dispatch_ref[:, i] = np.linalg.solve(A, np.concatenate((grf_ref[:, i], M_CoP[:, i])))
    return grf_dispatch_ref


# --- Problem parameter --- #
biorbd_model = (
    biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_1contact_deGroote_3d_Heel.bioMod"),
    biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_3contacts_deGroote_3d.bioMod"),
    biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_contact_deGroote_3d_Forefoot.bioMod"),
)
number_shooting_points = [10, 5, 25]

# Generate data from file
Data_to_track = Data_to_track("normal01", multiple_contact=True)
[T, T_stance, T_swing] = Data_to_track.GetTime()
phase_time = T_stance
grf_ref = Data_to_track.load_data_GRF(biorbd_model[0], T_stance, number_shooting_points)  # get ground reaction forces
M_ref = Data_to_track.load_data_Moment(biorbd_model[0], T_stance, number_shooting_points)
markers_ref = Data_to_track.load_data_markers(biorbd_model[0], T_stance, number_shooting_points, "stance")
q_ref = Data_to_track.load_q_kalman(biorbd_model[0], T_stance, number_shooting_points, "stance")
qdot_ref = Data_to_track.load_qdot_kalman(biorbd_model[0], T_stance, number_shooting_points, "stance")
emg_ref = Data_to_track.load_data_emg(biorbd_model[0], T_stance, number_shooting_points, "stance")
excitation_ref = []
for i in range(len(phase_time)):
    excitation_ref.append(Data_to_track.load_muscularExcitation(emg_ref[i]))
M_CoP = Data_to_track.load_data_Moment_at_CoP(biorbd_model[0], T_stance, number_shooting_points)
CoP = Data_to_track.load_data_CoP(biorbd_model[0], T_stance, number_shooting_points)

# Divide contact forces in contact point
Heel = np.array([np.mean(markers_ref[1][0, 19, :] + 0.04), np.mean(markers_ref[1][1, 19, :]), 0])
Meta1 = np.array([np.mean(markers_ref[1][0, 21, :]), np.mean(markers_ref[1][1, 21, :]), 0])
Meta5 = np.array([np.mean(markers_ref[1][0, 24, :]), np.mean(markers_ref[1][1, 24, :]), 0])
grf_flatfoot_ref = get_dispatch_flatfoot_contact_forces(grf_ref[1], M_CoP[1], [Meta1, Meta5, Heel], CoP[1], number_shooting_points[1])
grf_forefoot_ref = get_dispatch_forefoot_contact_forces(grf_ref[2], M_ref[2], [Meta1, Meta5], number_shooting_points[2])
GRF = (grf_ref[0], grf_flatfoot_ref)

# --- Load the solution --- #
nb_q = biorbd_model[0].nbQ()
nb_qdot = biorbd_model[0].nbQdot()
nb_tau = biorbd_model[0].nbGeneralizedTorque()
nb_mus = biorbd_model[0].nbMuscleTotal()

PROJET = Path(__file__).parent
path_file = str(PROJET) + "/RES/heel_strike_force/"
q = np.load(path_file + "q.npy")
q_dot = np.load(path_file + "q_dot.npy")
activations = np.load(path_file + "activations.npy")
tau = np.load(path_file + "tau.npy")
excitations = np.load(path_file + "excitations.npy")
params = np.load(path_file + "params.npy")
t = np.linspace(0, phase_time[0], number_shooting_points[0] + 1)
t = np.concatenate((t[:-1], t[-1] + np.linspace(0, phase_time[1], number_shooting_points[1] + 1)))
# t = np.concatenate((t[:-1], t[-1] + np.linspace(0, phase_time[2], number_shooting_points[2] + 1)))

# --- Muscle activation and excitation --- #
figure, axes = plt.subplots(4, 5, sharex=True)
axes = axes.flatten()
for i in range(nb_mus):
    name_mus = biorbd_model[0].muscle(i).name().to_string()
    param_value = str(np.round(params[i], 2))
    e = np.concatenate((excitation_ref[0][i, :], excitation_ref[1][i, 1:]))
    plot_control(axes[i], t, e, color="k--")
    plot_control(axes[i], t, excitations[i, :], color="tab:red", linestyle="--", linewidth=0.7)
    axes[i].plot(t, activations[i, :], color="tab:red", linestyle="-", linewidth=1)
    axes[i].plot([phase_time[0], phase_time[0]], [0, 1], color="k", linestyle="--", linewidth=1)
    # axes[i].plot([phase_time[0] + phase_time[1], phase_time[0] + phase_time[1]], [0, 1], color="k", linestyle="--", linewidth=1)
    axes[i].set_title(name_mus)
    axes[i].set_ylim([0, 1])
    axes[i].set_xlim([0, t[-1]])
    axes[i].set_yticks(np.arange(0, 1, step=1 / 5,))
    axes[i].grid(color="k", linestyle="--", linewidth=0.5)
    axes[i].text(0.03, 0.9, param_value)
axes[-1].remove()
axes[-2].remove()
axes[-3].remove()
plt.show()

# --- Generalized positions --- #
q_name = []
for s in range(biorbd_model[0].nbSegment()):
    seg_name = biorbd_model[0].segment(s).name().to_string()
    for d in range(biorbd_model[0].segment(s).nbDof()):
        dof_name = biorbd_model[0].segment(s).nameDof(d).to_string()
        q_name.append(seg_name + "_" + dof_name)

figure, axes = plt.subplots(4, 3, sharex=True)
axes = axes.flatten()
for i in range(nb_q):
    Q = np.concatenate((q_ref[0][i, :], q_ref[1][i, 1:]))
    # Q = np.concatenate((q_ref[0][i, :], q_ref[1][i, 1:], q_ref[2][i, 1:]))
    axes[i].plot(t, q[i, :], color="tab:red", linestyle="-", linewidth=1)
    axes[i].plot(t, Q, color="k", linestyle="--", linewidth=0.7)
    axes[i].set_title(q_name[i])
    axes[i].plot(
        [phase_time[0], phase_time[0]], [np.max(q[i, :]), np.min(q[i, :])], color="k", linestyle="--", linewidth=1
    )
    # axes[i].plot([phase_time[0] + phase_time[1], phase_time[0] + phase_time[1]], [np.max(q[i, :]), np.min(q[i, :])], color="k", linestyle="--", linewidth=1)
    axes[i].grid(color="k", linestyle="--", linewidth=0.5)
plt.show()

# --- Compute contact forces --- #
contact_forces = []
idx = 0
for n_p in range(2):
    cf = np.zeros((biorbd_model[n_p].nbContacts(), number_shooting_points[n_p] + 1))
    symbolic_states = MX.sym("x", nb_q + nb_qdot + nb_mus, 1)
    symbolic_controls = MX.sym("u", nb_tau + nb_mus, 1)
    symbolic_params = MX.sym("p", nb_mus, 1)
    computeGRF = Function(
        "ComputeGRF",
        [symbolic_states, symbolic_controls, symbolic_params],
        [get_forces(biorbd_model[n_p], symbolic_states, symbolic_controls, symbolic_params)],
        ["x", "u", "p"],
        ["GRF"],
    ).expand()
    for i in range(number_shooting_points[n_p] + 1):
        state = np.concatenate((q[:, idx + i], q_dot[:, idx + i], activations[:, idx + i]))
        control = np.concatenate((tau[:, idx + i], excitations[:, idx + i]))
        cf[:, i] = np.array(computeGRF(state, control, params)).squeeze()
    contact_forces.append(cf)
    idx = number_shooting_points[n_p]

figure, axes = plt.subplots(1, 3)
axes = axes.flatten()
t1 = np.linspace(0, phase_time[0], number_shooting_points[0] + 1)
t2 = t1[-1] + np.linspace(0, phase_time[1], number_shooting_points[1] + 1)
t3 = t2[-1] + np.linspace(0, phase_time[2], number_shooting_points[2] + 1)
t = np.concatenate((t1[:-1], t2))
# t = np.concatenate((t1[:-1], t2[:-1], t3))

axes[0].set_title("contact forces in x")
Gx = np.concatenate((grf_ref[0][0, :], grf_ref[1][0, 1:]))
Fx_heel = np.concatenate((contact_forces[0][0, :], contact_forces[1][0, 1:]))
Fx_meta5 = np.concatenate((np.zeros(number_shooting_points[0] + 1), contact_forces[1][4, 1:]))
axes[0].plot(t, Gx, color="k", linestyle="--", linewidth=1.2)
axes[0].plot(t, Fx_heel, color="tab:green", linestyle="-", linewidth=1)
axes[0].plot(t, Fx_meta5, color="tab:blue", linestyle="-", linewidth=1)
axes[0].plot(t, Fx_heel + Fx_meta5, color="tab:purple", linestyle="-", linewidth=1)
axes[0].plot([phase_time[0], phase_time[0]], [np.min(Gx), np.max(Gx)], color="k", linestyle="--", linewidth=1)
axes[0].legend(("plateforme", "Heel", "Meta 5", "sum"))
axes[0].grid(color="k", linestyle="--", linewidth=0.5)

axes[1].set_title("contact forces in y")
Gy = np.concatenate((grf_ref[0][1, :], grf_ref[1][1, 1:]))
Fy_heel = np.concatenate((contact_forces[0][1, :], contact_forces[1][1, 1:]))
axes[1].plot(t, Gy, color="k", linestyle="--", linewidth=1.2)
axes[1].plot(t, Fy_heel, color="tab:green", linestyle="-", linewidth=1)
axes[1].plot([phase_time[0], phase_time[0]], [np.min(Gy), np.max(Gy)], color="k", linestyle="--", linewidth=1)
axes[1].legend(("plateforme", "Heel"))
axes[1].grid(color="k", linestyle="--", linewidth=0.5)

axes[2].set_title("contact forces in z")
Gz = np.concatenate((grf_ref[0][2, :], grf_ref[1][2, 1:]))
Fz_heel = np.concatenate((contact_forces[0][2, :-1], contact_forces[1][2, :]))
Fz_meta1 = np.concatenate((np.zeros(number_shooting_points[0]), contact_forces[1][3, :]))
Fz_meta5 = np.concatenate((np.zeros(number_shooting_points[0]), contact_forces[1][5, :]))
axes[2].plot(t, Gz, color="k", linestyle="--", linewidth=1.2)
axes[2].plot(t, Fz_heel, color="tab:green", linestyle="-", linewidth=1)
axes[2].plot(t, Fz_meta1, color="tab:red", linestyle="-", linewidth=1)
axes[2].plot(t, Fz_meta5, color="tab:blue", linestyle="-", linewidth=1)
axes[2].plot(t, Fz_heel + Fz_meta1 + Fz_meta5, color="tab:purple", linestyle="-", linewidth=1)
axes[2].plot([phase_time[0], phase_time[0]], [np.min(Gz), np.max(Gz)], color="k", linestyle="--", linewidth=1)
axes[2].legend(("plateforme", "Heel", "Meta 1", "Meta 5"))
axes[2].grid(color="k", linestyle="--", linewidth=0.5)
plt.show()

