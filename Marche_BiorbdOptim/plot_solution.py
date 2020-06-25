import numpy as np
from casadi import dot, Function, vertcat, MX
from matplotlib import pyplot as plt
import biorbd
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

# --- Define problem --- #
PROJECT_FOLDER = Path(__file__).parent / ".."
biorbd_model = (
    biorbd.Model(str(PROJECT_FOLDER) + "/ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact_deGroote_3d.bioMod"),
    biorbd.Model(
        str(PROJECT_FOLDER) + "/ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_0contact_deGroote_3d.bioMod"
    ),
)

# Problem parameters
number_shooting_points = [25, 25]

nb_phases = len(biorbd_model)
nb_q = biorbd_model[0].nbQ()
nb_qdot = biorbd_model[0].nbQdot()
nb_mus = biorbd_model[0].nbMuscleTotal()
nb_tau = biorbd_model[0].nbGeneralizedTorque()
nb_markers = biorbd_model[0].nbMarkers()

# Generate data from file
markers_ref = []
q_ref = []
emg_ref = []
Data_to_track = Data_to_track("equincocont01", multiple_contact=False)
[T, T_stance, T_swing] = Data_to_track.GetTime()
phase_time = [T_stance, T_swing]
grf_ref = Data_to_track.load_data_GRF(biorbd_model[0], T_stance, number_shooting_points[0])
markers_ref.append(Data_to_track.load_data_markers(biorbd_model[0], T_stance, number_shooting_points[0], "stance"))
markers_ref.append(Data_to_track.load_data_markers(biorbd_model[1], phase_time[1], number_shooting_points[1], "swing"))
q_ref.append(Data_to_track.load_q_kalman(biorbd_model[0], T_stance, number_shooting_points[0], "stance"))
q_ref.append(Data_to_track.load_q_kalman(biorbd_model[0], phase_time[1], number_shooting_points[1], "swing"))
emg_ref.append(Data_to_track.load_data_emg(biorbd_model[0], T_stance, number_shooting_points[0], "stance"))
emg_ref.append(Data_to_track.load_data_emg(biorbd_model[-1], phase_time[-1], number_shooting_points[-1], "swing"))
excitation_ref = []
for i in range(len(phase_time)):
    excitation_ref.append(Data_to_track.load_muscularExcitation(emg_ref[i]))

# --- Load the optimal control program and the solution --- #
file = "./multiphase/RES/equincocont03/"
params = np.load(file + "params.npy")
q = np.load(file + "q.npy")
q_dot = np.load(file + "q_dot.npy")
activations = np.load(file + "activations.npy")
excitations = np.load(file + "excitations.npy")
tau = np.load(file + "tau.npy")

# --- Muscle activation and excitation --- #
params_MUSCOD = np.array([0.2, 0.2098, 0.5238, 0.22309, 0.2, 0.2, 1.6849, 0.2815, 0.2, 2.8412, 0.1997, 0.2067, 0.3862, 4.9731, 5.0, 1.1807, 5.0])
figure, axes = plt.subplots(4, 5, sharex=True)
axes = axes.flatten()
t = np.linspace(0, phase_time[0], number_shooting_points[0] + 1)
t = np.concatenate((t[:-1], t[-1] + np.linspace(0, phase_time[1], number_shooting_points[1] + 1)))
for i in range(nb_mus):
    name_mus = biorbd_model[0].muscle(i).name().to_string()
    e = np.concatenate((excitation_ref[0][i, :], excitation_ref[1][i, 1:]))
    plot_control(axes[i], t, e, color="k--")
    plot_control(axes[i], t, excitations[i, :], color="tab:red", linestyle="--", linewidth=0.7)
    axes[i].plot(t, activations[i, :], color="tab:red", linestyle="-", linewidth=1)
    axes[i].plot([phase_time[0], phase_time[0]], [0, 1], color="k", linestyle="--", linewidth=1)
    axes[i].set_title(name_mus)
    axes[i].set_ylim([0, 1])
    axes[i].set_xlim([0, t[-1]])
    axes[i].set_yticks(np.arange(0, 1, step=1 / 5,))
    axes[i].grid(color="k", linestyle="--", linewidth=0.5)
    axes[i].text(0.03, 0.9, f"parameter : {np.round(params[i][0], 2)} / muscod : {np.round(params_MUSCOD[i], 2)}")
axes[-1].remove()
axes[-2].remove()
axes[-3].remove()
plt.show()

# --- Generalized positions --- #
# init
Q_ref = np.zeros((nb_q, np.sum(number_shooting_points) + 1))
Q_ref[:, :number_shooting_points[0] + 1] = q_ref[0]
Q_ref[:, number_shooting_points[0] :] = q_ref[1]

# compute RMSE
diff_q = np.sqrt((q - Q_ref) * (q - Q_ref))
mean_diff_q = np.zeros(nb_q)
mean_diff_q_deg = np.zeros(nb_q)
for i in range(nb_q):
    mean_diff_q[i] = np.mean(diff_q[i, :])
    if (i>2):
        mean_diff_q_deg[i] = np.mean(diff_q[i, :])*180/np.pi
    else:
        mean_diff_q_deg[i] = np.mean(diff_q[i, :])

# compute R2
mean_q = np.zeros((nb_q, np.sum(number_shooting_points) + 1))
for i in range(nb_q):
    mean_q[i, :] = np.repeat(np.mean(q[i, :]), np.sum(number_shooting_points) + 1)
diff_q_square = (q - Q_ref) * (q - Q_ref)
diff_q_mean = (q - mean_q) * (q - mean_q)

R2_q = np.zeros(nb_q)
for i in range(nb_q):
    s = (np.sum(diff_q_square[i, :])/np.sum(diff_q_mean[i, :]))
    if (s>1):
        R2_q[i] = s - 1
    else:
        R2_q[i] = 1 - s

# compute pic error
max_diff = np.zeros(nb_q)
min_diff = np.zeros(nb_q)
for i in range(nb_q):
    max_diff[i] = np.sqrt((np.max(q[i, :]) - np.max(Q_ref[i, :])) * (np.max(q[i, :]) - np.max(Q_ref[i, :])))
    min_diff[i] = np.sqrt((np.min(q[i, :]) - np.min(Q_ref[i, :])) * (np.min(q[i, :]) - np.min(Q_ref[i, :])))

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
    if (i>2):
        axes[i].plot(t, q[i, :]*180/np.pi, color="tab:red", linestyle="-", linewidth=1)
        axes[i].plot(t, Q*180/np.pi, color="k", linestyle="--", linewidth=1)
        axes[i].plot(
            [phase_time[0], phase_time[0]], [np.min(q[i, :]*180/np.pi), np.max(q[i, :]*180/np.pi)], color="k", linestyle="--", linewidth=1
        )
        axes[i].text(0, np.min(Q*180/np.pi), f"R2 : {np.round(R2_q[i], 3)}")
        axes[i].set_ylabel('angle (degre)')
    else:
        axes[i].plot(t, q[i, :], color="tab:red", linestyle="-", linewidth=1)
        axes[i].plot(t, Q, color="k", linestyle="--", linewidth=1)
        axes[i].plot(
            [phase_time[0], phase_time[0]], [np.max(q[i, :]), np.min(q[i, :])], color="k", linestyle="--", linewidth=1
        )
        axes[i].text(0, np.min(Q), f"R2 : {np.round(R2_q[i], 3)}")
        axes[i].set_ylabel('position (m)')

    axes[i].set_title(q_name[i])
    axes[i].grid(color="k", linestyle="--", linewidth=0.5)

plt.show()

# --- Get markers position from q_sol and q_ref --- #
# init
markers_sol = np.ndarray((3, nb_markers, np.sum(number_shooting_points) + 1))
markers_from_q_ref = np.ndarray((3, nb_markers, np.sum(number_shooting_points) + 1))
M_ref = np.zeros((3, nb_markers, np.sum(number_shooting_points) + 1))
M_ref[:, :, :number_shooting_points[0] + 1] = markers_ref[0]
M_ref[:, :, number_shooting_points[0] :] = markers_ref[1]

markers_func_3d = []
symbolic_states = MX.sym("x", nb_q + nb_qdot + nb_mus, 1)
symbolic_controls = MX.sym("u", nb_tau + nb_mus, 1)
for i in range(nb_markers):
    markers_func_3d.append(
        Function(
            "ForwardKin",
            [symbolic_states],
            [biorbd_model[0].marker(symbolic_states[:nb_q], i).to_mx()],
            ["q"],
            ["marker_" + str(i)],
        ).expand()
    )


for i in range(np.sum(number_shooting_points) + 1):
    for j, mark_func in enumerate(markers_func_3d):
        markers_sol[:, j, i] = np.array(mark_func(vertcat(q[:, i], q_dot[:, i], activations[:, i]))).squeeze()
        markers_from_q_ref[:, j, i] = np.array(mark_func(vertcat(Q_ref[:, i], q_dot[:, i], activations[:, i]))).squeeze()

diff_track = np.sqrt((markers_sol - M_ref) * (markers_sol - M_ref)) * 1e3
diff_sol = np.sqrt((markers_sol - markers_from_q_ref) * (markers_sol - markers_from_q_ref)) * 1e3

hist_diff_track = np.zeros((3, nb_markers))
hist_diff_sol = np.zeros((3, nb_markers))

for n_mark in range(nb_markers):
    hist_diff_track[0, n_mark] = sum(diff_track[0, n_mark, :]) / (np.sum(number_shooting_points) + 1)
    hist_diff_track[1, n_mark] = sum(diff_track[1, n_mark, :]) / (np.sum(number_shooting_points) + 1)
    hist_diff_track[2, n_mark] = sum(diff_track[2, n_mark, :]) / (np.sum(number_shooting_points) + 1)

    hist_diff_sol[0, n_mark] = sum(diff_sol[0, n_mark, :]) / (np.sum(number_shooting_points) + 1)
    hist_diff_sol[1, n_mark] = sum(diff_sol[1, n_mark, :]) / (np.sum(number_shooting_points) + 1)
    hist_diff_sol[2, n_mark] = sum(diff_sol[2, n_mark, :]) / (np.sum(number_shooting_points) + 1)

mean_diff_track = [
    sum(hist_diff_track[0, :]) / nb_markers,
    sum(hist_diff_track[1, :]) / nb_markers,
    sum(hist_diff_track[2, :]) / nb_markers,
]
mean_diff_sol = [
    sum(hist_diff_sol[0, :]) / nb_markers,
    sum(hist_diff_sol[1, :]) / nb_markers,
    sum(hist_diff_sol[2, :]) / nb_markers,
]

mean_diff_markers_track = np.zeros(nb_markers)
mean_diff_markers_sol = np.zeros(nb_markers)
for i in range(nb_markers):
    mean_diff_markers_track[i] = np.sum(hist_diff_track[:, i])/3
    mean_diff_markers_sol[i] = np.sum(hist_diff_sol[:, i])/3

# --- Plot markers --- #
label_markers = []
markers_name = biorbd_model[0].markerNames()
for n_mark in range(nb_markers):
    label_markers.append(markers_name[n_mark].to_string())

figure, axes = plt.subplots(2, 3)
axes = axes.flatten()
title_markers = ["x axis", "y axis", "z axis"]
for i in range(3):
    if (i==1):
        axes[i].set_title("markers differences between sol and exp")
        axes[i + 3].set_title("markers differences between sol and ref")

    axes[i].bar(
        np.linspace(0, nb_markers, nb_markers), hist_diff_track[i, :], width=1.0, facecolor="b", edgecolor="k", alpha=0.5,
    )
    axes[i].set_xticks(np.arange(nb_markers))
    axes[i].set_xticklabels(label_markers, rotation=90)
    axes[i].set_ylabel("Mean differences in " + title_markers[i] + " (mm)")
    axes[i].plot([0, nb_markers], [mean_diff_track[i], mean_diff_track[i]], "--r")
    axes[i].set_ylim([0, 100])

    axes[i + 3].bar(
        np.linspace(0, nb_markers, nb_markers), hist_diff_sol[i, :], width=1.0, facecolor="b", edgecolor="k", alpha=0.5,
    )
    axes[i + 3].set_xticks(np.arange(nb_markers))
    axes[i + 3].set_xticklabels(label_markers, rotation=90)
    axes[i + 3].set_ylabel("Mean differences in " + title_markers[i] + " (mm)")
    axes[i + 3].plot([0, nb_markers], [mean_diff_sol[i], mean_diff_sol[i]], "--r")
    axes[i + 3].set_ylim([0, 100])


figure, axes = plt.subplots(2, 3)
axes = axes.flatten()
for i in range(3):
    if (i==1):
        axes[i].set_title("markers differences between sol and exp")
        axes[i + 3].set_title("markers differences between sol and ref")

    axes[i].plot(t, diff_track[i, :, :].T)
    axes[i].set_xlabel("time (s)")
    axes[i].set_ylabel("Differences in " + title_markers[i] + " (mm)")
    axes[i].plot(
        [phase_time[0], phase_time[0]], [0, 120], color="k", linestyle="--", linewidth=1
    )
    axes[i].grid(color="k", linestyle="--", linewidth=0.5)
    axes[i].set_ylim([0, 120])
    axes[i].set_xlim([0, t[-1]])

    axes[i + 3].plot(t, diff_sol[i, :, :].T)
    axes[i + 3].set_xlabel("time (s)")
    axes[i + 3].set_ylabel("Differences in " + title_markers[i] + " (mm)")
    axes[i + 3].plot(
        [phase_time[0], phase_time[0]], [0, 120], color="k", linestyle="--", linewidth=1
    )
    axes[i + 3].grid(color="k", linestyle="--", linewidth=0.5)
    axes[i + 3].set_ylim([0, 120])
    axes[i + 3].set_xlim([0, t[-1]])
plt.legend(label_markers, bbox_to_anchor=(1,2), loc='best')

figure, axes = plt.subplots(1, 2, sharey=True)
axes = axes.flatten()
axes[0].bar(
    np.linspace(0, nb_markers, nb_markers), mean_diff_markers_track, width=1.0, facecolor="b", edgecolor="k", alpha=0.5,
)
axes[0].set_xticks(np.arange(nb_markers))
axes[0].set_xticklabels(label_markers, rotation=90)
axes[0].set_title('Mean error between simulated marker position and position tracked')
axes[0].plot([0, nb_markers], [np.mean(mean_diff_markers_track), np.mean(mean_diff_markers_track)], "--r")
axes[0].set_ylim([0, 60])
axes[0].set_ylabel('Mean difference (mm)')

axes[1].bar(
    np.linspace(0, nb_markers, nb_markers), mean_diff_markers_sol, width=1.0, facecolor="b", edgecolor="k", alpha=0.5,
)
axes[1].set_xticks(np.arange(nb_markers))
axes[1].set_xticklabels(label_markers, rotation=90, )
axes[1].set_title('Mean error between simulated marker position and position from q_ref')
axes[1].plot([0, nb_markers], [np.mean(mean_diff_markers_sol), np.mean(mean_diff_markers_sol)], "--r")
axes[1].set_ylim([0, 60])
plt.show()

# --- Compute and plot ground reaction forces --- #
contact_forces = np.zeros((3, np.sum(number_shooting_points) + 1))

symbolic_states = MX.sym("x", nb_q + nb_qdot + nb_mus, 1)
symbolic_controls = MX.sym("u", nb_tau + nb_mus, 1)
symbolic_params = MX.sym("p", nb_mus, 1)
computeGRF = Function(
    "ComputeGRF",
    [symbolic_states, symbolic_controls, symbolic_params],
    [get_forces(biorbd_model[0], symbolic_states, symbolic_controls, symbolic_params)],
    ["x", "u", "p"],
    ["GRF"],
).expand()

for i in range(number_shooting_points[0] + 1):
    state = np.concatenate((q[:, i], q_dot[:, i], activations[:, i]))
    control = np.concatenate((tau[:, i], excitations[:, i]))
    contact_forces[:, i] = np.array(computeGRF(state, control, params)).squeeze()

title_axis = ["x", "y", "z"]
figure, axes = plt.subplots(1, 3)
for i in range(3):
    axes[i].set_title("contact forces in " + title_axis[i])
    axes[i].plot(t, contact_forces[i, :], color="tab:red", linestyle="-", linewidth=1)
    axes[i].plot(t[:number_shooting_points[0] + 1], grf_ref[i, :], color="k", linestyle="--", linewidth=1)
    axes[i].grid(color="k", linestyle="--", linewidth=0.5)
    axes[i].set_xlim([0, t[-1]])
    axes[i].plot(
        [phase_time[0], phase_time[0]], [np.min(contact_forces[i, :]), np.max(contact_forces[i, :])], color="k", linestyle="--", linewidth=1
    )
axes[0].set_ylabel('Contact forces (N)')
axes[1].set_xlabel('time (s)')
plt.show()
