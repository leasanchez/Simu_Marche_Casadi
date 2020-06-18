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

def modify_isometric_force(biorbd_model, value):
    n_muscle = 0
    for nGrp in range(biorbd_model.nbMuscleGroups()):
        for nMus in range(biorbd_model.muscleGroup(nGrp).nbMuscles()):
            biorbd_model.muscleGroup(nGrp).muscle(nMus).characteristics().setForceIsoMax(value[n_muscle] * fiso_init)
            n_muscle += 1

# --- Define problem --- #
PROJECT_FOLDER = Path(__file__).parent / ".."
biorbd_model = (
    biorbd.Model(str(PROJECT_FOLDER) + "/ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact_deGroote_3d.bioMod"),
    biorbd.Model(str(PROJECT_FOLDER) + "/ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_0contact_deGroote_3d.bioMod"),
)

# Problem parameters
number_shooting_points = [25, 25]

nb_phases = len(biorbd_model)
nb_q = biorbd_model[0].nbQ()
nb_qdot = biorbd_model[0].nbQdot()
nb_mus = biorbd_model[0].nbMuscleTotal()
nb_tau = biorbd_model[0].nbGeneralizedTorque()

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
file = "./multiphase/RES/equincocont01/"
params = np.load(file + 'params.npy')
q = np.load(file + 'q.npy')
q_dot = np.load(file + 'q_dot.npy')
activations = np.load(file + 'activations.npy')
excitations = np.load(file + 'excitations.npy')
tau = np.load(file + 'tau.npy')

# --- Muscle activation and excitation --- #
figure, axes = plt.subplots(4, 5, sharex=True)
axes = axes.flatten()
t = np.linspace(0, phase_time[0], number_shooting_points[0] + 1)
t = np.concatenate((t[:-1], t[-1] + np.linspace(0, phase_time[1], number_shooting_points[1] + 1)))
for i in range(nb_mus):
    name_mus = biorbd_model[0].muscle(i).name().to_string()
    param_value = str(np.round(params[i], 2))
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
        q_name.append(seg_name + '_' + dof_name)

figure, axes = plt.subplots(4, 3, sharex=True)
axes = axes.flatten()
for i in range(nb_q):
    Q = np.concatenate((q_ref[0][i, :], q_ref[1][i, 1:]))
    axes[i].plot(t, q[i, :], color="tab:red", linestyle='-', linewidth=1)
    axes[i].plot(t, Q, color="k", linestyle='--', linewidth=0.7)
    axes[i].set_title(q_name[i])
    axes[i].grid(color="k", linestyle="--", linewidth=0.5)
    axes[i].plot([phase_time[0], phase_time[0]], [np.max(q[i, :]), np.min(q[i, :])], color="k", linestyle="--", linewidth=1)
plt.show()

# --- Get markers position from q_sol and q_ref --- #
markers_sol = np.ndarray((3, nb_marker, ocp.nlp[0]["ns"] + 1))
markers_from_q_ref = np.ndarray((3, nb_marker, ocp.nlp[0]["ns"] + 1))

markers_func_3d = []
symbolic_states = MX.sym("x", ocp.nlp[0]["nx"], 1)
symbolic_controls = MX.sym("u", ocp.nlp[0]["nu"], 1)
for i in range(nb_marker):
    markers_func_3d.append(
        Function(
            "ForwardKin",
            [symbolic_states],
            [biorbd_model.marker(symbolic_states[:nb_q], i).to_mx()],
            ["q"],
            ["marker_" + str(i)],
        ).expand()
    )

model_q = biorbd.Model("../../ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact_deGroote.bioMod")
markers_func_2d = []
symbolic_states = MX.sym("x", model_q.nbQ() + model_q.nbQdot() + model_q.nbMuscleTotal(), 1)
symbolic_controls = MX.sym("u", model_q.nbGeneralizedTorque() + model_q.nbMuscleTotal(), 1)
for i in range(nb_marker):
    markers_func_2d.append(
        Function(
            "ForwardKin",
            [symbolic_states],
            [model_q.marker(symbolic_states[: model_q.nbQ()], i).to_mx()],
            ["q"],
            ["marker_" + str(i)],
        ).expand()
    )

for i in range(ocp.nlp[0]["ns"] + 1):
    for j, mark_func in enumerate(markers_func_3d):
        markers_sol[:, j, i] = np.array(mark_func(vertcat(q[:, i], q_dot[:, i], activations[:, i]))).squeeze()
        Q_ref = np.concatenate([q_ref[:, i], np.zeros(model_q.nbQ()), np.zeros(model_q.nbMuscleTotal())])
        markers_from_q_ref[:, j, i] = np.array(markers_func_2d[j](Q_ref)).squeeze()

diff_track = np.sqrt((markers_sol - markers_ref) * (markers_sol - markers_ref)) * 1e3
diff_sol = np.sqrt((markers_sol - markers_from_q_ref) * (markers_sol - markers_from_q_ref)) * 1e3
hist_diff_track = np.zeros((3, nb_marker))
hist_diff_sol = np.zeros((3, nb_marker))

for n_mark in range(nb_marker):
    hist_diff_track[0, n_mark] = sum(diff_track[0, n_mark, :]) / n_shooting_points
    hist_diff_track[1, n_mark] = sum(diff_track[1, n_mark, :]) / n_shooting_points
    hist_diff_track[2, n_mark] = sum(diff_track[2, n_mark, :]) / n_shooting_points

    hist_diff_sol[0, n_mark] = sum(diff_sol[0, n_mark, :]) / n_shooting_points
    hist_diff_sol[1, n_mark] = sum(diff_sol[1, n_mark, :]) / n_shooting_points
    hist_diff_sol[2, n_mark] = sum(diff_sol[2, n_mark, :]) / n_shooting_points

mean_diff_track = [
    sum(hist_diff_track[0, :]) / nb_marker,
    sum(hist_diff_track[1, :]) / nb_marker,
    sum(hist_diff_track[2, :]) / nb_marker,
]
mean_diff_sol = [
    sum(hist_diff_sol[0, :]) / nb_marker,
    sum(hist_diff_sol[1, :]) / nb_marker,
    sum(hist_diff_sol[2, :]) / nb_marker,
]

# --- Plot markers --- #
label_markers = []
for mark in range(nb_marker):
    label_markers.append(ocp.nlp[0]["model"].markerNames()[mark].to_string())

figure, axes = plt.subplots(2, 3)
axes = axes.flatten()
title_markers = ["x axis", "y axis", "z axis"]
for i in range(3):
    axes[i].bar(
        np.linspace(0, nb_marker, nb_marker),
        hist_diff_track[i, :],
        width=1.0,
        facecolor="b",
        edgecolor="k",
        alpha=0.5,
    )
    axes[i].set_xticks(np.arange(nb_marker))
    axes[i].set_xticklabels(label_markers, rotation=90)
    axes[i].set_ylabel("Mean differences in " + title_markers[i] + " (mm)")
    axes[i].plot([0, nb_marker], [mean_diff_track[i], mean_diff_track[i]], "--r")
    axes[i].set_title("markers differences between sol and exp")

    axes[i + 3].bar(
        np.linspace(0, nb_marker, nb_marker),
        hist_diff_sol[i, :],
        width=1.0,
        facecolor="b",
        edgecolor="k",
        alpha=0.5,
    )
    axes[i + 3].set_xticks(np.arange(nb_marker))
    axes[i + 3].set_xticklabels(label_markers, rotation=90)
    axes[i + 3].set_ylabel("Mean differences in " + title_markers[i] + " (mm)")
    axes[i + 3].plot([0, nb_marker], [mean_diff_sol[i], mean_diff_sol[i]], "--r")
    axes[i + 3].set_title("markers differences between sol and ref")

figure, axes = plt.subplots(2, 3)
axes = axes.flatten()
t = np.linspace(0, final_time, n_shooting_points + 1)
for i in range(3):
    axes[i].plot(t, diff_track[i, :, :].T)
    axes[i].set_xlabel("time (s)")
    axes[i].set_ylabel("Mean differences in " + title_markers[i] + " (mm)")
    axes[i].set_title("markers differences between sol and exp")

    axes[i + 3].plot(t, diff_sol[i, :, :].T)
    axes[i + 3].set_xlabel("time (s)")
    axes[i + 3].set_ylabel("Meaen differences in " + title_markers[i] + " (mm)")
    axes[i + 3].set_title("markers differences between sol and ref")

