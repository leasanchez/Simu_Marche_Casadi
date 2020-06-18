import numpy as np
from casadi import dot, Function, vertcat, MX
from matplotlib import pyplot as plt
import biorbd
from pathlib import Path
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
            biorbd_model.muscleGroup(nGrp).muscle(nMus).characteristics().setForceIsoMax(value[n_muscle] * fiso_init)
            n_muscle += 1


# --- Load the optimal control program and the solution --- #
PROJET = Path(__file__).parent
# file = "stance/marche_stance_excitation_2.bo"
file_2 = "swing/RES/parametres/marche_swing_excitation_sans_params.bo"
gaitphase = "swing"
# ocp, sol = OptimalControlProgram.load(str(PROJET) + "/" + file)
ocp_init, sol_init = OptimalControlProgram.load(str(PROJET) + "/" + file_2)

biorbd_model = ocp_init.nlp[0]["model"]
n_shooting_points = ocp_init.nlp[0]["ns"]
final_time = ocp_init.nlp[0]["tf"]
nb_q = ocp_init.nlp[0]["nbQ"]
nb_qdot = ocp_init.nlp[0]["nbQdot"]
nb_marker = biorbd_model.nbMarkers()
nb_mus = ocp_init.nlp[0]["nbMuscles"]
nb_tau = ocp_init.nlp[0]["nbTau"]

# --- Get Results --- #
# states_sol, controls_sol, params_sol = Data.get_data(ocp, sol, get_parameters=True)
states_init, controls_init = Data.get_data(ocp_init, sol_init)
# q = states_sol["q"]
# q_dot = states_sol["q_dot"]
activations = np.load(str(PROJET) + "/swing/RES/parametres/activations_params.npy")
activations_init = states_init["muscles"]
# tau = controls_sol["tau"]
excitations = np.load(str(PROJET) + "/swing/RES/parametres/excitations_params.npy")
excitations_init = controls_init["muscles"]
# params_sol = params_sol[ocp.nlp[0]['p'].name()]
params_sol = np.load(str(PROJET) + "/swing/RES/parametres/params.npy")
print(params_sol)

# Generate data from file
Data_to_track = Data_to_track(name_subject="equincocont01")

grf_ref = Data_to_track.load_data_GRF(biorbd_model, final_time, n_shooting_points)  # get ground reaction forces
markers_ref = Data_to_track.load_data_markers(
    biorbd_model, final_time, n_shooting_points, gaitphase
)  # get markers position
q_ref = Data_to_track.load_data_q(biorbd_model, final_time, n_shooting_points, gaitphase)  # get q from kalman
emg_ref = Data_to_track.load_data_emg(biorbd_model, final_time, n_shooting_points, gaitphase)  # get emg
excitation_ref = Data_to_track.load_muscularExcitation(emg_ref)

# --- Comparison muscle activation and excitation with/without parameters --- #
def plot_control(ax, t, x, color="b"):
    nbPoints = len(np.array(x))
    for n in range(nbPoints - 1):
        ax.plot([t[n], t[n + 1], t[n + 1]], [x[n], x[n], x[n + 1]], color)


figure, axes = plt.subplots(4, 5, sharex=True)
axes = axes.flatten()
t = np.linspace(0, final_time, n_shooting_points + 1)
for i in range(biorbd_model.nbMuscleTotal()):
    name_mus = biorbd_model.muscle(i).name().to_string()
    param_value = str(np.round(params_sol[i], 2))
    plot_control(axes[i], t, excitation_ref[i, :], color="k--")

    plot_control(axes[i], t, excitations_init[i, :], color="r--")
    axes[i].plot(t, activations_init[i, :], "r.-")  # without parameters

    plot_control(axes[i], t, excitations[i, :], color="b--")
    axes[i].plot(t, activations[i, :], "b.-")  # with parameters
    axes[i].text(0.03, 0.9, param_value)

    axes[i].set_title(name_mus)
    axes[i].set_ylim([0, 1])
    axes[i].set_yticks(np.arange(0, 1, step=1 / 5,))
    axes[i].grid(color="k", linestyle="--", linewidth=0.5)
axes[-1].remove()
axes[-2].remove()
axes[-3].remove()

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
        np.linspace(0, nb_marker, nb_marker), hist_diff_track[i, :], width=1.0, facecolor="b", edgecolor="k", alpha=0.5,
    )
    axes[i].set_xticks(np.arange(nb_marker))
    axes[i].set_xticklabels(label_markers, rotation=90)
    axes[i].set_ylabel("Mean differences in " + title_markers[i] + " (mm)")
    axes[i].plot([0, nb_marker], [mean_diff_track[i], mean_diff_track[i]], "--r")
    axes[i].set_title("markers differences between sol and exp")

    axes[i + 3].bar(
        np.linspace(0, nb_marker, nb_marker), hist_diff_sol[i, :], width=1.0, facecolor="b", edgecolor="k", alpha=0.5,
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

# --- Show results --- #
ocp.add_plot("q", lambda x, u: q_ref, PlotType.STEP, axes_idx=[0, 1, 5, 8, 9, 11])
result = ShowResult(ocp, sol)
result.graphs()
