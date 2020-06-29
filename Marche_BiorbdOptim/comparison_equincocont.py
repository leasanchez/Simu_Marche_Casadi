import numpy as np
from casadi import dot, Function, vertcat, MX
from matplotlib import pyplot as plt
import biorbd
from pathlib import Path
import os
from Marche_BiorbdOptim.LoadData import Data_to_track

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

q_name = []
for s in range(biorbd_model[0].nbSegment()):
    seg_name = biorbd_model[0].segment(s).name().to_string()
    for d in range(biorbd_model[0].segment(s).nbDof()):
        dof_name = biorbd_model[0].segment(s).nameDof(d).to_string()
        q_name.append(seg_name + "_" + dof_name)

label_markers = []
markers_name = biorbd_model[0].markerNames()
for n_mark in range(nb_markers):
    label_markers.append(markers_name[n_mark].to_string())

# casadi fcn
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

# init
R2_mean = np.zeros(nb_q)
RMSE_mean = np.zeros(nb_q)
pic_max_mean = np.zeros(nb_q)
pic_min_mean = np.zeros(nb_q)
Diff_markers = np.zeros(nb_markers)
Diff_marker = 0

# get results
list_subjects = os.listdir(str(Path(__file__).parent) + "/multiphase/RES")
results = []

for subject in list_subjects:
    if subject != "equincocont09":
        dict = {}
        dict["name_subject"] = subject

        # --- get solution ----
        file = "./multiphase/RES/" + subject + "/"
        dict["file"] = file
        dict["params"] = np.load(file + "params.npy")
        dict["q"] = np.load(file + "q.npy")
        dict["q_dot"] = np.load(file + "q_dot.npy")
        dict["activations"] = np.load(file + "activations.npy")
        dict["excitations"] = np.load(file + "excitations.npy")
        dict["tau"] = np.load(file + "tau.npy")

        # --- get data to track ----
        markers_ref = []
        q_ref = []
        emg_ref = []
        Data = Data_to_track(subject, multiple_contact=False)
        [T, T_stance, T_swing] = Data.GetTime()
        phase_time = [T_stance, T_swing]
        t = np.linspace(0, phase_time[0], number_shooting_points[0] + 1)
        t = np.concatenate((t[:-1], t[-1] + np.linspace(0, phase_time[1], number_shooting_points[1] + 1)))
        grf_ref = Data.load_data_GRF(biorbd_model[0], T_stance, number_shooting_points[0])
        markers_ref.append(Data.load_data_markers(biorbd_model[0], T_stance, number_shooting_points[0], "stance"))
        markers_ref.append(
            Data.load_data_markers(biorbd_model[1], phase_time[1], number_shooting_points[1], "swing"))
        q_ref.append(Data.load_q_kalman(biorbd_model[0], T_stance, number_shooting_points[0], "stance"))
        q_ref.append(Data.load_q_kalman(biorbd_model[0], phase_time[1], number_shooting_points[1], "swing"))
        emg_ref.append(Data.load_data_emg(biorbd_model[0], T_stance, number_shooting_points[0], "stance"))
        emg_ref.append(Data.load_data_emg(biorbd_model[-1], phase_time[-1], number_shooting_points[-1], "swing"))
        excitation_ref = []
        for i in range(len(phase_time)):
            excitation_ref.append(Data.load_muscularExcitation(emg_ref[i]))

        dict["phase_time"] = phase_time
        dict["time"] = t
        dict["grf_ref"] = grf_ref
        dict["markers_ref"] = markers_ref
        dict["q_ref"] = q_ref
        dict["excitation_ref"] = excitation_ref
        results.append(dict)

        # --- Generalized positions indicators --- #
        # init
        Q_ref = np.zeros((nb_q, np.sum(number_shooting_points) + 1))
        Q_ref[:, :number_shooting_points[0] + 1] = q_ref[0]
        Q_ref[:, number_shooting_points[0]:] = q_ref[1]

        # compute RMSE
        diff_q = np.sqrt((dict["q"] - Q_ref) * (dict["q"] - Q_ref))
        mean_diff_q = np.zeros(nb_q)
        for i in range(nb_q):
            mean_diff_q[i] = np.mean(diff_q[i, :])
        dict["mean_diff_q"] = mean_diff_q

        # compute pic error
        max_diff = np.zeros(nb_q)
        min_diff = np.zeros(nb_q)
        for i in range(nb_q):
            max_diff[i] = np.sqrt((np.max(dict["q"][i, :]) - np.max(Q_ref[i, :])) * (np.max(dict["q"][i, :]) - np.max(Q_ref[i, :])))
            min_diff[i] = np.sqrt((np.min(dict["q"][i, :]) - np.min(Q_ref[i, :])) * (np.min(dict["q"][i, :]) - np.min(Q_ref[i, :])))
        dict["max_diff"] = max_diff
        dict["min_diff"] = min_diff

        # compute R2
        mean_q = np.zeros((nb_q, np.sum(number_shooting_points) + 1))
        for i in range(nb_q):
            mean_q[i, :] = np.repeat(np.mean(dict["q"][i, :]), np.sum(number_shooting_points) + 1)
        diff_q_square = (dict["q"] - Q_ref) * (dict["q"] - Q_ref)
        diff_q_mean = (dict["q"] - mean_q) * (dict["q"] - mean_q)

        R2_q = np.zeros(nb_q)
        for i in range(nb_q):
            if (np.sum(diff_q_square[i, :]) > np.sum(diff_q_mean[i, :])):
                s = np.sum(diff_q_mean[i, :]) / np.sum(diff_q_square[i, :])
            else:
                s = np.sum(diff_q_square[i, :]) / np.sum(diff_q_mean[i, :])
            R2_q[i] = 1 - s
        dict["R2_q"] = R2_q

        # comparison
        R2_mean += R2_q
        RMSE_mean += mean_diff_q
        pic_max_mean += max_diff
        pic_min_mean += min_diff

        # --- Compute ground reaction forces --- #
        contact_forces = np.zeros((3, np.sum(number_shooting_points) + 1))
        for i in range(number_shooting_points[0] + 1):
            state = np.concatenate((dict["q"][:, i], dict["q_dot"][:, i], dict["activations"][:, i]))
            control = np.concatenate((dict["tau"][:, i], dict["excitations"][:, i]))
            contact_forces[:, i] = np.array(computeGRF(state, control, dict["params"])).squeeze()
        dict["contact_forces"] = contact_forces

        # --- Compute markers positions and difference --- #
        M_ref = np.zeros((3, nb_markers, np.sum(number_shooting_points) + 1))
        M_ref[:, :, :number_shooting_points[0] + 1] = markers_ref[0]
        M_ref[:, :, number_shooting_points[0]:] = markers_ref[1]

        markers_sol = np.ndarray((3, nb_markers, np.sum(number_shooting_points) + 1))
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
                markers_sol[:, j, i] = np.array(mark_func(vertcat(dict["q"][:, i], dict["q_dot"][:, i], dict["activations"][:, i]))).squeeze()

        diff_track = np.sqrt((markers_sol - M_ref) * (markers_sol - M_ref)) * 1e3
        hist_diff_track = np.zeros((3, nb_markers))
        for n_mark in range(nb_markers):
            hist_diff_track[0, n_mark] = sum(diff_track[0, n_mark, :]) / (np.sum(number_shooting_points) + 1)
            hist_diff_track[1, n_mark] = sum(diff_track[1, n_mark, :]) / (np.sum(number_shooting_points) + 1)
            hist_diff_track[2, n_mark] = sum(diff_track[2, n_mark, :]) / (np.sum(number_shooting_points) + 1)

        mean_diff_markers_track = np.zeros(nb_markers)
        for i in range(nb_markers):
            mean_diff_markers_track[i] = np.sum(hist_diff_track[:, i]) / 3
        dict["mean_diff_markers"] = mean_diff_markers_track
        dict["mean_diff_marker"] = np.sum(mean_diff_markers_track)/nb_markers
        Diff_marker += np.sum(mean_diff_markers_track)/nb_markers
        Diff_markers += mean_diff_markers_track


# mean between subject
R2_mean = R2_mean/(len(list_subjects) - 1)
RMSE_mean = RMSE_mean/(len(list_subjects) - 1)
pic_max_mean = pic_max_mean/(len(list_subjects) - 1)
pic_min_mean = pic_min_mean/(len(list_subjects) - 1)
Diff_marker = Diff_marker/(len(list_subjects) - 1)
Diff_markers = Diff_markers/(len(list_subjects) - 1)

# markers differences
figure = plt.figure("Mean markers differences (mm)")
plt.bar(
    np.linspace(0, nb_markers, nb_markers), Diff_markers, width=1.0, facecolor="b", edgecolor="k", alpha=0.5,
)
plt.xticks(np.arange(nb_markers), label_markers, rotation=90)
plt.plot([0, nb_markers], [Diff_marker, Diff_marker], "--r")
plt.ylim([0, 60])
plt.ylabel('Mean difference (mm)')

# markers differences per subject
figure, ax = plt.subplots()
leg = []
x = np.arange(nb_markers)
width = 1/len(list_subjects)
xplot = x - 3*width/2
for dic in results:
    leg.append(dic["name_subject"])
    plt.bar(
        xplot, dic["mean_diff_markers"], width=width, edgecolor="k", alpha=0.7,
    )
    xplot += width
plt.xticks(np.arange(nb_markers), label_markers, rotation=90)
plt.ylim([0, 60])
plt.ylabel('Mean difference (mm)')
plt.legend(leg)
plt.title("Mean markers differences per subject")
plt.show()

# Plot Q
figure, axes = plt.subplots(4, 3, sharex=True)
axes = axes.flatten()
for i in range(nb_q):
    for dic in results:
        Q = np.concatenate((dic["q_ref"][0][i, :], dic["q_ref"][1][i, 1:]))
        if (i>2):
            axes[i].plot(dic["q"][i, :]*180/np.pi, linestyle="-", linewidth=1)
        else:
            axes[i].plot(dic["q"][i, :], linestyle="-", linewidth=1)

    if (i > 2):
        axes[i].text(0, np.min(Q * 180 / np.pi), f"R2 : {np.round(R2_mean[i], 3)}")
        axes[i].set_ylabel('angle (degre)')
        axes[i].plot(
                [number_shooting_points[0], number_shooting_points[0]], [np.min(dic["q"][i, :]*180/np.pi), np.max(dic["q"][i, :]*180/np.pi)], color="k", linestyle="--", linewidth=1
            )
    else:
        axes[i].text(0, np.min(Q), f"R2 : {np.round(R2_mean[i], 3)}")
        axes[i].set_ylabel('position (m)')
        axes[i].plot(
                [number_shooting_points[0], number_shooting_points[0]], [np.min(dic["q"][i, :]), np.max(dic["q"][i, :])], color="k", linestyle="--", linewidth=1
            )
    axes[i].set_title(q_name[i])
    axes[i].grid(color="k", linestyle="--", linewidth=0.5)
axes[0].legend(leg)
plt.show()


figure, axes = plt.subplots(1, 3)
axes[0].bar(
    np.linspace(0, nb_q, nb_q), RMSE_mean, width=1.0, facecolor="b", edgecolor="k", alpha=0.5,
)
axes[0].xticks(np.arange(nb_q), q_name, rotation=90)


