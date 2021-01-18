import numpy as np
from casadi import dot, Function, vertcat, MX, mtimes, nlpsol
import biorbd
import bioviz
from matplotlib import pyplot as plt
from Marche_BiorbdOptim.LoadData import Data_to_track
from Marche_BiorbdOptim.marche_saine.Affichage_resultats import Affichage


def plot_control(ax, t, x, color="k", linestyle="--", linewidth=1):
    nbPoints = len(np.array(x))
    for n in range(nbPoints - 1):
        ax.plot([t[n], t[n + 1], t[n + 1]], [x[n], x[n], x[n + 1]], color, linestyle, linewidth)

def get_time_vector(phase_time, nb_shooting):
    nb_phases = len(phase_time)
    t = np.linspace(0, phase_time[0], nb_shooting[0] + 1)
    for p in range(1, nb_phases):
        t = np.concatenate((t[:-1], t[-1] + np.linspace(0, phase_time[p], nb_shooting[p] + 1)))
    return t

def get_q_name(model):
    q_name = []
    for s in range(model.nbSegment()):
        seg_name = model.segment(s).name().to_string()
        for d in range(model.segment(s).nbDof()):
            dof_name = model.segment(s).nameDof(d).to_string()
            q_name.append(seg_name + "_" + dof_name)
    return q_name

def get_q_range(model):
    q_max = []
    q_min = []
    for s in range(model.nbSegment()):
        q_range = model.segment(s).QRanges()
        for r in q_range:
            q_max.append(r.max())
            q_min.append(r.min())
    return q_max, q_min


# Define the problem -- model path
biorbd_model = (
    biorbd.Model("../../ModelesS2M/Marche_saine/Florent_1leg_12dof_heel.bioMod"),
    biorbd.Model("../../ModelesS2M/Marche_saine/Florent_1leg_12dof_flatfoot.bioMod"),
    biorbd.Model("../../ModelesS2M/Marche_saine/Florent_1leg_12dof_forefoot.bioMod"),
    biorbd.Model("../../ModelesS2M/Marche_saine/Florent_1leg_12dof_0contact.bioMod")
)

# Problem parameters
dt = 0.01
nb_q = biorbd_model[0].nbQ()
nb_qdot = biorbd_model[0].nbQdot()
nb_tau = biorbd_model[0].nbGeneralizedTorque()
nb_mus = biorbd_model[0].nbMuscleTotal()
nb_phases = len(biorbd_model)

# Generate data from file
Data_to_track = Data_to_track("normal01", model=biorbd_model[0], multiple_contact=True)
phase_time = Data_to_track.GetTime()
number_shooting_points = []
for time in phase_time:
    number_shooting_points.append(int(time / 0.01))
grf_ref = Data_to_track.load_data_GRF(number_shooting_points)  # get ground reaction forces
markers_ref = Data_to_track.load_data_markers(number_shooting_points)
q_ref = Data_to_track.load_q_kalman(number_shooting_points)
qdot_ref = Data_to_track.load_qdot_kalman(number_shooting_points)
CoP = Data_to_track.load_data_CoP(number_shooting_points)
M_ref = Data_to_track.load_data_Moment_at_CoP(number_shooting_points)
EMG_ref = Data_to_track.load_data_emg(number_shooting_points)
excitations_ref = []
for p in range(nb_phases):
    excitations_ref.append(Data_to_track.load_muscularExcitation(EMG_ref[p]))

Q_ref = np.zeros((nb_q, sum(number_shooting_points) + 1))
for i in range(nb_q):
    n_shoot = 0
    for n in range(nb_phases):
        Q_ref[i, n_shoot:n_shoot + number_shooting_points[n] + 1] = q_ref[n][i, :]
        n_shoot += number_shooting_points[n]

E_ref = np.zeros((nb_mus, sum(number_shooting_points) + 1))
for i in range(nb_mus):
    n_shoot = 0
    for n in range(nb_phases):
        E_ref[i, n_shoot:n_shoot + number_shooting_points[n] + 1] = excitations_ref[n][i, :]
        n_shoot += number_shooting_points[n]


t = get_time_vector(phase_time, number_shooting_points)
q_name = get_q_name(biorbd_model[0])
q_max, q_min = get_q_range(biorbd_model[0])

# LOAD RESULTS
# --- q ---
q_init = np.load('./RES/1leg/cycle/muscles/3_contacts/q.npy')
q_2 = np.load('./RES/1leg/cycle/muscles/3_contacts/fort/square/q.npy')
q_3 = np.load('./RES/1leg/cycle/muscles/3_contacts/fort/cube/q.npy')
# q_SO = np.load('./RES/1leg/cycle/muscles/3_contacts/SO/q.npy')

# --- muscles ---
activation_init = np.load('./RES/1leg/cycle/muscles/3_contacts/activation.npy')
activation_2 = np.load('./RES/1leg/cycle/muscles/3_contacts/fort/square/activation.npy')
activation_3 = np.load('./RES/1leg/cycle/muscles/3_contacts/fort/cube/activation.npy')
activation_SO = np.load('./RES/1leg/cycle/muscles/3_contacts/SO/activation.npy')

# PLOT Q
figure, axes = plt.subplots(3, 4)
axes = axes.flatten()
for i in range(nb_q):
    axes[i].plot(t, Q_ref[i, :], color="k", linestyle="--")
    axes[i].plot(t, q_init[i, :], color="m", linestyle="-")
    axes[i].plot(t, q_2[i, :], color="b", linestyle="-")
    axes[i].plot(t, q_3[i, :], color="g", linestyle="-")

    # plot phase transition
    if (nb_phases > 1):
        pt = 0
        for p in range(nb_phases):
            pt += phase_time[p]
            axes[i].plot([pt, pt], [q_min[i], q_max[i]], color='k', linestyle="--", linewidth=0.7)
    axes[i].set_title(q_name[i])
    axes[i].set_ylim([q_min[i], q_max[i]])
plt.legend(['reference', 'init', 'fort', 'fort 3'])

# PLOT ACTIVATION
figure, axes = plt.subplots(3, 6)
axes = axes.flatten()
for i in range(nb_mus):
    plot_control(axes[i], t, E_ref[i, :])
    plot_control(axes[i], t, activation_init[i, :], color="m", linestyle="-")
    plot_control(axes[i], t, activation_2[i, :], color="b", linestyle="-")
    plot_control(axes[i], t, activation_3[i, :], color="g", linestyle="-")
    # plot_control(axes[i], t, activation_SO[i, :], color="y", linestyle="-")

    if (nb_phases > 1):
        pt = 0
        for p in range(nb_phases):
            pt += phase_time[p]
            axes[i].plot([pt, pt], [np.min(E_ref), np.max(E_ref)], color='k', linestyle="--", linewidth=0.7)
    axes[i].set_title(biorbd_model[0].muscle(i).name().to_string())
    axes[i].set_ylim([0.0, 1.01])

plt.show()