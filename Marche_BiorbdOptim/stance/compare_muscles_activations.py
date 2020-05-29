from scipy.integrate import solve_ivp
import numpy as np
from casadi import dot, Function, vertcat, MX, tanh
from matplotlib import pyplot as plt
import biorbd

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
            f = 0.5 * tanh(0.1*(e[i] - a[i]))
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
        activation_ref[:, i + 1]=activation_init

    # t = np.linspace(0, final_time, nb_shooting + 1)
    #
    # def plot_control(ax, t, x, color='b'):
    #     nbPoints = len(np.array(x))
    #     for n in range(nbPoints - 1):
    #         ax.plot([t[n], t[n + 1], t[n + 1]], [x[n], x[n], x[n + 1]], color)
    #
    # figure2, axes2 = plt.subplots(4, 5, sharex=True)
    # axes2 = axes2.flatten()
    # for i in range(biorbd_model.nbMuscleTotal()):
    #     name_mus = biorbd_model.muscle(i).name().to_string()
    #     plot_control(axes2[i], t, emg_ref[i, :], color='r')
    #     axes2[i].set_title(name_mus)
    #     axes2[i].set_ylim([0, 1])
    #     axes2[i].set_yticks(np.arange(0, 1, step=1 / 5, ))
    #     axes2[i].grid(color="k", linestyle="--", linewidth=0.5)
    #     for j in range(nb_shooting):
    #         t2 = np.linspace(t[j], t[j+1], sol_act[j].shape[1])
    #         axes2[i].plot(t2, sol_act[j][i, :], 'b-')
    #         axes2[i].plot(t2[-1], sol_act[j][i, -1], 'b.')
    # axes2[-1].remove()
    # axes2[-2].remove()
    # axes2[-3].remove()

    return activation_ref

def plot_control(ax, t, x, color='b'):
    nbPoints = len(np.array(x))
    for n in range(nbPoints - 1):
        ax.plot([t[n], t[n + 1], t[n + 1]], [x[n], x[n], x[n + 1]], color)


# Define the problem
biorbd_model = biorbd.Model("../../ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact_deGroote.bioMod")
n_shooting_points = 25
Gaitphase = 'stance'

# Generate data from file
from Marche_BiorbdOptim.LoadData import load_data_markers, load_data_q, load_data_emg, load_data_GRF, load_muscularExcitation

name_subject = "equincocont01"
grf_ref, T, T_stance, T_swing = load_data_GRF(name_subject, biorbd_model, n_shooting_points)
final_time = T_stance

t, markers_ref = load_data_markers(name_subject, biorbd_model, final_time, n_shooting_points, Gaitphase)
emg_ref = load_data_emg(name_subject, biorbd_model, final_time, n_shooting_points, Gaitphase)
excitation_ref = load_muscularExcitation(emg_ref)
activation_ref = generate_activation(biorbd_model=biorbd_model, final_time=final_time, nb_shooting=n_shooting_points, emg_ref=excitation_ref)
mus = np.load('./RES/equincocont01/activations/mus/mus.npy')
mus_int = np.load("./RES/equincocont01/activations/mus_int/mus_int.npy")
mus_state = np.load("./RES/equincocont01/excitations/mus_state/mus_state.npy")
mus_shift_ref = np.concatenate((mus[:, 0:1], mus[:, :-1]), axis=1)
mus_shift = np.load("./RES/equincocont01/activations/mus_shift/mus_shift.npy")

rms_mus = np.zeros(17)
rms_mus_state = np.zeros(17)
rms_mus_int = np.zeros(17)
rms_mus_shift = np.zeros(17)
rms_mus_shift_ref=np.zeros(17)
for i in range(biorbd_model.nbMuscleTotal()):
    rms_mus[i] = np.sqrt(np.mean((mus[i, :] - activation_ref[i, :]) * (mus[i, :] - activation_ref[i, :])))
    rms_mus_int[i] = np.sqrt(np.mean((mus_int[i, :] - activation_ref[i, :]) * (mus_int[i, :] - activation_ref[i, :])))
    rms_mus_state[i] = np.sqrt(np.mean((mus_state[i, :] - activation_ref[i, :]) * (mus_state[i, :] - activation_ref[i, :])))
    rms_mus_shift[i] = np.sqrt(np.mean((mus_shift[i, :] - activation_ref[i, :]) * (mus_shift[i, :] - activation_ref[i, :])))
    rms_mus_shift_ref[i] = np.sqrt(np.mean((mus_shift_ref[i, :] - activation_ref[i, :]) * (mus_shift_ref[i, :] - activation_ref[i, :])))

mean_mus = np.mean(rms_mus)
mean_mus_int = np.mean(rms_mus_int)
mean_mus_shift = np.mean(rms_mus_shift)
mean_mus_state = np.mean(rms_mus_state)
mean_mus_post_shift = np.mean(rms_mus_shift_ref)

print(f"rms activation as emg     : {mean_mus}")
print(f"rms activation as emg with next node     : {mean_mus_post_shift}")
print(f"rms integrated activation : {mean_mus_int}")
print(f"rms shifted activation    : {mean_mus_shift}")
print(f"rms excitation            : {mean_mus_state}")


# rms_mus = np.sqrt(np.mean((mus - activation_ref) * (mus - activation_ref)))
# rms_mus_int = np.sqrt(np.mean((mus_int - activation_ref) * (mus_int - activation_ref)))
# rms_mus_state = np.sqrt(np.mean((mus_state - activation_ref) * (mus_state - activation_ref)))
# rms_mus_shift = np.sqrt(np.mean((mus_shift - activation_ref) * (mus_shift - activation_ref)))

figure, axes = plt.subplots(4, 5, sharex=True)
axes = axes.flatten()
for i in range(biorbd_model.nbMuscleTotal()):
    name_mus = biorbd_model.muscle(i).name().to_string()
    plot_control(axes[i], t, excitation_ref[i, :], color='k--')
    plot_control(axes[i], t, activation_ref[i, :], color='k')
    plot_control(axes[i], t, mus[i, :], color='r')
    plot_control(axes[i], t, mus_int[i, :], color='b')
    plot_control(axes[i], t, mus_state[i, :], color='b')
    # axes[i].plot(t, mus_state[i, :], 'g.-')
    plot_control(axes[i], t, mus_shift[i, :], 'm')
    # plot_control(axes[i], t, mus_shift_ref[i, :], 'y')
    axes[i].set_title(name_mus)
    axes[i].set_ylim([0, 1])
    axes[i].set_yticks(np.arange(0, 1, step=1 / 5, ))
    axes[i].grid(color="k", linestyle="--", linewidth=0.5)
axes[-1].remove()
axes[-2].remove()
axes[-3].remove()

figure2, axes2 = plt.subplots(4, 5, sharex=True)
axes2 = axes2.flatten()
plt.title('difference with the reference')
for i in range(biorbd_model.nbMuscleTotal()):
    name_mus = biorbd_model.muscle(i).name().to_string()
    plot_control(axes2[i], t, mus[i, :] - activation_ref[i, :], color='r')
    plot_control(axes2[i], t, mus_int[i, :] - activation_ref[i, :], color='b')
    plot_control(axes[i], t, mus_state[i, :] - activation_ref[i, :], color='b')
    # axes2[i].plot(t, mus_state[i, :] - activation_ref[i, :], 'g.-')
    plot_control(axes2[i], t, mus_shift[i, :] - activation_ref[i, :], 'm')
    # plot_control(axes2[i], t, mus_shift_ref[i, :] - activation_ref[i, :], 'y')
    axes2[i].set_title(name_mus)
    axes2[i].grid(color="k", linestyle="--", linewidth=0.5)
axes2[-1].remove()
axes2[-2].remove()
axes2[-3].remove()

plt.show()
