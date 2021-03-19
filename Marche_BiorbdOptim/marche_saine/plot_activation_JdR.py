import numpy as np
from casadi import dot, Function, vertcat, MX, mtimes, nlpsol, mmax
import biorbd
import bioviz
import seaborn
from matplotlib import pyplot as plt

def compute_mean_activation(activation, number_shooting_points):
    mean_activation = []
    n_shoot = 0
    for p in range(len(number_shooting_points)):
        mean_activation.append(np.mean(activation[n_shoot:n_shoot + number_shooting_points[p]]))
        n_shoot+=number_shooting_points[p]
    return mean_activation

def plot_bar_mean_activity(activation_hip, activation_no_hip, number_shooting_points, muscle_name):
    # seaborn.set_style("whitegrid")
    seaborn.color_palette('deep')

    mean_hip = compute_mean_activation(activation_hip, number_shooting_points)
    mean_no_hip = compute_mean_activation(activation_no_hip, number_shooting_points)

    label_phases = ["Talon",
                    "Pied a plat",
                    "Avant pied",
                    "Swing"]
    x = np.arange(len(label_phases))
    width = 0.4
    fig, ax = plt.subplots()
    rect_hip = ax.bar(x - width / 2, mean_hip, width, color='tab:blue', label='avec iliopsoas')
    rect_no_hip = ax.bar(x + width / 2, mean_no_hip, width, color='lightsteelblue', label='sans iliopsoas')
    ax.set_ylabel("Activation musculaire")
    ax.set_ylim([0.0, 1.0])
    ax.set_yticks(np.arange(0.0, 1.5, 0.5))
    ax.set_title(muscle_name)
    ax.set_xticks(x)
    ax.set_xticklabels(label_phases)
    ax.legend()


model=biorbd.Model("models/Gait_1leg_12dof_heel.bioMod")
# activations_hip = np.load('./RES/muscle_driven/Hip_muscle/muscle.npy')
# activations_no_hip = np.load('./RES/muscle_driven/No_hip/muscle.npy')
activations_hip = np.load('/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/article/muscle_hip/hip2/activation_hip.npy')
activations_no_hip = np.load('/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/article/activation.npy')
number_shooting_points = [5, 35, 16, 38]

plot_bar_mean_activity(activations_hip[11, :], activations_no_hip[9, :], number_shooting_points, muscle_name='Rectus Femoris')
plot_bar_mean_activity(activations_hip[-1, :], activations_no_hip[-1, :], number_shooting_points, muscle_name='Tibial anterieur')
plot_bar_mean_activity(activations_hip[15, :], activations_no_hip[13, :], number_shooting_points, muscle_name='Gastrocnemien medial')
plot_bar_mean_activity(sum(activations_hip[15:18, :])/3, sum(activations_no_hip[13:16, :])/3, number_shooting_points, muscle_name='Triceps sural')
plt.show()

