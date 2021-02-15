import numpy as np
from casadi import dot, Function, vertcat, MX, mtimes, nlpsol, mmax
import biorbd
import bioviz
import seaborn
from matplotlib import pyplot as plt


def show_q_bioviz(q):
    b = bioviz.Viz(loaded_model=model)
    b.set_q(q)


def plot_muscles(activations):
    seaborn.set_style("whitegrid")
    seaborn.color_palette('deep')

    label_muscles = ["Gluteus Maximus",
                       "Hamstrings",
                       "Vastus Medialis",
                       "Rectus Femoris",
                       "Gastrocnemius Medialis",
                       "Tibialis Anterior"]
    act_plot = [np.mean(activations[:3]), # Gluteus Maximus
                np.mean(activations[6:9]),  # Hamstrings
                activations[10],  # Vastus Medialis
                activations[9],  # Rectus Femoris
                activations[13],  # Gastoc Med
                activations[16]  # Tibial Anterior
                ]

    seaborn.barplot(label_muscles, act_plot)
    plt.ylabel("Muscle activation")
    plt.ylim([0.0, 1.0])

def plot_muscles_2(activations):
    seaborn.color_palette('deep')
    label_muscles = ("Gluteus Maximus",
                       "Hamstrings",
                       "Vastus Medialis",
                       "Rectus Femoris",
                       "Gastrocnemius Medialis",
                       "Tibialis Anterior")
    ACT_plot = np.zeros(len(label_muscles))
    ACT_plot[0] = np.mean(activations[:3])  # Gluteus Maximus
    ACT_plot[1] = activations[6]  # Semimembranous
    ACT_plot[2] = activations[10]  # Vastus Medialis
    ACT_plot[3] = activations[9]  # Rectus Femoris
    ACT_plot[4] = activations[13]  # Gastrocnemius Medialis
    ACT_plot[5] = activations[16]  # Tibial Anterior

    fig, ax = plt.subplots()
    for m in range(len(label_muscles)):
        act = np.zeros(len(label_muscles))
        act[m]=ACT_plot[m]
        plt.bar(np.arange(6), act, width=1.0, edgecolor='k')
    plt.xticks(np.arange(6), label_muscles, rotation=90)
    plt.ylabel("Muscle activation")
    plt.yticks(np.arange(0.0, 1.5, 0.5))
    plt.ylim([0.0, 1.0])
    plt.xlim([-0.5, 5.5])


# --- Load model and parameters --- #
model = biorbd.Model("Modeles/Gait_1leg_12dof_heel.bioMod")
nb_q = model.nbQ()
nb_mus = model.nbMuscleTotal()
nb_tau = model.nbGeneralizedTorque()

# --- Load results --- #
q = np.load("q.npy")
qdot = np.load("qdot.npy")
tau = np.load("tau.npy")
activations = np.load("activation.npy")

# --- Plot muscles activity --- #

list_idx = (1,9,53,60,80,90)
for i in list_idx:
    plot_muscles_2(activations[:, i])  # Test 2
plt.show()

b = bioviz.Viz(loaded_model=model, show_local_ref_frame=False)
b.set_q(q[:, 90])

#
# # show_q_bioviz(q[:, 2])
#
# # --- Load movements --- #
# b = bioviz.Viz(loaded_model=model)
# b.load_movement(q)
# b.exec()



