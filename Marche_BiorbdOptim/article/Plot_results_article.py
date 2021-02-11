import numpy as np
from casadi import dot, Function, vertcat, MX, mtimes, nlpsol, mmax
import biorbd
import bioviz
import seaborn
from matplotlib import pyplot as plt


def show_q_bioviz(q):
    b = bioviz.Viz(loaded_model=model)
    b.set_q(q)

def plot_muscles_1(activations):
    label_muscles_1 = ("Gluteus Maximus", "Hamstrings", "Vastus Medialis", "Rectus Femoris", "Gastrocnemius Medialis", "Tibialis Anterior")

    ACT_plot_1 = np.zeros(len(label_muscles_1))
    ACT_plot_1[0] = np.mean(activations[:3])  # Gluteus Maximus
    ACT_plot_1[1] = np.mean(activations[6:9])  # Hamstrings
    ACT_plot_1[2] = activations[10]  # Vastus Medialis
    ACT_plot_1[3] = activations[9]  # Rectus Femoris
    ACT_plot_1[4] = activations[13]  # Triceps Surae
    ACT_plot_1[5] = activations[16]  # Tibial Anterior

    fig, ax = plt.subplots()
    for m in range(len(label_muscles_1)):
        act = np.zeros(len(label_muscles_1))
        act[m] = ACT_plot_1[m]
        plt.bar(np.arange(6), act, width=1.0, edgecolor='k')
    plt.xticks(np.arange(6), label_muscles_1, rotation=90)
    plt.ylabel("Muscle activation")
    plt.ylim([0.0, 1.0])
    plt.xlim([-0.5, 5.5])

def plot_muscles_2(activations):
    label_muscles_2 = ("Gluteus Maximus", "Semimembranous", "Vastus Medialis", "Rectus Femoris",
                       "Gastrocnemius Medialis", "Tibialis Anterior")
    ACT_plot_2 = np.zeros(len(label_muscles_2))
    ACT_plot_2[0] = np.mean(activations[:3])  # Gluteus Maximus
    ACT_plot_2[1] = activations[6]  # Semimembranous
    ACT_plot_2[2] = activations[10]  # Vastus Medialis
    ACT_plot_2[3] = activations[9]  # Rectus Femoris
    ACT_plot_2[4] = activations[13]  # Gastrocnemius Medialis
    ACT_plot_2[5] = activations[16]  # Tibial Anterior

    fig, ax = plt.subplots()
    plt.bar(np.arange(6) - 0.5, ACT_plot_2, width=1.0)
    plt.xticks(np.arange(6) - 0.5, label_muscles_2, rotation=90)
    plt.ylabel("Muscle activation")
    plt.ylim([0.0, 1.0])
    plt.xlim([-1.0, 5.0])


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
seaborn.set_style("whitegrid")
seaborn.color_palette()

list_idx = (1,9,20,40,53,70,80,90)
for i in list_idx:
    plot_muscles_1(activations[:,i]) # Test 1
    # plot_muscles_2(activations[:, i])  # Test 2

b = bioviz.Viz(loaded_model=model)
b.set_q(q[:, 90])


# show_q_bioviz(q[:, 2])

# --- Load movements --- #
b = bioviz.Viz(loaded_model=model)
b.load_movement(q)
b.exec()



