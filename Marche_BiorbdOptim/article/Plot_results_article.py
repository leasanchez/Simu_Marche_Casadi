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
    plt.bar(np.arange(6), ACT_plot, width=1.0)
    plt.xticks(np.arange(6), label_muscles, rotation=90)
    plt.ylabel("Muscle activation")
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

list_idx = (1,9,20,40,53,70,80,90)
for i in list_idx:
    plot_muscles(activations[:,i]) # Test 1
    plot_muscles_2(activations[:, i])  # Test 2

b = bioviz.Viz(loaded_model=model)
b.set_q(q[:, 90])


# show_q_bioviz(q[:, 2])

# --- Load movements --- #
b = bioviz.Viz(loaded_model=model)
b.load_movement(q)
b.exec()



