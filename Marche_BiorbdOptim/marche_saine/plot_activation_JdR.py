import numpy as np
from casadi import dot, Function, vertcat, MX, mtimes, nlpsol, mmax
import biorbd
import bioviz
import seaborn
from matplotlib import pyplot as plt


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

def get_muscle_name(model):
    muscle_name = []
    for m in range(model.nbMuscleTotal()):
        muscle_name.append(model.muscle(m).name().to_string())
    return muscle_name

def muscles_tau(model, q, qdot, activation):
    muscles_states = model.stateSet()
    for k in range(model.nbMuscles()):
        muscles_states[k].setActivation(activation[k])
    muscles_tau = model.muscularJointTorque(muscles_states, q, qdot).to_mx()
    return muscles_tau

def muscular_torque(model):
    qMX = MX.sym("qMX", model.nbQ(), 1)
    dqMX = MX.sym("dqMX", model.nbQ(), 1)
    aMX = MX.sym("aMX", model.nbMuscles(), 1)
    return Function("MuscleTorque", [qMX, dqMX, aMX],
                    [muscles_tau(model, qMX, dqMX, aMX)],
                    ["qMX", "dqMX", "aMX"], ["Torque"]).expand()

def muscles_forces(model, q, qdot, activation):
    muscles_states = model.stateSet()
    for k in range(model.nbMuscles()):
        muscles_states[k].setActivation(activation[k])
    muscles_forces = model.muscleForces(muscles_states, q, qdot).to_mx()
    return muscles_forces

def muscular_force(model):
    qMX = MX.sym("qMX", model.nbQ(), 1)
    dqMX = MX.sym("dqMX", model.nbQ(), 1)
    aMX = MX.sym("aMX", model.nbMuscles(), 1)
    return Function("MuscleTorque", [qMX, dqMX, aMX],
                    [muscles_forces(model, qMX, dqMX, aMX)],
                    ["qMX", "dqMX", "aMX"], ["Forces"]).expand()
def muscles_jac(model):
    qMX = MX.sym("qMX", model.nbQ(), 1)
    return Function("MuscleJac", [qMX], [model.musclesLengthJacobian(qMX).to_mx()],["qMX"], ["momentarm"]).expand()

def force_isometric(model):
    fiso = []
    for nGrp in range(model.nbMuscleGroups()):
        for nMus in range(model.muscleGroup(nGrp).nbMuscles()):
            fiso.append(model.muscleGroup(nGrp).muscle(nMus).characteristics().forceIsoMax().to_mx())
    return vertcat(*fiso)

def get_force_iso(model):
    return Function("Fiso", [], [force_isometric(model)], [], ["fiso"]).expand()

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
    rect_hip = ax.bar(x - width / 2, mean_hip, width, color='tab:red', label='avec iliopsoas')
    rect_no_hip = ax.bar(x + width / 2, mean_no_hip, width, color='lightsteelblue', label='sans iliopsoas')
    ax.set_ylabel("Activation musculaire")
    ax.set_ylim([0.0, 1.0])
    ax.set_yticks(np.arange(0.0, 1.5, 0.5))
    ax.set_title(muscle_name)
    ax.set_xticks(x)
    ax.set_xticklabels(label_phases)
    ax.legend()


model=biorbd.Model("models/Gait_1leg_12dof_heel.bioMod")
nb_mus = model.nbMuscles()
nb_q = model.nbQ()

q_hip = np.load('/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/article/muscle_hip/hip2/q_hip.npy')
qdot_hip = np.load('/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/article/muscle_hip/hip2/qdot_hip.npy')
tau_hip = np.load('/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/article/muscle_hip/hip2/tau_hip.npy')
activations_hip = np.load('/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/article/muscle_hip/hip2/activation_hip.npy')

q_no_hip = np.load('/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/article/q.npy')
qdot_no_hip = np.load('/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/article/qdot.npy')
tau_no_hip = np.load('/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/article/tau.npy')
a = np.load('/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/article/activation.npy')
activations_no_hip = np.zeros((nb_mus, a.shape[1]))
activations_no_hip[:6, :] = a[:6, :]
activations_no_hip[8:, :] = a[6:, :]
number_shooting_points = [5, 35, 16, 38]

# CASADI FUNCTION
get_muscle_torque = muscular_torque(model)
get_jacobian = muscles_jac(model)
get_muscle_force = muscular_force(model)
get_forceIso = get_force_iso(model)

# INIT
muscular_torque_hip=np.zeros((nb_q, q_hip.shape[1]))
muscular_force_hip=np.zeros((nb_mus, q_hip.shape[1]))
jacobian_hip = np.zeros((nb_mus, nb_q, q_hip.shape[1]))
muscular_individual_torque_hip=np.zeros((nb_mus, nb_q, q_hip.shape[1]))

muscular_torque_no_hip=np.zeros((nb_q, q_no_hip.shape[1]))
muscular_force_no_hip=np.zeros((nb_mus, q_no_hip.shape[1]))
jacobian_no_hip = np.zeros((nb_mus, nb_q, q_no_hip.shape[1]))
muscular_individual_torque_no_hip=np.zeros((nb_mus, nb_q, q_no_hip.shape[1]))

# COMPUTE MUSCLE FORCE
fiso = get_forceIso()['fiso']
for j in range(nb_mus):
    for i in range(q_hip.shape[1]):
        muscular_force_hip[j, i]=get_muscle_force(q_hip[:, i], qdot_hip[:, i], activations_hip[:, i])[j, :]
        muscular_force_no_hip[j, i] = get_muscle_force(q_no_hip[:, i], qdot_no_hip[:, i], activations_no_hip[:, i])[j, :]

# COMPUTE JACOBIAN
for i in range(q_hip.shape[1]):
    jacobian_hip[:, :, i] = get_jacobian(q_hip[:, i])
    jacobian_no_hip[:, :, i] = get_jacobian(q_no_hip[:, i])

# COMPUTE INDIVIDUAL TORQUE
for i in range(nb_q):
    for j in range(nb_mus):
        muscular_individual_torque_hip[j, i, :] = -jacobian_hip[j, i, :] * muscular_force_hip[j, :]
        muscular_individual_torque_no_hip[j, i, :] = -jacobian_no_hip[j, i, :] * muscular_force_no_hip[j, :]

# COMPUTE MUSCLE TORQUE
for j in range(nb_q):
    for i in range(q_hip.shape[1]):
        muscular_torque_hip[j, i]=get_muscle_torque(q_hip[:, i], qdot_hip[:, i], activations_hip[:, i])[j, :]
        muscular_torque_no_hip[j, i] = get_muscle_torque(q_no_hip[:, i], qdot_no_hip[:, i], activations_no_hip[:, i])[j, :]

t = np.linspace(0, q_hip.shape[1]*0.01, q_hip.shape[1])
plt.figure()
plt.plot(t, muscular_individual_torque_hip[11, 8, :], color='tab:blue')
plt.plot(t, muscular_individual_torque_hip[6, 8, :], color='tab:red')
plt.plot(t, muscular_individual_torque_hip[7, 8, :], color='tab:brown')

plt.plot(t, muscular_individual_torque_no_hip[11, 8, :], color='tab:blue', alpha=0.5)
plt.plot([t[0], t[-1]], [0, 0], 'k--')
plt.ylim([-50, 150])
plt.xlim([t[0], t[-1]])
plt.legend(['droit fémoral', 'iliaque', 'psoas'])

plt.figure()
plt.plot(t, -muscular_individual_torque_hip[11, 9, :], color='tab:blue')
plt.plot(t, -muscular_individual_torque_hip[15, 9, :], color='tab:green')

plt.plot(t, -muscular_individual_torque_no_hip[11, 9, :], color='tab:blue', alpha=0.5)
plt.plot(t, -muscular_individual_torque_no_hip[15, 9, :], color='tab:green', alpha=0.5)
plt.plot([t[0], t[-1]], [0, 0], 'k--')
plt.ylim([-200, 100])
plt.xlim([t[0], t[-1]])
plt.legend(['droit fémoral', 'gastrocnémien médial'])

plt.figure()
plt.plot(t, muscular_individual_torque_hip[15, 11, :], color='tab:green')
plt.plot(t, muscular_individual_torque_hip[18, 11, :], color='tab:orange')

plt.plot(t, muscular_individual_torque_no_hip[15, 11, :], color='tab:green', alpha=0.5)
plt.plot(t, muscular_individual_torque_no_hip[18, 11, :], color='tab:orange', alpha=0.5)
plt.plot([t[0], t[-1]], [0, 0], 'k--')
plt.xlim([t[0], t[-1]])
plt.ylim([-200, 150])
plt.legend(['gastrocnémien médial', 'tibial antérieur'])

plot_bar_mean_activity(activations_hip[11, :], activations_no_hip[11, :], number_shooting_points, muscle_name='Rectus Femoris')
plot_bar_mean_activity(activations_hip[-1, :], activations_no_hip[-1, :], number_shooting_points, muscle_name='Tibial anterieur')
plot_bar_mean_activity(activations_hip[15, :], activations_no_hip[15, :], number_shooting_points, muscle_name='Gastrocnemien medial')
plot_bar_mean_activity(activations_hip[6, :], np.zeros(activations_no_hip[13, :].shape[0]), number_shooting_points, muscle_name='Iliaque')
plot_bar_mean_activity(activations_hip[7, :], np.zeros(activations_no_hip[13, :].shape[0]), number_shooting_points, muscle_name='Psoas')
plot_bar_mean_activity(sum(activations_hip[15:18, :])/3, sum(activations_no_hip[15:18, :])/3, number_shooting_points, muscle_name='Triceps sural')
plt.show()

