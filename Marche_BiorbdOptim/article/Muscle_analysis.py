import numpy as np
from casadi import dot, Function, vertcat, MX, mtimes, nlpsol, mmax
import biorbd
import bioviz
from matplotlib import pyplot as plt


# FONCTIONS
def muscles_tau(model, q, qdot, activation):
    muscles_states = biorbd.VecBiorbdMuscleState(model.nbMuscles())
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
    muscles_states = biorbd.VecBiorbdMuscleState(model.nbMuscles())
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

# PARAM
# --- Load model and parameters --- #
model = biorbd.Model("Modeles/Gait_1leg_12dof_heel.bioMod")
nb_q = model.nbQ()
nb_mus = model.nbMuscleTotal()
nb_tau = model.nbGeneralizedTorque()

# --- Load results --- #
q = np.load("q.npy")
qdot = np.load("qdot.npy")
qddot = np.diff(qdot)
tau_residual = np.load("tau.npy")
activations = np.load("activation.npy")
nb_shooting = q.shape[1] - 1

# --- Casadi function --- #
get_muscle_torque = muscular_torque(model)
get_jacobian = muscles_jac(model)
get_muscle_force = muscular_force(model)
get_forceIso = get_force_iso(model)

# --- Init --- #
muscularTorque=np.zeros((nb_q, q.shape[1]))
muscularSumTorque=np.zeros((nb_q, q.shape[1]))
muscularIndividualTorque=np.zeros((nb_mus, nb_q, q.shape[1]))
muscularIndividualPower=np.zeros((nb_mus, nb_q, q.shape[1]))
muscularForce=np.zeros((nb_mus, q.shape[1]))
jacobian = np.zeros((nb_mus, nb_q, q.shape[1]))

# COMPUTE RESULTS
# --- Compute Muscle Force --- #
fiso = get_forceIso()['fiso']
for j in range(nb_mus):
    for i in range(q.shape[1]):
        muscularForce[j, i]=get_muscle_force(q[:, i], qdot[:, i], activations[:, i])[j, :]

# --- Compute Jacobian --- #
for i in range(q.shape[1]):
    jacobian[:, :, i] = get_jacobian(q[:, i])

# --- Compute Individual Torque and Power --- #
for i in range(nb_q):
    for j in range(nb_mus):
        muscularIndividualTorque[j, i, :] = -jacobian[j, i, :] * muscularForce[j, :]
        muscularIndividualPower[j, i, :] = -jacobian[j, i, :] * muscularForce[j, :] * qdot[i, :]

# --- Compute Muscle Torque --- #
for j in range(nb_q):
    for i in range(q.shape[1]):
        muscularTorque[j, i]=get_muscle_torque(q[:, i], qdot[:, i], activations[:, i])[j, :]
        muscularSumTorque[j, i] = np.sum(muscularIndividualTorque[:, j, i])

# PLOT
q_name = get_q_name(model)
muscle_name = get_muscle_name(model)

# # --- Plot Torque --- #
# figure, axes = plt.subplots(3, 4)
# axes = axes.flatten()
# for i in range(nb_q):
#     axes[i].plot(muscularTorque[i, :], color="r", linestyle="-")  # "muscle torque"
#     axes[i].plot(tau_residual[i, :], color="b", linestyle="-")  # "residual torque"
#     axes[i].plot(tau_residual[i, :] + muscularTorque[i, :], color="g", linestyle="-")  # "sum torque"
#     axes[i].set_title(q_name[i])
# plt.legend(["muscle torque", "residual torque", "sum torque"])
#
# # --- Plot torque cuisse post --- #
# figure, axes = plt.subplots(3, 4)
# axes = axes.flatten()
# sum_thigh_post = np.zeros((nb_q, muscularTorque.shape[1]))
# for i in range(nb_q):
#     axes[i].plot(muscularTorque[i, :], color="k", linestyle="--")  # "muscle torque"
#     axes[i].set_title(q_name[i])
#     for j in range(9):
#         axes[i].plot(muscularIndividualTorque[j, i, :])  # muscle cuisse post
#         sum_thigh_post[i, :] += muscularIndividualTorque[j, i, :]
#     axes[i].plot(sum_thigh_post[i, :], color="r", linestyle="--")  # "sum thigh post muscle torque"
# leg = []
# leg.append("muscle torque")
# for j in range(9):
#     leg.append(muscle_name[j])
# leg.append("sum muscle thigh post")
# figure.legend(leg)
#
# # --- Plot Torque Cuisse Ant --- #
# figure, axes = plt.subplots(3, 4)
# axes = axes.flatten()
# sum_thigh_ant = np.zeros((nb_q, muscularTorque.shape[1]))
# for i in range(nb_q):
#     axes[i].plot(muscularTorque[i, :], color="k", linestyle="--")  # "muscle torque"
#     for j in range(9, 13):
#         axes[i].plot(muscularIndividualTorque[j, i, :])  # muscle cuisse ant
#         sum_thigh_ant[i, :] += muscularIndividualTorque[j, i, :]
#     axes[i].plot(sum_thigh_ant[i, :], color="r", linestyle="--")
#     axes[i].set_title(q_name[i])
# leg = []
# leg.append("muscle torque")
# for j in range(9, 13):
#     leg.append(muscle_name[j])
# figure.legend(leg)
# leg.append("sum muscle thigh ant")
#
# # --- Plot Torque Jambe post --- #
# figure, axes = plt.subplots(3, 4)
# axes = axes.flatten()
# sum_shank_post = np.zeros((nb_q, muscularTorque.shape[1]))
# for i in range(nb_q):
#     axes[i].plot(muscularTorque[i, :], color="k", linestyle="--")  # "muscle torque"
#     for j in range(13, 16):
#         axes[i].plot(muscularIndividualTorque[j, i, :])  # muscle jambe post
#         sum_shank_post[i, :] += muscularIndividualTorque[j, i, :]
#     axes[i].plot(sum_shank_post[i, :], color="r", linestyle="--")
#     axes[i].set_title(q_name[i])
# leg = []
# leg.append("muscle torque")
# for j in range(13, 16):
#     leg.append(muscle_name[j])
# leg.append("sum muscle shank post")
# figure.legend(leg)
#
# # --- Plot Torque Jambe Ant --- #
# figure, axes = plt.subplots(3, 4)
# axes = axes.flatten()
# for i in range(nb_q):
#     axes[i].plot(muscularTorque[i, :], color="k", linestyle="--")  # "muscle torque"
#     axes[i].plot(muscularIndividualTorque[16, i, :])  # muscle jambe ant
#     axes[i].set_title(q_name[i])
# leg = []
# leg.append("muscle torque")
# leg.append(muscle_name[16])
# figure.legend(leg)
# plt.show()

# # --- Plot Co Contraction --- #
# fig = plt.figure()
# grid = plt.GridSpec(6, 3)
# ax_act_GM = fig.add_subplot(grid[0, :])
# ax_act_GL = fig.add_subplot(grid[1, :])
# ax_act_SOL = fig.add_subplot(grid[2, :])
# ax_act_TA = fig.add_subplot(grid[3, :])
# ax_tau_tibia = fig.add_subplot(grid[4:, 0])
# ax_tau_talus_Y = fig.add_subplot(grid[4:, 1])
# ax_tau_talus_Z = fig.add_subplot(grid[4:, 2])
#
# # --- plot muscles activation --- #
# ax_act_GM.plot(activations[13, :], color="r")
# ax_act_GM.set_title("Activation Gastocnemien Medial")
# ax_act_GM.set_xlim([0.0, activations.shape[1]])
#
# ax_act_GL.plot(activations[14, :], color="r")
# ax_act_GL.set_title("Activation Gastocnemien Lateral")
# ax_act_GL.set_xlim([0.0, activations.shape[1]])
#
# ax_act_SOL.plot(activations[15, :], color="r")
# ax_act_SOL.set_title("Activation Soleaire")
# ax_act_SOL.set_xlim([0.0, activations.shape[1]])
#
# ax_act_TA.plot(activations[16, :], color="r")
# ax_act_TA.set_title("Activation Tibial Anterieur")
# ax_act_TA.set_xlim([0.0, activations.shape[1]])
#
# # --- plot torque --- #
# leg = []
# ax_tau_tibia.plot(muscularTorque[9, :], color="r", linestyle="--")
# leg.append("muscle torque")
# ax_tau_tibia.plot(tau_residual[9, :], color="b", linestyle="--")
# leg.append("residual torque")
# ax_tau_tibia.plot(muscularTorque[9, :] + tau_residual[9, :], color="k", linestyle="--")
# leg.append("somme torque")
# for j in range(13, nb_mus):
#     ax_tau_tibia.plot(muscularIndividualTorque[j, 9, :])
#     leg.append(muscle_name[j])
# ax_tau_tibia.set_title("Torque flexion genou")
#
# ax_tau_talus_Y.plot(muscularTorque[10, :], color="r", linestyle="--")
# ax_tau_talus_Y.plot(tau_residual[10, :], color="b", linestyle="--")
# ax_tau_talus_Y.plot(muscularTorque[10, :] + tau_residual[10, :], color="k", linestyle="--")
# for j in range(13, nb_mus):
#     ax_tau_talus_Y.plot(muscularIndividualTorque[j, 10, :])
# ax_tau_talus_Y.set_title("Torque inversion/eversion cheville")
#
# ax_tau_talus_Z.plot(muscularTorque[11, :], color="r", linestyle="--")
# ax_tau_talus_Z.plot(tau_residual[11, :], color="b", linestyle="--")
# ax_tau_talus_Z.plot(muscularTorque[11, :] + tau_residual[11, :], color="k", linestyle="--")
# for j in range(13, nb_mus):
#     ax_tau_talus_Z.plot(muscularIndividualTorque[j, 11, :])
# ax_tau_talus_Z.set_title("Torque flexion/extension cheville")
# ax_tau_talus_Z.legend(leg)

# --- plot muscles activation --- #
fig, ax = plt.subplots(4, 1)
ax = ax.flatten()
ax[0].plot(activations[13, :], color="r")
ax[0].set_title("Activation Gastocnemien Medial")
ax[0].set_xlim([0.0, activations.shape[1]])

ax[1].plot(activations[14, :], color="r")
ax[1].set_title("Activation Gastocnemien Lateral")
ax[1].set_xlim([0.0, activations.shape[1]])

ax[2].plot(activations[15, :], color="r")
ax[2].set_title("Activation Soleaire")
ax[2].set_xlim([0.0, activations.shape[1]])

ax[3].plot(activations[16, :], color="r")
ax[3].set_title("Activation Tibial Anterieur")
ax[3].set_xlim([0.0, activations.shape[1]])

# --- plot torque --- #
fig, ax = plt.subplots(3, 1)
ax = ax.flatten()
leg = []
ax[0].plot(muscularTorque[9, :], color="r", linestyle="--")
leg.append("muscle torque")
ax[0].plot(tau_residual[9, :], color="b", linestyle="--")
leg.append("residual torque")
ax[0].plot(muscularTorque[9, :] + tau_residual[9, :], color="k", linestyle="--")
leg.append("somme torque")
for j in range(13, nb_mus):
    ax[0].plot(muscularIndividualTorque[j, 9, :])
    leg.append(muscle_name[j])
ax[0].set_title("Torque flexion genou")
ax[0].set_xlim([0.0, activations.shape[1]])

ax[1].plot(muscularTorque[10, :], color="r", linestyle="--")
ax[1].plot(tau_residual[10, :], color="b", linestyle="--")
ax[1].plot(muscularTorque[10, :] + tau_residual[10, :], color="k", linestyle="--")
for j in range(13, nb_mus):
    ax[1].plot(muscularIndividualTorque[j, 10, :])
ax[1].set_title("Torque inversion/eversion cheville")
ax[1].set_xlim([0.0, activations.shape[1]])

ax[2].plot(muscularTorque[11, :], color="r", linestyle="--")
ax[2].plot(tau_residual[11, :], color="b", linestyle="--")
ax[2].plot(muscularTorque[11, :] + tau_residual[11, :], color="k", linestyle="--")
for j in range(13, nb_mus):
    ax[2].plot(muscularIndividualTorque[j, 11, :])
ax[2].set_title("Torque flexion/extension cheville")
ax[2].set_xlim([0.0, activations.shape[1]])
plt.legend(leg)
plt.show()
