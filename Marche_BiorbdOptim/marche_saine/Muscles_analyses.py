import numpy as np
from casadi import dot, Function, vertcat, MX, DM
import biorbd
import bioviz
from matplotlib import pyplot as plt
from Marche_BiorbdOptim.LoadData import Data_to_track

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

def get_muscle_name(model):
    muscle_name = []
    for m in range(model.nbMuscleTotal()):
        muscle_name.append(model.muscle(m).name().to_string())
    return muscle_name

def get_q_range(model):
    q_max = []
    q_min = []
    for s in range(model.nbSegment()):
        q_range = model.segment(s).QRanges()
        for r in q_range:
            q_max.append(r.max())
            q_min.append(r.min())
    return q_max, q_min

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
    for nGrp in range(biorbd_model[0].nbMuscleGroups()):
        for nMus in range(biorbd_model[0].muscleGroup(nGrp).nbMuscles()):
            fiso.append(biorbd_model[0].muscleGroup(nGrp).muscle(nMus).characteristics().forceIsoMax().to_mx())
    return vertcat(*fiso)

def get_force_iso(model):
    return Function("Fiso", [], [force_isometric(model)], [], ["fiso"]).expand()


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
Data = Data_to_track("normal01", model=biorbd_model[0], multiple_contact=True)
phase_time = Data.GetTime()
number_shooting_points = []
for time in phase_time:
    number_shooting_points.append(int(time / 0.01))
grf_ref = Data.load_data_GRF(number_shooting_points)  # get ground reaction forces
markers_ref = Data.load_data_markers(number_shooting_points)
q_ref = Data.load_q_kalman(number_shooting_points)
qdot_ref = Data.load_qdot_kalman(number_shooting_points)
CoP = Data.load_data_CoP(number_shooting_points)
M_ref = Data.load_data_Moment_at_CoP(number_shooting_points)
EMG_ref = Data.load_data_emg(number_shooting_points)
excitations_ref = []
for p in range(nb_phases):
    excitations_ref.append(Data.load_muscularExcitation(EMG_ref[p]))


Data_2 = Data_to_track("normal02", model=biorbd_model[0], multiple_contact=True)
EMG_ref_2 = Data_2.load_data_emg(number_shooting_points)
excitations_ref_2 = []
for p in range(nb_phases):
    excitations_ref_2.append(Data_2.load_muscularExcitation(EMG_ref[p]))
Data_3 = Data_to_track("normal03", model=biorbd_model[0], multiple_contact=True)
EMG_ref_3 = Data_3.load_data_emg(number_shooting_points)
excitations_ref_3 = []
for p in range(nb_phases):
    excitations_ref_3.append(Data_3.load_muscularExcitation(EMG_ref[p]))
Data_4 = Data_to_track("normal04", model=biorbd_model[0], multiple_contact=True)
EMG_ref_4 = Data_4.load_data_emg(number_shooting_points)
excitations_ref_4 = []
for p in range(nb_phases):
    excitations_ref_4.append(Data_4.load_muscularExcitation(EMG_ref[p]))

Q_ref = np.zeros((nb_q, sum(number_shooting_points) + 1))
Qdot_ref = np.zeros((nb_q, sum(number_shooting_points) + 1))
for i in range(nb_q):
    n_shoot = 0
    for n in range(nb_phases):
        Q_ref[i, n_shoot:n_shoot + number_shooting_points[n] + 1] = q_ref[n][i, :]
        Qdot_ref[i, n_shoot:n_shoot + number_shooting_points[n] + 1] = qdot_ref[n][i, :]
        n_shoot += number_shooting_points[n]

E_ref = np.zeros((nb_mus, sum(number_shooting_points) + 1))
E_ref2 = np.zeros((nb_mus, sum(number_shooting_points) + 1))
E_ref3 = np.zeros((nb_mus, sum(number_shooting_points) + 1))
E_ref4 = np.zeros((nb_mus, sum(number_shooting_points) + 1))
for i in range(nb_mus):
    n_shoot = 0
    for n in range(nb_phases):
        E_ref[i, n_shoot:n_shoot + number_shooting_points[n] + 1] = excitations_ref[n][i, :]
        E_ref2[i, n_shoot:n_shoot + number_shooting_points[n] + 1] = excitations_ref_2[n][i, :]
        E_ref3[i, n_shoot:n_shoot + number_shooting_points[n] + 1] = excitations_ref_3[n][i, :]
        E_ref4[i, n_shoot:n_shoot + number_shooting_points[n] + 1] = excitations_ref_4[n][i, :]
        n_shoot += number_shooting_points[n]



t = get_time_vector(phase_time, number_shooting_points)
q_name = get_q_name(biorbd_model[0])
q_max, q_min = get_q_range(biorbd_model[0])


# LOAD RESULTS
q = np.load('./RES/cycle/muscles/3_contacts/fort/square/q.npy')
q_dot = np.load('./RES/cycle/muscles/3_contacts/fort/square/q_dot.npy')
tau = np.load('./RES/cycle/muscles/3_contacts/fort/square/tau.npy')
tau_inverse = np.load('./RES/cycle/muscles/3_contacts/fort/square/tau_inverse.npy')
tau_ocp_torque = np.load('./RES/1leg/cycle/3points/tau.npy')
activation = np.load('./RES/1leg/cycle/muscles/4_contacts/decal_talon/activation.npy')
activation_SO = np.load('./RES/cycle/muscles/3_contacts/fort/square/activations_SO.npy')

# CASADI FUNCTION
get_muscle_torque = muscular_torque(biorbd_model[0])
get_jacobian = muscles_jac(biorbd_model[0])
get_muscle_force = muscular_force(biorbd_model[0])
get_forceIso = get_force_iso(biorbd_model[0])

# INIT
muscularTorque=np.zeros((nb_q, q.shape[1]))
muscularSumTorque=np.zeros((nb_q, q.shape[1]))
muscularTorque_ref=np.zeros((nb_q, q.shape[1]))
muscularIndividualTorque=np.zeros((nb_mus, nb_q, q.shape[1]))
muscularIndividualPower=np.zeros((nb_mus, nb_q, q.shape[1]))
muscularForce=np.zeros((nb_mus, q.shape[1]))
muscularForce_ref=np.zeros((nb_mus, q.shape[1]))
jacobian = np.zeros((nb_mus, nb_q, q.shape[1]))
jacobian_ref = np.zeros((nb_mus, nb_q, q.shape[1]))

# COMPUTE MUSCLE FORCE
fiso = get_forceIso()['fiso']
for j in range(nb_mus):
    for i in range(q.shape[1]):
        muscularForce[j, i]=get_muscle_force(q[:, i], q_dot[:, i], activation[:, i])[j, :]
        muscularForce_ref[j, i] = get_muscle_force(Q_ref[:, i], Qdot_ref[:, i], E_ref[:, i])[j, :]

# COMPUTE JACOBIAN
for i in range(q.shape[1]):
    jacobian_ref[:, :, i] = get_jacobian(Q_ref[:, i])
    jacobian[:, :, i] = get_jacobian(q[:, i])

# COMPUTE INDIVIDUAL TORQUE AND POWER
for i in range(nb_q):
    for j in range(nb_mus):
        muscularIndividualTorque[j, i, :] = -jacobian[j, i, :] * muscularForce[j, :]
        muscularIndividualPower[j, i, :] = -jacobian[j, i, :] * muscularForce[j, :] * q_dot[i, :]

# COMPUTE MUSCLE TORQUE
for j in range(nb_q):
    for i in range(q.shape[1]):
        muscularTorque[j, i]=get_muscle_torque(q[:, i], q_dot[:, i], activation[:, i])[j, :]
        muscularTorque_ref[j, i] = get_muscle_torque(Q_ref[:, i], Qdot_ref[:, i], E_ref[:, i])[j, :]
        muscularSumTorque[j, i] = np.sum(muscularIndividualTorque[:, j, i])

# PLOT PARAM
t = get_time_vector(phase_time, number_shooting_points)
q_name = get_q_name(biorbd_model[0])
muscle_name = get_muscle_name(biorbd_model[0])

# PLOT Q
figure, axes = plt.subplots(3, 4)
axes = axes.flatten()
for i in range(nb_q):
    axes[i].plot(t, Q_ref[i, :], color="b", linestyle="--")
    axes[i].plot(t, q[i, :], color="r", linestyle="-")
    # plot phase transition
    if (nb_phases > 1):
        pt = 0
        for p in range(nb_phases):
            pt += phase_time[p]
            axes[i].plot([pt, pt], [q_min[i], q_max[i]], color='k', linestyle="--", linewidth=0.7)
    axes[i].set_title(f" Q : {q_name[i]}")
    axes[i].set_ylim([q_min[i], q_max[i]])
figure.legend(['reference', 'simu'])

# PLOT QDOT
figure, axes = plt.subplots(3, 4)
axes = axes.flatten()
for i in range(nb_q):
    axes[i].plot(t, Qdot_ref[i, :], color="b", linestyle="--")
    axes[i].plot(t, q_dot[i, :], color="r", linestyle="-")
    # plot phase transition
    if (nb_phases > 1):
        pt = 0
        for p in range(nb_phases):
            pt += phase_time[p]
            axes[i].plot([pt, pt], [np.min(q_dot[i, :]), np.max(q_dot[i, :])], color='k', linestyle="--", linewidth=0.7)
    axes[i].set_title(f" Qdot : {q_name[i]}")
figure.legend(['reference', 'simu'])

# PLOT ACTIVATION
figure, axes = plt.subplots(3, 6)
axes = axes.flatten()
for i in range(nb_mus):
    plot_control(axes[i], t, activation[i, :], color="r", linestyle="-")
    plot_control(axes[i], t, E_ref[i, :], color="b", linestyle="-")
    plot_control(axes[i], t, E_ref2[i, :], color="b", linestyle="-")
    plot_control(axes[i], t, E_ref3[i, :], color="b", linestyle="-")
    plot_control(axes[i], t, E_ref4[i, :], color="b", linestyle="-")
    if (nb_phases > 1):
        pt = 0
        for p in range(nb_phases):
            pt += phase_time[p]
            axes[i].plot([pt, pt], [np.min(E_ref), np.max(E_ref)], color='k', linestyle="--", linewidth=0.7)
    axes[i].set_title(biorbd_model[0].muscle(i).name().to_string())
    axes[i].set_ylim([0.0, 1.01])
plt.show()

# PLOT TORQUE CUISSE POST
figure, axes = plt.subplots(3, 4)
axes = axes.flatten()
for i in range(nb_q):
    axes[i].plot(t, muscularTorque[i, :], color="r", linestyle="--")  # "muscle torque"
    axes[i].plot(t, tau[i, :], color="b", linestyle="--")  # "residual torque"
    axes[i].plot(t, tau_inverse[i, :], color="k", linestyle="--")  # "inverse dynamics"
    axes[i].plot(t, tau_ocp_torque[i, :], color="m", linestyle="--")  # "torque driven"
    for j in range(9):
        axes[i].plot(t, muscularIndividualTorque[j, i, :])  # muscle cuisse post
    # plot phase transition
    if (nb_phases > 1):
        pt = 0
        for p in range(nb_phases):
            pt += phase_time[p]
            axes[i].plot([pt, pt], [min(tau[i, :]), max(tau[i, :])], color='k', linestyle="--", linewidth=0.7)
    axes[i].set_title(q_name[i])
leg = []
leg.append("muscle torque")
leg.append("residual torque")
leg.append("inverse dynamics")
leg.append("torque driven")
for j in range(9):
    leg.append(muscle_name[j])
figure.legend(leg)

# PLOT TORQUE CUISSE ANT
figure, axes = plt.subplots(3, 4)
axes = axes.flatten()
for i in range(nb_q):
    axes[i].plot(t, muscularTorque[i, :], color="r", linestyle="--")  # "muscle torque"
    axes[i].plot(t, tau[i, :], color="b", linestyle="--")  # "residual torque"
    axes[i].plot(t, tau_inverse[i, :], color="k", linestyle="--")  # "inverse dynamics"
    axes[i].plot(t, tau_ocp_torque[i, :], color="m", linestyle="--")  # "torque driven"
    for j in range(9, 13):
        axes[i].plot(t, muscularIndividualTorque[j, i, :])  # muscle cuisse ant
    # plot phase transition
    if (nb_phases > 1):
        pt = 0
        for p in range(nb_phases):
            pt += phase_time[p]
            axes[i].plot([pt, pt], [min(tau[i, :]), max(tau[i, :])], color='k', linestyle="--", linewidth=0.7)
    axes[i].set_title(q_name[i])
leg = []
leg.append("muscle torque")
leg.append("residual torque")
leg.append("inverse dynamics")
leg.append("torque driven")
for j in range(9, 13):
    leg.append(muscle_name[j])
figure.legend(leg)

# PLOT TORQUE JAMBE POST
figure, axes = plt.subplots(3, 4)
axes = axes.flatten()
for i in range(nb_q):
    axes[i].plot(t, muscularTorque[i, :], color="r", linestyle="--")  # "muscle torque"
    axes[i].plot(t, tau[i, :], color="b", linestyle="--")  # "residual torque"
    axes[i].plot(t, tau_inverse[i, :], color="k", linestyle="--")  # "inverse dynamics"
    axes[i].plot(t, tau_ocp_torque[i, :], color="m", linestyle="--")  # "torque driven"
    for j in range(13, 16):
        axes[i].plot(t, muscularIndividualTorque[j, i, :])  # muscle jambe post
    # plot phase transition
    if (nb_phases > 1):
        pt = 0
        for p in range(nb_phases):
            pt += phase_time[p]
            axes[i].plot([pt, pt], [min(tau[i, :]), max(tau[i, :])], color='k', linestyle="--", linewidth=0.7)
    axes[i].set_title(q_name[i])
leg = []
leg.append("muscle torque")
leg.append("residual torque")
leg.append("inverse dynamics")
leg.append("torque driven")
for j in range(13, 16):
    leg.append(muscle_name[j])
figure.legend(leg)

# PLOT TORQUE JAMBE ANT
figure, axes = plt.subplots(3, 4)
axes = axes.flatten()
for i in range(nb_q):
    axes[i].plot(t, muscularTorque[i, :], color="r", linestyle="--")  # "muscle torque"
    axes[i].plot(t, tau[i, :], color="b", linestyle="--")  # "residual torque"
    axes[i].plot(t, tau_inverse[i, :], color="k", linestyle="--")  # "inverse dynamics"
    axes[i].plot(t, tau_ocp_torque[i, :], color="m", linestyle="--")  # "torque driven"
    axes[i].plot(t, muscularIndividualTorque[16, i, :])  # muscle jambe ant
    # plot phase transition
    if (nb_phases > 1):
        pt = 0
        for p in range(nb_phases):
            pt += phase_time[p]
            axes[i].plot([pt, pt], [min(tau[i, :]), max(tau[i, :])], color='k', linestyle="--", linewidth=0.7)
    axes[i].set_title(q_name[i])
leg = []
leg.append("muscle torque")
leg.append("residual torque")
leg.append("inverse dynamics")
leg.append("torque driven")
leg.append(muscle_name[16])
figure.legend(leg)
plt.show()

# PLOT POWER CUISSE POST
figure, axes = plt.subplots(3, 4)
axes = axes.flatten()
for i in range(nb_q):
    for j in range(9):
        axes[i].plot(t, muscularIndividualPower[j, i, :])  # muscle cuisse post
    # plot phase transition
    if (nb_phases > 1):
        pt = 0
        for p in range(nb_phases):
            pt += phase_time[p]
            axes[i].plot([pt, pt],
                         [min(muscularIndividualPower[8, i, :]), max(muscularIndividualPower[8, i, :])],
                         color='k', linestyle="--", linewidth=0.7)
    axes[i].set_title(q_name[i])
figure.legend(muscle_name[:9])

# PLOT POWER CUISSE ANT
figure, axes = plt.subplots(3, 4)
axes = axes.flatten()
for i in range(nb_q):
    for j in range(9, 13):
        axes[i].plot(t, muscularIndividualPower[j, i, :])  # muscle cuisse ant
    # plot phase transition
    if (nb_phases > 1):
        pt = 0
        for p in range(nb_phases):
            pt += phase_time[p]
            axes[i].plot([pt, pt],
                         [min(muscularIndividualPower[9, i, :]), max(muscularIndividualPower[9, i, :])],
                         color='k', linestyle="--", linewidth=0.7)
    axes[i].set_title(q_name[i])
figure.legend(muscle_name[9:13])

# PLOT TORQUE JAMBE POST
figure, axes = plt.subplots(3, 4)
axes = axes.flatten()
for i in range(nb_q):
    for j in range(13, 16):
        axes[i].plot(t, muscularIndividualPower[j, i, :])  # muscle jambe post
    # plot phase transition
    if (nb_phases > 1):
        pt = 0
        for p in range(nb_phases):
            pt += phase_time[p]
            axes[i].plot([pt, pt],
                         [min(muscularIndividualPower[15, i, :]), max(muscularIndividualPower[15, i, :])],
                         color='k', linestyle="--", linewidth=0.7)
    axes[i].set_title(q_name[i])
figure.legend(muscle_name[13:16])

# PLOT TORQUE JAMBE ANT
figure, axes = plt.subplots(3, 4)
axes = axes.flatten()
for i in range(nb_q):
    axes[i].plot(t, muscularIndividualTorque[16, i, :])  # muscle jambe ant
    # plot phase transition
    if (nb_phases > 1):
        pt = 0
        for p in range(nb_phases):
            pt += phase_time[p]
            axes[i].plot([pt, pt],
                         [min(muscularIndividualTorque[16, i, :]), max(muscularIndividualTorque[16, i, :])],
                         color='k', linestyle="--", linewidth=0.7)
    axes[i].set_title(q_name[i])
figure.legend([muscle_name[16]])
plt.show()