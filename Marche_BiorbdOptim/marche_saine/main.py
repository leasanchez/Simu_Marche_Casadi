"""
This is an example on gait biomechanics.
Experimental data (markers trajectories, ground reaction forces and moments) are tracked.
"""
from time import time

import numpy as np
import biorbd
from casadi import MX, Function
from matplotlib import pyplot as plt

from gait.load_experimental_data import LoadData
from gait.ocp import gait_muscle_driven
from gait.muscle_functions import muscle
from gait.compute_tracking_functions import tracking
from gait.contact_forces_function import contact

def contact_position_func_casadi(model):
    symbolic_q = MX.sym("q", model.nbQ(), 1)
    nb_contact = model.nbContacts()
    contact_pos_func = []
    for c in range(nb_contact):
        contact_pos_func.append(Function(
                "ForwardKin_contact",
                [symbolic_q], [model.constraintsInGlobal(symbolic_q, True)[c].to_mx()],
                ["q"],
                ["contact_pos"]).expand())
    return contact_pos_func

def merged_reference(x, number_shooting_points):
    x_merged = np.empty((x[0].shape[0], x[0].shape[1], sum(number_shooting_points) + 1))
    for i in range(x[0].shape[1]):
        n_shoot = 0
        for phase in range(len(x)):
            x_merged[:, i, n_shoot:n_shoot + number_shooting_points[phase] + 1] = x[phase][:, i, :]
            n_shoot += number_shooting_points[phase]
    return x_merged

def get_q_name(model):
    q_name = []
    for s in range(model.nbSegment()):
        seg_name = model.segment(s).name().to_string()
        for d in range(model.segment(s).nbDof()):
            dof_name = model.segment(s).nameDof(d).to_string()
            q_name.append(seg_name + "_" + dof_name)
    return q_name

def get_results(sol):
    q = sol.states["q"]
    qdot = sol.states["qdot"]
    tau = sol.controls["tau"]
    muscle = sol.controls["muscles"]
    return q, qdot, tau, muscle

def save_results(ocp, sol, save_path):
    ocp.save(sol, save_path + 'cycle.bo')
    sol_merged = sol.merge_phases()
    q, qdot, tau, muscle = get_results(sol_merged)
    np.save(save_path + 'qdot', qdot)
    np.save(save_path + 'q', q)
    np.save(save_path + 'tau', tau)
    np.save(save_path + 'muscle', muscle)

def plot_muscular_torque(muscle_hip, muscle_no_hip, muscle_no_rf, idx_q, idx_muscle, number_shooting_points):
    q_name = get_q_name(muscle_no_hip.model)
    fig = plt.figure()
    fig.suptitle(f"{muscle_hip.muscle_name[idx_muscle]} - - {q_name[idx_q]}")
    plt.plot(muscle_hip.individual_muscle_torque[idx_muscle, idx_q, :], "r")
    plt.plot(muscle_no_hip.individual_muscle_torque[idx_muscle, idx_q, :], "b")
    plt.plot(muscle_no_rf.individual_muscle_torque[idx_muscle, idx_q, :], "g")
    plt.xlim([0.0, muscle_hip.n_shooting + 1])
    for p in range(nb_phases):
        plt.plot([sum(number_shooting_points[:p + 1]), sum(number_shooting_points[:p + 1])],
                     [min(muscle_hip.individual_muscle_torque[idx_muscle, idx_q, :]),
                      max(muscle_hip.individual_muscle_torque[idx_muscle, idx_q, :])], "k--")
    plt.legend(["iliopsoas", "no iliopsoas", "no rectus femoris"])


def compute_muscle_jacobian(model, q):
    muscle_jacobian = np.zeros((model.nbMuscleTotal(), model.nbQ(), q.shape[1]))
    muscle_jacobian_func = biorbd.to_casadi_func("momentarm", model.musclesLengthJacobian, MX.sym("q", nb_q, 1))
    for n in range(q.shape[1]):
        muscle_jacobian[:, :, n] = muscle_jacobian_func(q[:, n])
    return muscle_jacobian

def plot_moment_arm(model, q, idx_q, idx_muscle, phase_time):
    muscle_jacobian = compute_muscle_jacobian(model, q)
    fig_1 = plt.figure()
    fig_1.suptitle(f"Bras de levier {model.muscle(idx_muscle).name().to_string()} en fonction du temps")
    t = np.linspace(0, q.shape[1]*0.01, q.shape[1])
    plt.plot(t, -muscle_jacobian[idx_muscle, idx_q, :])
    plt.xlim([0.0, q.shape[1]*0.01])
    plt.xlabel('time (sec)')
    plt.ylabel('bras de levier (m)')
    for p in range(len(phase_time)):
        plt.plot([sum(phase_time[:p + 1]), sum(phase_time[:p + 1])],
                 [min(-muscle_jacobian[idx_muscle, idx_q, :]), max(-muscle_jacobian[idx_muscle, idx_q, :])],
                 "k--")


    q_name = get_q_name(model)
    fig_2 = plt.figure()
    fig_2.suptitle(f"Bras de levier {model.muscle(idx_muscle).name().to_string()} en fonction de {q_name[idx_q]}")
    plt.plot(q[idx_q, :]*180/np.pi, -muscle_jacobian[idx_muscle, idx_q, :])
    plt.xlim([min(q[idx_q, :]*180/np.pi), max(q[idx_q, :]*180/np.pi)])
    plt.xlabel(f"{q_name[idx_q]} (deg)")
    plt.ylabel('bras de levier (m)')

    fig_3 = plt.figure()
    fig_3.suptitle(f"Trajectoire {q_name[idx_q]} en fonction du temps ")
    plt.plot(t, q[idx_q, :]*180/np.pi)
    plt.xlim([0.0, q.shape[1]*0.01])
    plt.xlabel('time (sec)')
    plt.ylim([min(q[idx_q, :]*180/np.pi), max(q[idx_q, :]*180/np.pi)])
    plt.ylabel(f"{q_name[idx_q]} (deg)")
    for p in range(len(phase_time)):
        plt.plot([sum(phase_time[:p + 1]), sum(phase_time[:p + 1])],
                 [min(q[idx_q, :]*180/np.pi), max(q[idx_q, :]*180/np.pi)],
                 "k--")



# Define the problem -- model path
biorbd_model = (
    biorbd.Model("models/Gait_1leg_12dof_heel.bioMod"),
    biorbd.Model("models/Gait_1leg_12dof_flatfoot.bioMod"),
    biorbd.Model("models/Gait_1leg_12dof_forefoot.bioMod"),
    biorbd.Model("models/Gait_1leg_12dof_0contact.bioMod"),
)

# Problem parameters
nb_q = biorbd_model[0].nbQ()
nb_qdot = biorbd_model[0].nbQdot()
nb_tau = biorbd_model[0].nbGeneralizedTorque()
nb_phases = len(biorbd_model)
nb_markers = biorbd_model[0].nbMarkers()
nb_mus = biorbd_model[0].nbMuscleTotal()

# Generate data from file
# --- files path ---
c3d_file = "../../DonneesMouvement/normal01_out.c3d"
q_kalman_filter_file = "../../DonneesMouvement/normal01_q_KalmanFilter.txt"
qdot_kalman_filter_file = "../../DonneesMouvement/normal01_qdot_KalmanFilter.txt"
data = LoadData(biorbd_model[0], c3d_file, q_kalman_filter_file, qdot_kalman_filter_file, 0.01, interpolation=True)

# --- phase time and number of shooting ---
phase_time=data.phase_time
number_shooting_points = data.number_shooting_points

# --- get experimental data ---
q_ref = data.q_ref
qdot_ref = data.qdot_ref
markers_ref = data.markers_ref
grf_ref = data.grf_ref
moments_ref = data.moments_ref
cop_ref = data.cop_ref

gait_muscle_driven_markers_tracking = gait_muscle_driven(models=biorbd_model,
                                                        nb_shooting=number_shooting_points,
                                                        phase_time=phase_time,
                                                        q_ref=q_ref,
                                                        qdot_ref=qdot_ref,
                                                        markers_ref=markers_ref,
                                                        grf_ref=grf_ref,
                                                        moments_ref=moments_ref,
                                                        cop_ref=cop_ref,
                                                        n_threads=8)
# tic = time()
# # --- Solve the program --- #
# sol = gait_muscle_driven_markers_tracking.solve()
# toc = time() - tic
#
# # --- Save results --- #
# save_path = './RES/muscle_driven/Hip_muscle/OpenSim/'
# save_results(gait_muscle_driven_markers_tracking.ocp, sol, save_path)

ocp_prev, sol = gait_muscle_driven_markers_tracking.ocp.load('./RES/muscle_driven/Hip_muscle/OpenSim/cycle.bo')
# --- Show results --- #
sol.animate()
sol.graphs()
sol.print()

contact_result = contact(gait_muscle_driven_markers_tracking.ocp, sol, muscles=True)
muscle_result = muscle(gait_muscle_driven_markers_tracking.ocp, sol.merge_phases())
tracking_result = tracking(gait_muscle_driven_markers_tracking.ocp, sol, data, muscles=True)
grf_merged = tracking_result.merged_reference(grf_ref)
forces_sim_merged = tracking_result.contact.merged_result(contact_result.forces)

# --- Plot markers --- #
sol_merged = sol.merge_phases()
q = sol_merged.states["q"]
markers = biorbd.to_casadi_func("markers", biorbd_model[0].markers, MX.sym("q", nb_q, 1))
markers_pos = np.zeros((3, nb_markers, q.shape[1]))
for n in range(q.shape[1]):
    markers_pos[:, :, n] = markers(q[:, n:n+1])

Mark_ref = merged_reference(markers_ref, number_shooting_points)
diff_marker_tot = []
for m in range(26):
    x = np.mean(np.sqrt((Mark_ref[0, m, :] - markers_pos[0, m, :]) ** 2))
    y = np.mean(np.sqrt((Mark_ref[1, m, :] - markers_pos[1, m, :]) ** 2))
    z = np.mean(np.sqrt((Mark_ref[2, m, :] - markers_pos[2, m, :]) ** 2))
    diff_marker_tot.append(np.mean([x, y, z]))

err_force = tracking_result.compute_error_force_tracking()
t = np.linspace(0, gait_muscle_driven_markers_tracking.ocp.nlp[0].tf, gait_muscle_driven_markers_tracking.ocp.nlp[0].ns + 1)
for p in range(1, nb_phases):
    t = np.concatenate((t[:-1], t[-1] + np.linspace(0, gait_muscle_driven_markers_tracking.ocp.nlp[p].tf, gait_muscle_driven_markers_tracking.ocp.nlp[p].ns + 1)))
plt.figure()
plt.plot(t, grf_merged[0, :], 'k')
plt.plot(t, forces_sim_merged["forces_r_X"], 'b')
plt.plot(t, grf_merged[1, :], 'k')
plt.plot(t, forces_sim_merged["forces_r_Y"], 'g')
plt.plot(t, grf_merged[2, :], 'k')
plt.plot(t, forces_sim_merged["forces_r_Z"], 'r')
pt = 0
for p in range(nb_phases):
    pt += gait_muscle_driven_markers_tracking.ocp.nlp[p].tf
    plt.plot([pt, pt],[-200, 840],'k--')

err_markers = tracking_result.compute_error_markers_tracking()
err_pelvis = np.mean([diff_marker_tot[0], diff_marker_tot[1], diff_marker_tot[2], diff_marker_tot[3]]) * 1e3 # passage en mm
err_anat = np.mean([diff_marker_tot[4], diff_marker_tot[9], diff_marker_tot[10], diff_marker_tot[11], diff_marker_tot[12], diff_marker_tot[17], diff_marker_tot[18]]) * 1e3
err_tissus = np.mean([diff_marker_tot[5], diff_marker_tot[6], diff_marker_tot[7], diff_marker_tot[8], diff_marker_tot[13], diff_marker_tot[14], diff_marker_tot[15], diff_marker_tot[16]]) * 1e3
err_pied = np.mean([diff_marker_tot[19:]]) * 1e3

plt.figure()
label_markers = ["pelvis", "anatomique", "tissus", "pied"]
err_plot = [err_pelvis, err_anat, err_tissus, err_pied]
x = np.arange(len(label_markers))
plt.bar(x, err_plot, color='tab:blue', alpha=0.8)
plt.xticks(x, labels=label_markers)

# # --- Compare contact position --- #
# ocp_hip, sol_hip = gait_muscle_driven_markers_tracking.ocp.load('./RES/muscle_driven/Hip_muscle/idx_ant/cycle.bo')
# contact_hip = contact(ocp_hip, sol_hip, muscles=True)
# track_hip = tracking(ocp_hip, sol_hip, data, muscles=True)
# cop_hip = contact_hip.merged_result(contact_hip.cop)
# COP_REF = track_hip.merged_reference(track_hip.cop_ref)
# ocp_decal, sol_decal = gait_muscle_driven.ocp.load('./RES/muscle_driven/decal_contact/cycle.bo')
# contact_decal = contact(ocp_decal, sol_decal, muscles=True)
# cop_decal= contact_decal.merged_result(contact_decal.cop)
#
# # --- plot cop --- #
# fig, axes = plt.subplots(2, 1)
# axes = axes.flatten()
# fig.suptitle('cop position ')
# axes[0].set_title("cop X")
# axes[0].scatter(COP_REF[0, :53], "k--")
# axes[0].scatter(cop_hip["cop_r_X"][:53], "r")
# axes[0].scatter(cop_decal["cop_r_X"][:53], "b")
# for p in range(nb_phases - 1):
#     axes[0].plot([sum(number_shooting_points[:p+1]), sum(number_shooting_points[:p+1])], [min(COP_REF[0, :]), max(COP_REF[0, :])], "k--")
#
# axes[1].set_title("cop Y")
# axes[1].scatter(COP_REF[1, :53], "k--")
# axes[1].scatter(cop_hip["cop_r_Y"][:53], "r")
# axes[1].scatter(cop_decal["cop_r_Y"][:53], "b")
# for p in range(nb_phases - 1):
#     axes[1].plot([sum(number_shooting_points[:p+1]), sum(number_shooting_points[:p+1])], [min(COP_REF[1, :]), max(COP_REF[1, :])], "k--")
# axes[1].legend(["reference", "marker position", "decalage"])

# # --- Load previous results --- #
# ocp_hip, sol_hip = gait_muscle_driven_markers_tracking.ocp.load('./RES/muscle_driven/Hip_muscle/cycle.bo')
# muscle_hip = muscle(ocp_hip, sol_hip.merge_phases())
# ocp_hip_ant, sol_hip_ant = gait_muscle_driven_markers_tracking.ocp.load('./RES/muscle_driven/Hip_muscle/idx_ant/cycle.bo')
# muscle_hip_ant = muscle(ocp_hip_ant, sol_hip_ant.merge_phases())

q_hip = np.load('./RES/muscle_driven/Hip_muscle/q.npy')
qdot_hip = np.load('./RES/muscle_driven/Hip_muscle/qdot.npy')
tau_hip = np.load('./RES/muscle_driven/Hip_muscle/tau.npy')
activations_hip = np.load('./RES/muscle_driven/Hip_muscle/muscle.npy')

# --- Plot markers --- #
markers = biorbd.to_casadi_func("markers", biorbd_model[0].markers, MX.sym("q", nb_q, 1))
markers_pos = np.zeros((3, nb_markers, q_hip.shape[1]))
for n in range(q_hip.shape[1]):
    markers_pos[:, :, n] = markers(q_hip[:, n:n+1])

Mark_ref = merged_reference(markers_ref, number_shooting_points)





plot_moment_arm(biorbd_model[0], q_hip, idx_q=9, idx_muscle=11, phase_time=phase_time)
activations_hip_ant = np.load('./RES/muscle_driven/Hip_muscle/idx_ant/muscle.npy')
# --- plot activations --- #
fig, axes = plt.subplots(4, 5)
axes = axes.flatten()
fig.suptitle('Muscle activations')
for m in range(nb_mus):
    axes[m].set_title(biorbd_model[0].muscle(m).name().to_string())
    axes[m].plot(activations_hip[m, :], "r")
    axes[m].plot(activations_hip_ant[m, :], "b")
    axes[m].set_ylim([0.0, 1.0])
    axes[m].set_xlim([0.0, activations_hip.shape[1]])
    for p in range(nb_phases):
        axes[m].plot([sum(number_shooting_points[:p+1]), sum(number_shooting_points[:p+1])], [0.0, 1.0], "k--")
axes[-2].legend(["iliopsoas", "iliopsoas decalage ant"])

# # --- Load previous results --- #
# ocp_hip, sol_hip = gait_muscle_driven_markers_tracking.ocp.load('./RES/muscle_driven/Hip_muscle/idx_ant/cycle.bo')
# muscle_hip = muscle(ocp_hip, sol_hip.merge_phases())
# ocp_no_hip, sol_no_hip = gait_muscle_driven_markers_tracking.ocp.load('./RES/muscle_driven/No_hip/idx_ant/cycle.bo')
# muscle_no_hip = muscle(ocp_no_hip, sol_no_hip.merge_phases())
# ocp_no_rf, sol_no_rf = gait_muscle_driven_markers_tracking.ocp.load('./RES/muscle_driven/No_RF/idx_ant/cycle.bo')
# muscle_no_rf = muscle(ocp_no_rf, sol_no_rf.merge_phases())
#
# # --- plot activations --- #
# fig, axes = plt.subplots(4, 5)
# axes = axes.flatten()
# fig.suptitle('Muscle activations')
# for (m, muscle) in enumerate(muscle_hip.muscle_name):
#     axes[m].set_title(muscle)
#     axes[m].plot(muscle_hip.activations[m, :], "r")
#     axes[m].plot(muscle_no_hip.activations[m, :], "b")
#     axes[m].plot(muscle_no_rf.activations[m, :], "g")
#     axes[m].set_ylim([0.0, 1.0])
#     axes[m].set_xlim([0.0, muscle_hip.n_shooting + 1])
#     for p in range(nb_phases):
#         axes[m].plot([sum(number_shooting_points[:p+1]), sum(number_shooting_points[:p+1])], [0.0, 1.0], "k--")
# axes[-2].legend(["iliopsoas", "no iliopsoas", "no rectus femoris"])

# --- plot muscle forces --- #
fig, axes = plt.subplots(4, 5)
axes = axes.flatten()
fig.suptitle('Muscle forces')
for (m, muscle) in enumerate(muscle_hip.muscle_name):
    axes[m].set_title(muscle)
    axes[m].plot(muscle_hip.muscle_force[m, :], "r")
    axes[m].plot(muscle_no_hip.muscle_force[m, :], "b")
    axes[m].plot(muscle_no_rf.muscle_force[m, :], "g")
    axes[m].set_ylim([0.0, max(muscle_hip.muscle_force[m, :])])
    axes[m].set_xlim([0.0, muscle_hip.n_shooting + 1])
    for p in range(nb_phases):
        axes[m].plot([sum(number_shooting_points[:p+1]), sum(number_shooting_points[:p+1])], [0.0, max(muscle_hip.muscle_force[m, :])], "k--")
axes[-2].legend(["iliopsoas", "no iliopsoas", "no rectus femoris"])

contact_pos_func = contact_position_func_casadi(biorbd_model[0])
# --- plot muscle torque --- #
plot_muscular_torque(muscle_hip, muscle_no_hip, muscle_no_rf, 9, 15, number_shooting_points)
