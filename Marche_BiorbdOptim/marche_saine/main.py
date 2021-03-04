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
                                                        save_path='./RES/muscle_driven/Hip_muscle/grf_idx/',
                                                        n_threads=8)
tic = time()
# --- Solve the program --- #
sol = gait_muscle_driven_markers_tracking.solve()
toc = time() - tic

# --- Show results --- #
sol.animate()
sol.graphs()
sol.print()

# --- Save results --- #
save_path = './RES/muscle_driven/No_hip/idx_ant/'
save_results(gait_muscle_driven_markers_tracking.ocp, sol, save_path)

# --- Compare contact position --- #
ocp_hip, sol_hip = gait_muscle_driven_markers_tracking.ocp.load('./RES/muscle_driven/Hip_muscle/idx_ant/cycle.bo')
contact_hip = contact(ocp_hip, sol_hip, muscles=True)
track_hip = tracking(ocp_hip, sol_hip, data, muscles=True)
cop_hip = contact_hip.merged_result(contact_hip.cop)
COP_REF = track_hip.merged_reference(track_hip.cop_ref)
ocp_decal, sol_decal = gait_muscle_driven.ocp.load('./RES/muscle_driven/decal_contact/cycle.bo')
contact_decal = contact(ocp_decal, sol_decal, muscles=True)
cop_decal= contact_decal.merged_result(contact_decal.cop)

# --- plot cop --- #
fig, axes = plt.subplots(2, 1)
axes = axes.flatten()
fig.suptitle('cop position ')
axes[0].set_title("cop X")
axes[0].scatter(COP_REF[0, :53], "k--")
axes[0].scatter(cop_hip["cop_r_X"][:53], "r")
axes[0].scatter(cop_decal["cop_r_X"][:53], "b")
for p in range(nb_phases - 1):
    axes[0].plot([sum(number_shooting_points[:p+1]), sum(number_shooting_points[:p+1])], [min(COP_REF[0, :]), max(COP_REF[0, :])], "k--")

axes[1].set_title("cop Y")
axes[1].scatter(COP_REF[1, :53], "k--")
axes[1].scatter(cop_hip["cop_r_Y"][:53], "r")
axes[1].scatter(cop_decal["cop_r_Y"][:53], "b")
for p in range(nb_phases - 1):
    axes[1].plot([sum(number_shooting_points[:p+1]), sum(number_shooting_points[:p+1])], [min(COP_REF[1, :]), max(COP_REF[1, :])], "k--")
axes[1].legend(["reference", "marker position", "decalage"])

# --- Load previous results --- #
ocp_hip, sol_hip = gait_muscle_driven.ocp.load('./RES/muscle_driven/Hip_muscle/cycle.bo')
muscle_hip = muscle(ocp_hip, sol_hip.merge_phases())
ocp_no_hip, sol_no_hip = gait_muscle_driven.ocp.load('./RES/muscle_driven/No_hip/cycle.bo')
muscle_no_hip = muscle(ocp_no_hip, sol_no_hip)
ocp_no_rf, sol_no_rf = gait_muscle_driven.ocp.load('./RES/muscle_driven/No_RF/cycle.bo')
muscle_no_rf = muscle(ocp_no_rf, sol_no_rf)

# --- plot activations --- #
fig, axes = plt.subplots(4, 5)
axes = axes.flatten()
fig.suptitle('Muscle activations')
for (m, muscle) in enumerate(muscle_hip.muscle_name):
    axes[m].set_title(muscle)
    axes[m].plot(muscle_hip.activations[m, :], "r")
    axes[m].plot(muscle_no_hip.activations[m, :], "b")
    axes[m].plot(muscle_no_rf.activations[m, :], "g")
    axes[m].set_ylim([0.0, 1.0])
    axes[m].set_xlim([0.0, muscle_hip.n_shooting + 1])
    for p in range(nb_phases):
        axes[m].plot([sum(number_shooting_points[:p+1]), sum(number_shooting_points[:p+1])], [0.0, 1.0], "k--")
axes[-2].legend(["iliopsoas", "no iliopsoas", "no rectus femoris"])

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
plot_muscular_torque(muscle_hip, muscle_no_hip, muscle_no_rf, 8, 7, number_shooting_points)
