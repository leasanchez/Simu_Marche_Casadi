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

# gait_muscle_driven_markers_tracking.ocp.print()
tic = time()
# --- Solve the program --- #
sol = gait_muscle_driven_markers_tracking.solve()
toc = time() - tic

# --- Save results --- #
save_path = './RES/muscle_driven/no_last_contact_cstr/'
save_results(gait_muscle_driven_markers_tracking.ocp, sol, save_path)

# ocp_prev, sol = gait_muscle_driven_markers_tracking.ocp.load('./RES/muscle_driven/Hip_muscle/OpenSim/cycle.bo')
# --- Show results --- #
sol.animate()
sol.graphs()
sol.print()