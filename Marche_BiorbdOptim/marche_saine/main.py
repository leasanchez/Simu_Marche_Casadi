"""
This is an example on gait biomechanics.
Experimental data (markers trajectories, ground reaction forces and moments) are tracked.
"""
from time import time

import numpy as np
import biorbd
from matplotlib import pyplot as plt

from gait.load_experimental_data import LoadData
from gait.ocp import gait_torque_driven, gait_muscle_driven


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

gait_muscle_driven = gait_muscle_driven(models=biorbd_model,
                                        nb_shooting=number_shooting_points,
                                        phase_time=phase_time,
                                        q_ref=q_ref,
                                        qdot_ref=qdot_ref,
                                        markers_ref=markers_ref,
                                        grf_ref=grf_ref,
                                        moments_ref=moments_ref,
                                        cop_ref=cop_ref,
                                        n_threads=8)
tic = time()
# --- Solve the program --- #
sol = gait_muscle_driven.solve()
toc = time() - tic

# --- Show results --- #
sol.animate()
sol.graphs()
sol.print()

    # # --- Save results --- #
    # ocp.save(sol, "gait.bo")
