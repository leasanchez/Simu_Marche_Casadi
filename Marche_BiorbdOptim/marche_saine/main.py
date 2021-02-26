"""
This is an example on gait biomechanics.
Experimental data (markers trajectories, ground reaction forces and moments) are tracked.
"""
from time import time

import numpy as np
import biorbd
from bioptim import Solver, Shooting

from gait.load_experimental_data import LoadData
from gait.ocp import gait_torque_driven

def get_phase_time_shooting_numbers(data, dt):
    phase_time = data.c3d_data.get_time()
    number_shooting_points = []
    for time in phase_time:
        number_shooting_points.append(int(time / dt) - 1)
    return phase_time, number_shooting_points


def get_experimental_data(data, number_shooting_points):
    q_ref = data.dispatch_data_interpolation(data=data.q, nb_shooting=number_shooting_points)
    qdot_ref = data.dispatch_data_interpolation(data=data.qdot, nb_shooting=number_shooting_points)
    markers_ref = data.dispatch_data_interpolation(data=data.c3d_data.trajectories, nb_shooting=number_shooting_points)
    grf_ref = data.dispatch_data_interpolation(data=data.c3d_data.forces, nb_shooting=number_shooting_points)
    moments_ref = data.dispatch_data_interpolation(data=data.c3d_data.moments, nb_shooting=number_shooting_points)
    cop_ref = data.dispatch_data_interpolation(data=data.c3d_data.cop, nb_shooting=number_shooting_points)
    return q_ref, qdot_ref, markers_ref, grf_ref, moments_ref, cop_ref



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

# Generate data from file
# --- files path ---
c3d_file = "../../DonneesMouvement/normal01_out.c3d"
q_kalman_filter_file = "../../DonneesMouvement/normal01_q_KalmanFilter.txt"
qdot_kalman_filter_file = "../../DonneesMouvement/normal01_qdot_KalmanFilter.txt"
data = LoadData(biorbd_model[0], c3d_file, q_kalman_filter_file, qdot_kalman_filter_file)

# --- phase time and number of shooting ---
phase_time, number_shooting_points = get_phase_time_shooting_numbers(data, 0.01)
# --- get experimental data ---
q_ref, qdot_ref, markers_ref, grf_ref, moments_ref, cop_ref = get_experimental_data(data, number_shooting_points)

gait_torque_driven = gait_torque_driven(biorbd_model,
                                        number_shooting_points,
                                        phase_time,
                                        q_ref,
                                        qdot_ref,
                                        markers_ref,
                                        grf_ref,
                                        moments_ref,
                                        cop_ref,
                                        n_threads=4,
                                        four_contact=True)
tic = time()
# --- Solve the program --- #
sol = gait_torque_driven.solve()
toc = time() - tic

# --- Show results --- #
sol.animate()
sol.graphs()
sol.print()

    # # --- Save results --- #
    # ocp.save(sol, "gait.bo")
