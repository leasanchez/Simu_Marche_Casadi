import numpy as np
from casadi import dot, Function, vertcat, MX, mtimes, nlpsol
import biorbd
import BiorbdViz as BiorbdViz
from matplotlib import pyplot as plt
from Marche_BiorbdOptim.LoadData import Data_to_track

from biorbd_optim import (
    OptimalControlProgram,
    DynamicsTypeList,
    DynamicsType,
    BoundsList,
    Bounds,
    QAndQDotBounds,
    InitialConditionsList,
    InitialConditions,
    ShowResult,
    ObjectiveList,
    Objective,
    InterpolationType,
    Data,
    ParameterList,
    Instant,
    ConstraintList,
    Constraint,
    Solver,
)


# Define the problem --- Model path
model = biorbd.Model("../../ModelesS2M/Marche_saine/2legs/2legs_24dof_0contact.bioMod")
# model =biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_1contact_deGroote_3d_Heel.bioMod")
# Problem parameters
nb_q = model.nbQ()
nb_qdot = model.nbQdot()
nb_tau = model.nbGeneralizedTorque()

# Generate data from file
Data_to_track = Data_to_track("normal02", model=model, multiple_contact=True, two_leg=True)
phase_time = Data_to_track.GetTime()
nb_phases = len(phase_time)
number_shooting_points = [5, 10, 15, 10, 10, 10, 10, 10]
markers_ref = Data_to_track.load_data_markers(number_shooting_points) # get markers positions
q_ref = Data_to_track.load_q_kalman(number_shooting_points) # get joint positions
qdot_ref = Data_to_track.load_qdot_kalman(number_shooting_points) # get joint velocities
qddot_ref = Data_to_track.load_qdot_kalman(number_shooting_points) # get joint accelerations
grf_ref = Data_to_track.load_data_GRF(number_shooting_points)  # get ground reaction forces


b = BiorbdViz.BiorbdViz(loaded_model=model)