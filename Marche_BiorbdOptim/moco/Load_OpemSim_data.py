import pandas as pd
import numpy as np
from scipy import interpolate
from BiorbdViz import BiorbdViz
import biorbd
from matplotlib import pyplot as plt

df =pd.read_excel("coordinates_test.xlsx")
df = np.array(df)
q_pelvis = df[366:1207, 1:7].T
q_rigth_leg = df[366:1207, 7:13].T
q_left_leg = df[366:1207, 15:21].T

final_time = 0.84
nb_shooting = 50
t = np.linspace(0, final_time, len(df[366:1207, 0]))
node_t = np.linspace(0, final_time, nb_shooting + 1)
f_pelvis = interpolate.interp1d(t, q_pelvis, kind="cubic")
q_pelvis = f_pelvis(node_t)
f_rigth_leg = interpolate.interp1d(t, q_rigth_leg, kind="cubic")
q_rigth_leg = f_rigth_leg(node_t)
f_left_leg = interpolate.interp1d(t, q_left_leg, kind="cubic")
q_left_leg = f_left_leg(node_t)

model = biorbd.Model("../../ModelesS2M/Open_Sim/subject_walk_armless_test_no_muscle.bioMod")

# Problem parameters
nb_q = model.nbQ()
nb_qdot = model.nbQdot()
nb_mus = model.nbMuscleTotal()
nb_tau = model.nbGeneralizedTorque()
nb_markers = model.nbMarkers()

q = np.zeros((nb_q, nb_shooting + 1))
q[:3, :]=q_pelvis[3:, :]  # pelvis translation
q[3:6, :]=q_pelvis[:3, :]*np.pi/180 # pelvis rotation
q[6:9, :]=q_rigth_leg[:3, :]*np.pi/180 #hip r
# q[9:11, :]=fcn -- ok
q[11, :]=q_rigth_leg[3, :]*np.pi/180 #knee r
# q[12:14, :]=fcn -- ok
# q[14:17, :]=fcn -- ok
q[17, :]=q_rigth_leg[5, :]*np.pi/180 #ankle r
q[18:21, :]=q_left_leg[:3, :]*np.pi/180 #hip l
# q[21:23, :]=fcn -- ok
q[23, :]=-q_left_leg[3, :]*np.pi/180 #knee l
# q[24:26, :]=fcn -- ok
# q[26:29, :]=fcn
q[29, :]=q_rigth_leg[5, :]*np.pi/180 # ankle l


# dof axis: walker_knee_r at translation1 on 0 1 0 q[9, :]=fcn
# dpdt knee_angle_r
x = np.array([0, 0.174533, 0.349066, 0.523599, 0.698132, 0.872665, 1.0472, 1.22173, 1.39626, 1.5708, 1.74533, 1.91986, 2.0944])
y = np.array([0, 0.000479, 0.000835, 0.001086, 0.001251, 0.001346, 0.001391, 0.001403, 0.0014, 0.0014, 0.001421, 0.001481, 0.001599])
f_ty_knee=interpolate.interp1d(x, y, kind="cubic")
q[9, :]=0.95799999999999996*f_ty_knee(q[11, :])

# knee_angle_r
# dof axis: walker_knee_r at translation2 on 0 0 1
y=np.array([0, 0.000988, 0.001899, 0.002734, 0.003492, 0.004173, 0.004777, 0.005305, 0.005756, 0.00613, 0.006427, 0.006648, 0.006792])
f_tz_knee=interpolate.interp1d(x, y, kind="cubic")
q[10, :]=0.95799999999999996*f_tz_knee(q[11, :])

#dof axis: walker_knee_r at rotation2 on 0 0 1
y=np.array([0, 0.0126809, 0.0226969, 0.0296054, 0.0332049, 0.0335354, 0.0308779, 0.0257548, 0.0189295, 0.011407, 0.00443314, -0.00050475, -0.0016782])
f_rz_knee=interpolate.interp1d(x, y, kind="cubic")
q[12, :]=f_rz_knee(q[11, :])

#dof axis: walker_knee_r at rotation2 on 0 0 1
y=np.array([0, 0.059461, 0.109399, 0.150618, 0.18392, 0.210107, 0.229983, 0.24435, 0.254012, 0.25977, 0.262428, 0.262788, 0.261654])
f_ry_knee=interpolate.interp1d(x, y, kind="cubic")
q[13, :]=f_ry_knee(q[11, :])

# patella -- knee_angle_r_beta
knee_angle_r_beta=q_rigth_leg[4, :]

# dof axis: patellofemoral_r at translation1 on 1 0 0
y=np.array([0.0524, 0.0488, 0.0437, 0.0371, 0.0296, 0.0216, 0.0136, 0.0057, -0.0019, -0.0088, -0.0148, -0.0196, -0.0227])
f_tx_patella=interpolate.interp1d(x, y, kind="cubic")
q[14, :]=0.95799999999999996*f_tx_patella(knee_angle_r_beta)

# dof axis: patellofemoral_r at translation2 on 0 1 0
y=np.array([-0.0108, -0.019, -0.0263, -0.0322, -0.0367, -0.0395, -0.0408, -0.0404, -0.0384, -0.0349, -0.0301, -0.0245, -0.0187])
f_ty_patella=interpolate.interp1d(x, y, kind="cubic")
q[15, :]=0.95799999999999996*f_ty_patella(knee_angle_r_beta)

# dof axis: patellofemoral_r at rotation1 on 0 0 1
y=np.array([0.00113686, -0.00629212, -0.105582, -0.253683, -0.414245, -0.579047, -0.747244, -0.91799, -1.09044, -1.26379, -1.43763, -1.61186, -1.78634])
f_rz_patella=interpolate.interp1d(x, y, kind="cubic")
q[16, :]=f_rz_patella(knee_angle_r_beta)

# knee_angle_l
# dof axis: walker_knee_l at translation1 on 0 1 0
y=np.array([0, 0.000479, 0.000835, 0.001086, 0.001251, 0.001346, 0.001391, 0.001403, 0.0014, 0.0014, 0.001421, 0.001481, 0.001599])
f_ty_knee_l=interpolate.interp1d(x, y, kind="cubic")
q[21, :]=-0.95799999999999996*f_ty_knee_l(-q[23, :])

# dof axis: walker_knee_l at translation2 on 0 0 1
y=np.array([0, -0.000988, -0.001899, -0.002734, -0.003492, -0.004173, -0.004777, -0.005305, -0.005756, -0.00613, -0.006427, -0.006648, -0.006792])
f_tz_knee_l=interpolate.interp1d(x, y, kind="cubic")
q[22, :]=-0.95799999999999996*f_tz_knee_l(-q[23, :])

# dof axis: walker_knee_l at rotation2 on 0 0 1
y=np.array([0, 0.0126809, 0.0226969, 0.0296054, 0.0332049, 0.0335354, 0.0308779, 0.0257548, 0.0189295, 0.011407, 0.00443314, -0.00050475, -0.0016782])
f_rz_knee_l=interpolate.interp1d(x, y, kind="cubic")
q[24, :]=-f_rz_knee_l(-q[23, :])

# dof axis: walker_knee_l at rotation3 on 0 1 0
y=np.array([0, -0.059461, -0.109399, -0.150618, -0.18392, -0.210107, -0.229983, -0.24435, -0.254012, -0.25977, -0.262428, -0.262788, -0.261654])
f_ry_knee_l=interpolate.interp1d(x, y, kind="cubic")
q[25, :]=-f_ry_knee_l(-q[23, :])

# q[26:29, :]=fcn
# patella -- knee_angle_l_beta
knee_angle_l_beta=q_left_leg[4, :]

# dof axis: patellofemoral_l at translation1 on 1 0 0
y=np.array([0.0524, 0.0488, 0.0437, 0.0371, 0.0296, 0.0216, 0.0136, 0.0057, -0.0019, -0.0088, -0.0148, -0.0196, -0.0227])
f_tx_patella_l=interpolate.interp1d(x, y, kind="cubic")
q[26, :]=0.95799999999999996*f_tx_patella_l(knee_angle_l_beta)

# dof axis: patellofemoral_l at translation2 on 0 1 0
y=np.array([-0.0108, -0.019, -0.0263, -0.0322, -0.0367, -0.0395, -0.0408, -0.0404, -0.0384, -0.0349, -0.0301, -0.0245, -0.0187])
f_ty_patella_l=interpolate.interp1d(x, y, kind="cubic")
q[27, :]=0.95799999999999996*f_ty_patella_l(knee_angle_l_beta)

# dof axis: patellofemoral_l at rotation1 on 0 0 1
y=np.array([0.00113686, -0.00629212, -0.105582, -0.253683, -0.414245, -0.579047, -0.747244, -0.91799, -1.09044, -1.26379, -1.43763, -1.61186, -1.78634])
f_rz_patella_l=interpolate.interp1d(x, y, kind="cubic")
q[28, :]=f_rz_patella_l(knee_angle_l_beta)

b = BiorbdViz(loaded_model=model)
b.load_movement(q)