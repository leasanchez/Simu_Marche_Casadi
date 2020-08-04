import pandas as pd
import numpy as np
from casadi import MX, Function
from scipy import interpolate
from BiorbdViz import BiorbdViz
import biorbd
from matplotlib import pyplot as plt


def get_q(t_init, t_end, final_time, nb_q, nb_shooting):
    # --- get q ---
    df =pd.read_excel("coordinates_test.xlsx")
    df = np.array(df)
    idx_init_q = np.where(df[:, 0]==t_init)[0][0]
    idx_end_q = np.where(df[:, 0]==t_end)[0][0]
    q_pelvis = df[idx_init_q:idx_end_q + 1, 1:7].T
    q_rigth_leg = df[idx_init_q:idx_end_q + 1, 7:13].T
    q_left_leg = df[idx_init_q:idx_end_q + 1, 15:21].T

    q = np.zeros((nb_q, q_left_leg.shape[1]))
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
    # q[26:29, :]=fcn -- ok
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
    knee_angle_r_beta=np.array(q_rigth_leg[4, :], dtype=float)

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
    knee_angle_l_beta=np.array(q_left_leg[4, :], dtype=float)

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

    t = np.linspace(0, final_time, q.shape[1])
    qdot = np.zeros((nb_q, q.shape[1]))
    qddot = np.zeros((nb_q, q.shape[1]))
    for i in range(nb_q):
        # compute derivative
        qdot[i, :-1]=np.diff(q[i, :])/np.diff(t)
        qdot[i, -1]=qdot[i, -2]
        qddot[i, :-1]=np.diff(qdot[i, :])/np.diff(t)
        qddot[i, -1]=qddot[i, -2]

    node_t = np.linspace(0, final_time, nb_shooting + 1)
    Q = np.zeros((nb_q, nb_shooting + 1))
    Qdot = np.zeros((nb_q, nb_shooting + 1))
    Qddot = np.zeros((nb_q, nb_shooting + 1))
    for i in range(nb_q):
        # interpolate data
        fq = interpolate.interp1d(t, q[i], kind="cubic")
        Q[i] = fq(node_t)
        fdq = interpolate.interp1d(t, qdot[i], kind="cubic")
        Qdot[i] = fdq(node_t)
        fddq = interpolate.interp1d(t, qddot[i], kind="cubic")
        Qddot[i] = fddq(node_t)
    return Q, Qdot, Qddot


def get_grf(t_init, t_end, final_time, nb_shooting):
    # --- get reaction forces ---
    df =pd.read_excel("grf_walk_test.xlsx")
    df = np.array(df) # read file

    idx_init_grf = np.where(df[:, 0]==t_init)[0][0]
    idx_end_grf = np.where(df[:, 0]==t_end)[0][0] # index for mvt

    data_grf = []
    data_grf.append(df[idx_init_grf: idx_end_grf + 1, 1:4].T)
    data_grf.append(df[idx_init_grf: idx_end_grf + 1, 7:10].T)
    data_grf.append(df[idx_init_grf: idx_end_grf + 1, 10:13].T)
    data_grf.append(df[idx_init_grf: idx_end_grf + 1, 16:19].T) # get data

    grf = np.zeros((6*2, nb_shooting+1)) # interpolate data
    t = np.linspace(0, final_time, len(df[idx_init_grf:idx_end_grf + 1, 0])) # time vector for data
    node_t = np.linspace(0, final_time, nb_shooting + 1) # time vector for simulation
    g = 0
    for data in data_grf:
        for i in range(data.shape[0]):
            f = interpolate.interp1d(t, data[i], kind="cubic")
            grf[g] = f(node_t)
            g+=1
    return grf

# problems parameters
t_init = 0.81
t_end = 1.65
final_time = t_end - t_init
nb_shooting = 50

model = biorbd.Model("../../ModelesS2M/Open_Sim/subject_walk_armless_test_no_muscle_fext.bioMod")
[Q, Qdot, Qddot] = get_q(t_init, t_end, final_time, model.nbQ(), nb_shooting)
GRF = get_grf(t_init, t_end, final_time, nb_shooting)

symbolic_q = MX.sym("q", model.nbQ(), 1)
symbolic_qdot = MX.sym("qdot", model.nbQ(), 1)
symbolic_qddot = MX.sym("qddot", model.nbQ(), 1)
symbolic_fext = MX.sym("fext", 6*2, 1)
IV=Function(
    "InverseDynamics",
    [symbolic_q, symbolic_qdot, symbolic_qddot],
    [model.InverseDynamics(symbolic_q, symbolic_qdot, symbolic_qddot, symbolic_fext).to_mx()],
    ["q", "qdot", "qddot", "fext"],
    ["tau"],
    ).expand()
tau = IV(Q, Qdot, Qddot, GRF)
# b = BiorbdViz(loaded_model=model)
# b.load_movement(q)