import pandas as pd
import numpy as np
import biorbd
from scipy import interpolate
from matplotlib import pyplot as plt

def echantillon_size(dt_init, dt_target, x, nb_shooting):
    idx = int(dt_target / dt_init)
    if len(x.shape) == 1:
        X = x[0::idx]
    else:
        X = np.zeros((x.shape[0], nb_shooting + 1))
        for i in range(x.shape[0]):
            X[i] = x[i, 0::idx]
    return X

def get_state_tracked(t_init, t_end, final_time, nb_q, nb_shooting):
    # --- get q ---
    df =pd.read_excel("coordinates.xlsx")
    df = np.array(df)
    idx_init = np.where(df[:, 0]==t_init)[0][0]
    idx_end = np.where(df[:, 0]==t_end)[0][0]

    q = np.zeros((nb_q, (idx_end - idx_init) + 1))
    q[:3, :]=df[idx_init:idx_end + 1, 4:7].T # pelvis translation
    q[3:6, :]=np.pi/180 * df[idx_init:idx_end + 1, 1:4].T  # pelvis rotation
    q[6:9, :]=np.pi/180 * df[idx_init:idx_end + 1, 7:10].T  #hip r
    # q[9:11, :]=fcn
    q[11, :]=np.pi/180 * df[idx_init:idx_end + 1, 10]  #knee r
    # q[12:14, :]=fcn
    # q[14:17, :]=fcn
    q[17, :]=np.pi/180 * df[idx_init:idx_end + 1, 12]   #ankle r

    q[18, :]=np.pi/180 * df[idx_init:idx_end + 1, 15]
    q[19:21, :]=-np.pi/180 * df[idx_init:idx_end + 1, 16:18].T  #hip l
    # q[21:23, :]=fcn
    q[23, :]=-np.pi/180 * df[idx_init:idx_end + 1, 18]  #knee l
    # q[24:26, :]=fcn -- ok
    # q[26:29, :]=fcn -- ok
    q[29, :]=np.pi/180 * df[idx_init:idx_end + 1, 20].T  # ankle l


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
    knee_angle_r_beta=np.array(df[idx_init:idx_end + 1, 11], dtype=float)

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
    q[21, :]=0.95799999999999996*f_ty_knee_l(-q[23, :])

    # dof axis: walker_knee_l at translation2 on 0 0 1
    y=np.array([0, -0.000988, -0.001899, -0.002734, -0.003492, -0.004173, -0.004777, -0.005305, -0.005756, -0.00613, -0.006427, -0.006648, -0.006792])
    f_tz_knee_l=interpolate.interp1d(x, y, kind="cubic")
    q[22, :]=0.95799999999999996*f_tz_knee_l(-q[23, :])

    # dof axis: walker_knee_l at rotation2 on 0 0 1
    y=np.array([0, 0.0126809, 0.0226969, 0.0296054, 0.0332049, 0.0335354, 0.0308779, 0.0257548, 0.0189295, 0.011407, 0.00443314, -0.00050475, -0.0016782])
    f_rz_knee_l=interpolate.interp1d(x, y, kind="cubic")
    q[24, :]=f_rz_knee_l(-q[23, :])

    # dof axis: walker_knee_l at rotation3 on 0 1 0
    y=np.array([0, -0.059461, -0.109399, -0.150618, -0.18392, -0.210107, -0.229983, -0.24435, -0.254012, -0.25977, -0.262428, -0.262788, -0.261654])
    f_ry_knee_l=interpolate.interp1d(x, y, kind="cubic")
    q[25, :]=f_ry_knee_l(-q[23, :])

    # q[26:29, :]=fcn
    # patella -- knee_angle_l_beta
    knee_angle_l_beta=np.array(df[idx_init:idx_end + 1, 19], dtype=float)

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
    Q = echantillon_size(t[1], node_t[1], q, nb_shooting)
    Qdot = echantillon_size(t[1], node_t[1], qdot, nb_shooting)
    Qddot = echantillon_size(t[1], node_t[1], qddot, nb_shooting)
    return Q, Qdot, Qddot

def get_state_from_solution_MI(t_init, t_end, final_time, nb_q, nb_shooting):
    # Alias
    nb_q_opensim = 18
    nb_mus = 80
    node_t = np.linspace(0, final_time, nb_shooting + 1)

    Q = np.zeros((nb_q, nb_shooting + 1))
    Qdot = np.zeros((nb_q, nb_shooting + 1))

    # --- read data from excel file ---
    df =pd.read_excel("example3DWalking_MocoInverse_solution.xlsx")
    df = np.array(df)

    # --- get idx rows ---
    idx_init = np.where(df[:, 0]==t_init)[0][0]
    idx_end = np.where(df[:, 0]==t_end)[0][0]
    t = np.array(df[idx_init:idx_end + 1, 0], dtype=float) - t_init

    # --- get muscles activation ---
    activation = df[idx_init:idx_end + 1, 1: (nb_mus + 1)].T  # nb_mus x nb_nodes

    # --- get joint angle and velocity ---
    # init
    q = np.zeros((nb_q, t.shape[0]))
    dq = np.zeros((nb_q, t.shape[0]))
    q_sol = np.zeros((nb_q_opensim, t.shape[0]))
    qdot_sol = np.zeros((nb_q_opensim, t.shape[0]))
    label = []
    for i in range(nb_q_opensim):
        q_sol[i, :] = df[idx_init:idx_end + 1, (nb_mus + 1) + 2*i]
        label.append(df[15, (nb_mus + 1) + 2*i])
        qdot_sol[i, :] = df[idx_init:idx_end + 1, (nb_mus + 1) + 2*i + 1]

    # Right leg
    q[:3, :]=q_sol[3:6, :]
    dq[:3, :]=qdot_sol[3:6, :] # pelvis translation
    q[3:6, :] = q_sol[:3, :]
    dq[3:6, :] = qdot_sol[:3, :]  # pelvis translation
    q[6:9, :]=q_sol[6:9, :]
    dq[6:9, :]=qdot_sol[6:9, :] #hip r
    q[11, :]=q_sol[12, :]
    dq[11, :]=qdot_sol[12, :] #knee r
    q[17, :]=q_sol[16, :]
    dq[17, :]=qdot_sol[16, :] #ankle r

    # Left leg
    q[18, :] = q_sol[9, :]
    dq[18, :] = qdot_sol[9, :]
    q[19:21, :]= -q_sol[10:12, :]
    dq[19:21, :]= -qdot_sol[10:12, :] #hip l
    q[23, :]=-q_sol[14, :]
    dq[23, :]=-qdot_sol[14, :] #knee l
    q[29, :]=q_sol[17, :]
    dq[29, :]=qdot_sol[17, :] # ankle l

    # dpdt knee_angle_r
    x = np.array([0, 0.174533, 0.349066, 0.523599, 0.698132, 0.872665, 1.0472, 1.22173, 1.39626, 1.5708, 1.74533, 1.91986, 2.0944])
    y = np.array([0, 0.000479, 0.000835, 0.001086, 0.001251, 0.001346, 0.001391, 0.001403, 0.0014, 0.0014, 0.001421, 0.001481, 0.001599])
    f_ty_knee=interpolate.interp1d(x, y, kind="cubic")
    q[9, :]=0.95799999999999996*f_ty_knee(q[11, :])
    dq[9, :-1] = np.diff(q[9, :]) / np.diff(t)
    dq[9, -1] = dq[9, -2]

    # knee_angle_r
    # dof axis: walker_knee_r at translation2 on 0 0 1
    y=np.array([0, 0.000988, 0.001899, 0.002734, 0.003492, 0.004173, 0.004777, 0.005305, 0.005756, 0.00613, 0.006427, 0.006648, 0.006792])
    f_tz_knee=interpolate.interp1d(x, y, kind="cubic")
    q[10, :]=0.95799999999999996*f_tz_knee(q[11, :])
    dq[10, :-1] = np.diff(q[10, :]) / np.diff(t)
    dq[10, -1] = dq[10, -2]

    #dof axis: walker_knee_r at rotation2 on 0 0 1
    y=np.array([0, 0.0126809, 0.0226969, 0.0296054, 0.0332049, 0.0335354, 0.0308779, 0.0257548, 0.0189295, 0.011407, 0.00443314, -0.00050475, -0.0016782])
    f_rz_knee=interpolate.interp1d(x, y, kind="cubic")
    q[12, :]=f_rz_knee(q[11, :])
    dq[12, :-1] = np.diff(q[12, :]) / np.diff(t)
    dq[12, -1] = dq[12, -2]

    #dof axis: walker_knee_r at rotation2 on 0 0 1
    y=np.array([0, 0.059461, 0.109399, 0.150618, 0.18392, 0.210107, 0.229983, 0.24435, 0.254012, 0.25977, 0.262428, 0.262788, 0.261654])
    f_ry_knee=interpolate.interp1d(x, y, kind="cubic")
    q[13, :]=f_ry_knee(q[11, :])
    dq[13, :-1] = np.diff(q[13, :]) / np.diff(t)
    dq[13, -1] = dq[13, -2]

    # patella -- knee_angle_r_beta
    knee_angle_r_beta=np.array(q_sol[13, :], dtype=float)

    # dof axis: patellofemoral_r at translation1 on 1 0 0
    y=np.array([0.0524, 0.0488, 0.0437, 0.0371, 0.0296, 0.0216, 0.0136, 0.0057, -0.0019, -0.0088, -0.0148, -0.0196, -0.0227])
    f_tx_patella=interpolate.interp1d(x, y, kind="cubic")
    q[14, :]=0.95799999999999996*f_tx_patella(knee_angle_r_beta)
    dq[14, :-1] = np.diff(q[14, :]) / np.diff(t)
    dq[14, -1] = dq[14, -2]

    # dof axis: patellofemoral_r at translation2 on 0 1 0
    y=np.array([-0.0108, -0.019, -0.0263, -0.0322, -0.0367, -0.0395, -0.0408, -0.0404, -0.0384, -0.0349, -0.0301, -0.0245, -0.0187])
    f_ty_patella=interpolate.interp1d(x, y, kind="cubic")
    q[15, :]=0.95799999999999996*f_ty_patella(knee_angle_r_beta)
    dq[15, :-1] = np.diff(q[15, :]) / np.diff(t)
    dq[15, -1] = dq[15, -2]

    # dof axis: patellofemoral_r at rotation1 on 0 0 1
    y=np.array([0.00113686, -0.00629212, -0.105582, -0.253683, -0.414245, -0.579047, -0.747244, -0.91799, -1.09044, -1.26379, -1.43763, -1.61186, -1.78634])
    f_rz_patella=interpolate.interp1d(x, y, kind="cubic")
    q[16, :]=f_rz_patella(knee_angle_r_beta)
    dq[16, :-1] = np.diff(q[16, :]) / np.diff(t)
    dq[16, -1] = dq[16, -2]

    # knee_angle_l
    # dof axis: walker_knee_l at translation1 on 0 1 0
    y=np.array([0, 0.000479, 0.000835, 0.001086, 0.001251, 0.001346, 0.001391, 0.001403, 0.0014, 0.0014, 0.001421, 0.001481, 0.001599])
    f_ty_knee_l=interpolate.interp1d(x, y, kind="cubic")
    q[21, :]=0.95799999999999996*f_ty_knee_l(-q[23, :])
    dq[21, :-1] = np.diff(q[21, :]) / np.diff(t)
    dq[21, -1] = dq[21, -2]

    # dof axis: walker_knee_l at translation2 on 0 0 1
    y=np.array([0, -0.000988, -0.001899, -0.002734, -0.003492, -0.004173, -0.004777, -0.005305, -0.005756, -0.00613, -0.006427, -0.006648, -0.006792])
    f_tz_knee_l=interpolate.interp1d(x, y, kind="cubic")
    q[22, :]=0.95799999999999996*f_tz_knee_l(-q[23, :])
    dq[22, :-1] = np.diff(q[22, :]) / np.diff(t)
    dq[22, -1] = dq[22, -2]

    # dof axis: walker_knee_l at rotation2 on 0 0 1
    y=np.array([0, 0.0126809, 0.0226969, 0.0296054, 0.0332049, 0.0335354, 0.0308779, 0.0257548, 0.0189295, 0.011407, 0.00443314, -0.00050475, -0.0016782])
    f_rz_knee_l=interpolate.interp1d(x, y, kind="cubic")
    q[24, :]=f_rz_knee_l(-q[23, :])
    dq[24, :-1] = np.diff(q[24, :]) / np.diff(t)
    dq[24, -1] = dq[24, -2]

    # dof axis: walker_knee_l at rotation3 on 0 1 0
    y=np.array([0, -0.059461, -0.109399, -0.150618, -0.18392, -0.210107, -0.229983, -0.24435, -0.254012, -0.25977, -0.262428, -0.262788, -0.261654])
    f_ry_knee_l=interpolate.interp1d(x, y, kind="cubic")
    q[25, :]=f_ry_knee_l(-q[23, :])
    dq[25, :-1] = np.diff(q[25, :]) / np.diff(t)
    dq[25, -1] = dq[25, -2]

    # q[26:29, :]=fcn
    # patella -- knee_angle_l_beta
    knee_angle_l_beta=np.array(q_sol[15, :], dtype=float)

    # dof axis: patellofemoral_l at translation1 on 1 0 0
    y=np.array([0.0524, 0.0488, 0.0437, 0.0371, 0.0296, 0.0216, 0.0136, 0.0057, -0.0019, -0.0088, -0.0148, -0.0196, -0.0227])
    f_tx_patella_l=interpolate.interp1d(x, y, kind="cubic")
    q[26, :]=0.95799999999999996*f_tx_patella_l(knee_angle_l_beta)
    dq[26, :-1] = np.diff(q[26, :]) / np.diff(t)
    dq[26, -1] = dq[26, -2]

    # dof axis: patellofemoral_l at translation2 on 0 1 0
    y=np.array([-0.0108, -0.019, -0.0263, -0.0322, -0.0367, -0.0395, -0.0408, -0.0404, -0.0384, -0.0349, -0.0301, -0.0245, -0.0187])
    f_ty_patella_l=interpolate.interp1d(x, y, kind="cubic")
    q[27, :]=0.95799999999999996*f_ty_patella_l(knee_angle_l_beta)
    dq[27, :-1] = np.diff(q[27, :]) / np.diff(t)
    dq[27, -1] = dq[27, -2]

    # dof axis: patellofemoral_l at rotation1 on 0 0 1
    y=np.array([0.00113686, -0.00629212, -0.105582, -0.253683, -0.414245, -0.579047, -0.747244, -0.91799, -1.09044, -1.26379, -1.43763, -1.61186, -1.78634])
    f_rz_patella_l=interpolate.interp1d(x, y, kind="cubic")
    q[28, :]=f_rz_patella_l(knee_angle_l_beta)
    dq[28, :-1] = np.diff(q[28, :]) / np.diff(t)
    dq[28, -1] = dq[28, -2]

    node_t = np.linspace(0, final_time, nb_shooting + 1)
    Q = echantillon_size(t[1], node_t[1], q, nb_shooting)
    Qdot = echantillon_size(t[1], node_t[1], dq, nb_shooting)
    Activation =  echantillon_size(t[1], node_t[1], activation, nb_shooting)
    return Q, Qdot, Activation

def get_control_from_solution_MI(t_init, t_end, final_time, nb_q, nb_shooting):
    # Alias
    nb_q_opensim = 18
    nb_mus = 80
    node_t = np.linspace(0, final_time, nb_shooting + 1)

    # --- read data from excel file ---
    df = pd.read_excel("example3DWalking_MocoInverse_solution.xlsx")
    df = np.array(df)

    # --- get idx rows ---
    idx_init = np.where(df[:, 0] == t_init)[0][0]
    idx_end = np.where(df[:, 0] == t_end)[0][0]
    idx_controls = (1 + 2 * nb_q_opensim + nb_mus)
    t = np.array(df[idx_init:idx_end + 1, 0], dtype=float) - t_init

    # --- get muscles excitations ---
    excitation_sol = df[idx_init:idx_end + 1, (idx_controls + 6):(idx_controls + 6) + nb_mus].T

    # --- get residual torques ---
    tau_sol = np.zeros((nb_q, t.shape[0]))
    tau_sol[:3, :] = df[idx_init:idx_end + 1, idx_controls + 3: idx_controls + 6].T # pelvis translations
    tau_sol[3:6, :] = df[idx_init:idx_end + 1, idx_controls: idx_controls + 3].T  # pelvis rotations
    tau_sol[6:9, :] = df[idx_init:idx_end + 1, (idx_controls + 6 + nb_mus): (idx_controls + 6 + nb_mus) + 3].T # hip r
    tau_sol[11, :] = df[idx_init:idx_end + 1, (idx_controls + 6 + nb_mus) + 3]  # knee r
    tau_sol[17, :] = df[idx_init:idx_end + 1, (idx_controls + 6 + nb_mus) + 4]  # ankle r
    tau_sol[18:21, :] = df[idx_init:idx_end + 1, (idx_controls + 6 + nb_mus) + 5: (idx_controls + 6 + nb_mus) + 8].T  # hip l
    tau_sol[23, :] = -df[idx_init:idx_end + 1,(idx_controls + 6 + nb_mus) + 8]  # knee l
    tau_sol[29, :] = df[idx_init:idx_end + 1,(idx_controls + 6 + nb_mus) + 9]  # ankle l

    Tau = echantillon_size(t[1], node_t[1], tau_sol, nb_shooting)
    Excitation = echantillon_size(t[1], node_t[1], excitation_sol, nb_shooting)
    return Tau, Excitation

def get_tau_from_inverse_dynamics(file, t_init, t_end, final_time, nb_q, nb_shooting):
    # --- read xlx data & get corresponding indexes ---
    df =pd.read_excel(file) # solution file
    df = np.array(df)
    idx_init_q = np.where(df[:, 0]==t_init)[0][0] # init time idx
    idx_end_q = np.where(df[:, 0]==t_end)[0][0]  # end time idx

    # --- time vector ---
    t = np.array(df[idx_init_q:idx_end_q + 1, 0], dtype=float) - t_init # OpenSim time vector
    node_t = np.linspace(0, final_time, nb_shooting + 1) # time vector

    # --- get control ---
    tau_iv = df[idx_init_q:idx_end_q + 1, 1:19] # get residual torque

    tau = np.zeros((nb_q, tau_iv.shape[0]))
    tau[:3, :] = tau_iv[:, 3:6].T # pelvis translations
    tau[3:6, :] = tau_iv[:, :3].T  # pelvis rotations
    tau[6:9, :] = tau_iv[:, 6:9].T  # hip r
    tau[11, :] = tau_iv[:, 12]  # knee r
    tau[17, :] = tau_iv[:, 16]  # ankle r
    tau[18, :] = tau_iv[:, 9]
    tau[19:21, :] = -tau_iv[:, 10:12].T  # hip l
    tau[23, :] = -tau_iv[:, 14]  # knee l
    tau[29, :] = tau_iv[:, 17]  # ankle l

    Tau = echantillon_size(round(t[1], 4), node_t[1], tau, nb_shooting)
    return Tau

def get_grf(t_init, t_end, final_time, nb_shooting):
    # --- read xlx file ---
    df =pd.read_excel("grf_walk.xlsx")
    df = np.array(df) # read file
    # --- get idx ---
    idx_init_grf = np.where(df[:, 0]==t_init)[0][0]
    idx_end_grf = np.where(df[:, 0]==t_end)[0][0] # index for mvt
    # --- time vector ---
    t = np.linspace(0, final_time, len(df[idx_init_grf:idx_end_grf + 1, 0]))  # time vector for data
    node_t = np.linspace(0, final_time, nb_shooting + 1)  # time vector for simulation

    # --- get reaction forces ---
    Force_grf = [df[idx_init_grf: idx_end_grf + 1, 1:4].T, df[idx_init_grf: idx_end_grf + 1, 10:13].T]
    Moment_grf = [df[idx_init_grf: idx_end_grf + 1, 7:10].T, df[idx_init_grf: idx_end_grf + 1, 16:19].T]

    Force = []
    Moment = []
    for i in range(len(Force_grf)):
        force = Force_grf[i]
        moment = Moment_grf[i]
        f = np.zeros((3, nb_shooting + 1))
        m = np.zeros((3, nb_shooting + 1))
        for i in range(force.shape[0]):
            f[i] = echantillon_size(t[1], node_t[1], force[i], nb_shooting)
            m[i] = echantillon_size(t[1], node_t[1], moment[i], nb_shooting)
        Force.append(f)
        Moment.append(m)
    return Force, Moment

def get_position(t_init, t_end, final_time, nb_shooting):
    # --- read xlx file ---
    df =pd.read_excel("grf_walk.xlsx")
    df = np.array(df) # read file
    # --- get idx ---
    idx_init_grf = np.where(df[:, 0] == t_init)[0][0]
    idx_end_grf = np.where(df[:, 0] == t_end)[0][0]  # index for mvt
    # --- time vector ---
    t = np.linspace(0, final_time, len(df[idx_init_grf:idx_end_grf + 1, 0]))  # time vector for data
    node_t = np.linspace(0, final_time, nb_shooting + 1)  # time vector for simulation

    # --- get CoP position ---
    position = [df[idx_init_grf: idx_end_grf + 1, 4:7].T, df[idx_init_grf: idx_end_grf + 1, 13:16].T] # get data
    # --- resample ---
    Position = []
    for data in position:
        pos = np.zeros((3, nb_shooting + 1))  # interpolate data
        for i in range(data.shape[0]):
            pos[i] = echantillon_size(t[1], node_t[1], data[i], nb_shooting)
        Position.append(pos)
    return Position

def compute_tau_from_muscles(model, states, controls):
    muscles_states = biorbd.VecBiorbdMuscleState(model.nbMuscleTotal())
    muscles_excitation = controls[model.nbQ():]
    muscles_activations = states[model.nbQ() + model.nbQdot():]

    for k in range(model.nbMuscleTotal()):
        muscles_states[k].setExcitation(muscles_excitation[k])
        muscles_states[k].setActivation(muscles_activations[k])

    muscles_tau = model.muscularJointTorque(muscles_states, states[:model.nbQ()], states[model.nbQ():model.nbQ() + model.nbQdot()]).to_mx()
    tau = muscles_tau + controls[:model.nbQ()]
    return tau