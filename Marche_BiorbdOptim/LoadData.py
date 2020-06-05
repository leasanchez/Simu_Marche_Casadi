from ezc3d import c3d
import numpy as np
from scipy.interpolate import interp1d
import scipy.io as sio


def Get_Event(file):
    # Find event from c3d file : heel strike (HS) and toe off (TO)
    # Determine the indexes of the beginning and end of each phases

    measurements = c3d(file)
    time = measurements["parameters"]["EVENT"]["TIMES"]["value"][1, :]
    labels_time = measurements["parameters"]["EVENT"]["LABELS"]["value"]
    freq = measurements["parameters"]["POINT"]["RATE"]["value"][0]
    get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]

    RHS = time[get_indexes("RHS", labels_time)]
    RTO = time[get_indexes("RTO", labels_time)]
    if len(RTO) > 1:
        RTO = max(RTO)
    else:
        RTO = RTO[0]

    start = round(RHS[0] * freq) + 1
    stop_stance = round(RTO * freq) + 1
    stop = round(RHS[1] * freq) + 1

    return start, stop_stance, stop


def GetForces(file):
    measurements = c3d(file)
    analog = measurements["data"]["analogs"]
    labels_analog = measurements["parameters"]["ANALOG"]["LABELS"]["value"]
    nbPF = measurements["parameters"]["FORCE_PLATFORM"]["USED"]["value"][0][0]

    F = np.zeros((len(analog[0, 0, :]), 3, nbPF))
    for p in range(nbPF):
        F[:, 0, p] = analog[0, labels_analog.index("Fx" + str(p + 1)), :].squeeze()  # Fx
        F[:, 1, p] = analog[0, labels_analog.index("Fy" + str(p + 1)), :].squeeze()  # Fy
        F[:, 2, p] = analog[0, labels_analog.index("Fz" + str(p + 1)), :].squeeze()  # Fz

    return F


def GetMoment(file):
    measurements = c3d(file)
    analog = measurements["data"]["analogs"]
    labels_analog = measurements["parameters"]["ANALOG"]["LABELS"]["value"]
    nbPF = measurements["parameters"]["FORCE_PLATFORM"]["USED"]["value"][0][0]

    M = np.zeros((len(analog[0, 0, :]), 3, nbPF))
    for p in range(nbPF):
        M[:, 0, p] = analog[0, labels_analog.index("Mx" + str(p + 1)), :].squeeze() * 1e-3  # Fx
        M[:, 1, p] = analog[0, labels_analog.index("My" + str(p + 1)), :].squeeze() * 1e-3  # Fy
        M[:, 2, p] = analog[0, labels_analog.index("Mz" + str(p + 1)), :].squeeze() * 1e-3  # Fz

    return M


def ComputeCoP(file):
    measurements = c3d(file)
    analog = measurements["data"]["analogs"]
    nbPF = measurements["parameters"]["FORCE_PLATFORM"]["USED"]["value"][0][0]
    corners = np.reshape(
        np.reshape(measurements["parameters"]["FORCE_PLATFORM"]["CORNERS"]["value"] * 1e-3, (3 * 4 * 2, 1)), (2, 4, 3)
    )  # platform x corners x coord

    CoP1 = np.zeros(((len(analog[0, 0, :]), 3, nbPF)))
    CoP = np.zeros(((len(analog[0, 0, :]), 3, nbPF)))
    F = GetForces(file)
    M = GetMoment(file)

    for p in range(nbPF):
        # Attention X et Y sont inversés sur la plaque !!!
        CoP1[:, 0, p] = np.divide(M[:, 0, p], F[:, 2, p])  # Mx/Fz
        CoP1[:, 1, p] = -np.divide(M[:, 1, p], F[:, 2, p])  # -My/Fz
        CoP1[:, :, p][np.isnan(CoP1[:, :, p])] = 0

        # Center of the platform
        if p == 0:
            CoP[:, 0, p] = (corners[p, 1, 0] - corners[p, 2, 0]) / 2 + CoP1[:, 0, p]  # xcenter + CoPx
            CoP[:, 1, p] = (corners[p, 0, 1] - corners[p, 1, 1]) / 2 + CoP1[:, 1, p]  # ycenter + CoPy
        else:
            CoP[:, 0, p] = corners[p, 2, 0] + (corners[p, 1, 0] - corners[p, 2, 0]) / 2 + CoP1[:, 0, p]
            CoP[:, 1, p] = (corners[p, 0, 1] - corners[p, 1, 1]) / 2 + CoP1[:, 1, p]
    return CoP


def GetGroundReactionForces(file):
    measurements = c3d(file)
    analog = measurements["data"]["analogs"]
    labels_analog = measurements["parameters"]["ANALOG"]["LABELS"]["value"]
    nbPF = measurements["parameters"]["FORCE_PLATFORM"]["USED"]["value"][0][0]

    GRF = np.zeros((len(analog[0, 0, :]), 3, nbPF))
    for p in range(nbPF):
        GRF[:, 0, p] = analog[0, labels_analog.index("Fx" + str(p + 1)), :].squeeze()  # Fx
        GRF[:, 1, p] = analog[0, labels_analog.index("Fy" + str(p + 1)), :].squeeze()  # Fy
        GRF[:, 2, p] = -analog[0, labels_analog.index("Fz" + str(p + 1)), :].squeeze()  # Fz
    return GRF


def load_data_markers(name_subject, biorbd_model, final_time, n_shooting_points, GaitPhase):
    # Load c3d file and get the muscular excitation from emg
    file = "../../DonneesMouvement/" + name_subject + "_out.c3d"
    nbMarker = biorbd_model.nbMarkers()

    # LOAD C3D FILE
    measurements = c3d(file)
    points = measurements["data"]["points"]
    labels_markers = measurements["parameters"]["POINT"]["LABELS"]["value"]

    # GET THE TIME OF TOE OFF & HEEL STRIKE
    [start, stop_stance, stop] = Get_Event(file)

    # GET THE MARKERS POSITION (X, Y, Z) AT EACH POINT
    markers = np.zeros((3, nbMarker, len(points[0, 0, :])))

    # pelvis markers
    markers[:, 0, :] = points[:3, labels_markers.index("L_IAS"), :] * 1e-3  # L_IAS
    markers[:, 1, :] = points[:3, labels_markers.index("L_IPS"), :] * 1e-3  # L_IPS
    markers[:, 2, :] = points[:3, labels_markers.index("R_IPS"), :] * 1e-3  # R_IPS
    markers[:, 3, :] = points[:3, labels_markers.index("R_IAS"), :] * 1e-3  # R_IAS
    # femur R markers
    markers[:, 4, :] = points[:3, labels_markers.index("R_FTC"), :] * 1e-3  # R_FTC
    markers[:, 5, :] = points[:3, labels_markers.index("R_Thigh_Top"), :] * 1e-3  # R_Thigh_Top
    markers[:, 6, :] = points[:3, labels_markers.index("R_Thigh_Down"), :] * 1e-3  # R_Thigh_Down
    markers[:, 7, :] = points[:3, labels_markers.index("R_Thigh_Front"), :] * 1e-3  # R_Thigh_Front
    markers[:, 8, :] = points[:3, labels_markers.index("R_Thigh_Back"), :] * 1e-3  # R_Thigh_Back
    markers[:, 9, :] = points[:3, labels_markers.index("R_FLE"), :] * 1e-3  # R_FLE
    markers[:, 10, :] = points[:3, labels_markers.index("R_FME"), :] * 1e-3  # R_FME
    #  tibia R markers
    markers[:, 11, :] = points[:3, labels_markers.index("R_FAX"), :] * 1e-3  # R_FAX
    markers[:, 12, :] = points[:3, labels_markers.index("R_TTC"), :] * 1e-3  # R_TTC
    markers[:, 13, :] = points[:3, labels_markers.index("R_Shank_Top"), :] * 1e-3  # R_Shank_Top
    markers[:, 14, :] = points[:3, labels_markers.index("R_Shank_Down"), :] * 1e-3  # R_Shank_Down
    markers[:, 15, :] = points[:3, labels_markers.index("R_Shank_Front"), :] * 1e-3  # R_Shank_Front
    markers[:, 16, :] = points[:3, labels_markers.index("R_Shank_Tibia"), :] * 1e-3  # R_Shank_Tibia
    markers[:, 17, :] = points[:3, labels_markers.index("R_FAL"), :] * 1e-3  # R_FAL
    markers[:, 18, :] = points[:3, labels_markers.index("R_TAM"), :] * 1e-3  # R_TAM
    #  foot R markers
    markers[:, 19, :] = points[:3, labels_markers.index("R_FCC"), :] * 1e-3  # R_FCC
    markers[:, 20, :] = points[:3, labels_markers.index("R_FM1"), :] * 1e-3  # R_FM1
    markers[:, 21, :] = points[:3, labels_markers.index("R_FMP1"), :] * 1e-3  # R_FMP1
    markers[:, 22, :] = points[:3, labels_markers.index("R_FM2"), :] * 1e-3  # R_FM2
    markers[:, 23, :] = points[:3, labels_markers.index("R_FMP2"), :] * 1e-3  # R_FMP2
    markers[:, 24, :] = points[:3, labels_markers.index("R_FM5"), :] * 1e-3  # R_FM5
    markers[:, 25, :] = points[:3, labels_markers.index("R_FMP5"), :] * 1e-3  # R_FMP5

    # INTERPOLATE AND GET REAL POSITION FOR SHOOTING POINT FOR THE SWING PHASE
    if GaitPhase == "stance":
        t = np.linspace(0, final_time, int(stop_stance - start) + 1)
        node_t = np.linspace(0, final_time, n_shooting_points + 1)
        f = interp1d(t, markers[:, :, int(start) : int(stop_stance) + 1], kind="cubic")
        markers_ref = f(node_t)
    elif GaitPhase == "swing":
        t = np.linspace(0, final_time, int(stop - stop_stance) + 1)
        node_t = np.linspace(0, final_time, n_shooting_points + 1)
        f = interp1d(t, markers[:, :, int(stop_stance) : int(stop) + 1], kind="cubic")
        markers_ref = f(node_t)
    else:
        raise RuntimeError("Gaitphase doesn't exist")

    return node_t, markers_ref


def load_data_q(name_subject, biorbd_model, final_time, n_shooting_points, GaitPhase):
    # Create initial vector for joint position (nbNoeuds x nbQ)
    # Based on Kalman filter??
    c3d_file = "../../DonneesMouvement/" + name_subject + "_out.c3d"
    kalman_file = "../../DonneesMouvement/" + name_subject + "_out_MOD5000_leftHanded_GenderF_Florent_.Q2"

    # LOAD MAT FILE FOR GENERALIZED COORDINATES
    kalman = sio.loadmat(kalman_file)
    Q_real = kalman["Q2"]

    [start, stop_stance, stop] = Get_Event(c3d_file)

    # INTERPOLATE AND GET KALMAN JOINT POSITION FOR SHOOTING POINT FOR THE CYCLE PHASE
    if GaitPhase == "stance":
        t = np.linspace(0, final_time, int(stop_stance - start) + 1)
        node_t = np.linspace(0, final_time, n_shooting_points + 1)
        f = interp1d(t, Q_real[:, int(start) : int(stop_stance) + 1], kind="cubic")
        q_ref = f(node_t)
    elif GaitPhase == "swing":
        t = np.linspace(0, final_time, int(stop - stop_stance) + 1)
        node_t = np.linspace(0, final_time, n_shooting_points + 1)
        f = interp1d(t, Q_real[:, int(stop_stance) : int(stop) + 1], kind="cubic")
        q_ref = f(node_t)
    else:
        raise RuntimeError("Gaitphase doesn't exist")
    return q_ref


def load_data_qdot(name_subject, biorbd_model, final_time, n_shooting_points, GaitPhase):
    # Create initial vector for joint position (nbNoeuds x nbQ)
    # Based on Kalman filter??
    c3d_file = "../../DonneesMouvement/" + name_subject + "_out.c3d"
    kalman_file = "../../DonneesMouvement/" + name_subject + "_out_MOD5000_leftHanded_GenderF_Florent_.Q2"

    # LOAD MAT FILE FOR GENERALIZED COORDINATES
    kalman = sio.loadmat(kalman_file)
    Q_real = kalman["Q2"]
    Qdot = np.zeros((biorbd_model.nbQ(), Q_real.shape[1] - 1))
    dt = final_time / n_shooting_points
    for i in range(biorbd_model.nbQ()):
        Qdot[i, :] = np.diff(Q_real[i, :]) / dt

    [start, stop_stance, stop] = Get_Event(c3d_file)

    # INTERPOLATE AND GET KALMAN JOINT POSITION FOR SHOOTING POINT FOR THE CYCLE PHASE
    if GaitPhase == "stance":
        t = np.linspace(0, final_time, int(stop_stance - start) + 1)
        node_t = np.linspace(0, final_time, n_shooting_points + 1)
        f = interp1d(t, Qdot[:, int(start) : int(stop_stance) + 1], kind="cubic")
        qdot_ref = f(node_t)
    elif GaitPhase == "swing":
        t = np.linspace(0, final_time, int(stop - stop_stance) + 1)
        node_t = np.linspace(0, final_time, n_shooting_points + 1)
        f = interp1d(t, Qdot[:, int(stop_stance) : int(stop) + 1], kind="cubic")
        qdot_ref = f(node_t)
    else:
        raise RuntimeError("Gaitphase doesn't exist")

    return qdot_ref


def load_data_emg(name_subject, biorbd_model, final_time, n_shooting_points, GaitPhase):
    # Load c3d file and get the muscular excitation from emg
    file = "../../DonneesMouvement/" + name_subject + "_out.c3d"
    nbMuscle = biorbd_model.nbMuscleTotal()

    # LOAD C3D FILE
    measurements = c3d(file)
    points = measurements["data"]["points"]
    labels_points = measurements["parameters"]["POINT"]["LABELS"]["value"]

    # GET THE TIME OF TOE OFF & HEEL STRIKE
    [start, stop_stance, stop] = Get_Event(file)

    # GET THE MUSCULAR EXCITATION FROM EMG (NOT ALL MUSCLES)
    EMG = np.zeros(((nbMuscle - 7), len(points[0, 0, :])))

    EMG[9, :] = points[0, labels_points.index("R_Tibialis_Anterior"), :].squeeze()  # R_Tibialis_Anterior
    EMG[8, :] = points[0, labels_points.index("R_Soleus"), :].squeeze()  # R_Soleus
    EMG[7, :] = points[0, labels_points.index("R_Gastrocnemius_Lateralis"), :].squeeze()  # R_Gastrocnemius_Lateralis
    EMG[6, :] = points[0, labels_points.index("R_Gastrocnemius_Medialis"), :].squeeze()  # R_Gastrocnemius_Medialis
    EMG[5, :] = points[0, labels_points.index("R_Vastus_Medialis"), :].squeeze()  # R_Vastus_Medialis
    EMG[4, :] = points[0, labels_points.index("R_Rectus_Femoris"), :].squeeze()  # R_Rectus_Femoris
    EMG[3, :] = points[0, labels_points.index("R_Biceps_Femoris"), :].squeeze()  # R_Biceps_Femoris
    EMG[2, :] = points[0, labels_points.index("R_Semitendinosus"), :].squeeze()  # R_Semitendinous
    EMG[1, :] = points[0, labels_points.index("R_Gluteus_Medius"), :].squeeze()  # R_Gluteus_Medius
    EMG[0, :] = points[0, labels_points.index("R_Gluteus_Maximus"), :].squeeze()  # R_Gluteus_Maximus

    # INTERPOLATE AND GET REAL MUSCULAR EXCITATION FOR SHOOTING POINT FOR THE GAIT CYCLE PHASE
    if GaitPhase == "stance":
        t = np.linspace(0, final_time, int(stop_stance - start) + 1)
        node_t = np.linspace(0, final_time, n_shooting_points + 1)
        f = interp1d(t, EMG[:, int(start) : int(stop_stance) + 1], kind="cubic")
        emg_ref = f(node_t)
    elif GaitPhase == "swing":
        t = np.linspace(0, final_time, int(stop - stop_stance) + 1)
        node_t = np.linspace(0, final_time, n_shooting_points + 1)
        f = interp1d(t, EMG[:, int(stop_stance) : int(stop) + 1], kind="cubic")
        emg_ref = f(node_t)
    else:
        raise RuntimeError("Gaitphase doesn't exist")

    # RECTIFY EMG VALUES BETWEEN 0 & 1
    emg_ref[emg_ref < 0] = 1e-3
    emg_ref[emg_ref == 0] = 1e-3
    emg_ref[emg_ref > 1] = 1

    return emg_ref


def find_platform(file):
    GRW = GetGroundReactionForces(file)
    [start, stop_stance, stop] = Get_Event(file)
    P = np.array([sum(GRW[int(start) : int(stop_stance) + 1, 2, 0]), sum(GRW[int(start) : int(stop_stance) + 1, 2, 1])])
    idx_platform = np.where(P == P.max())[0][0]
    return idx_platform


def GetTime(file):
    measurements = c3d(file)
    freq = measurements["parameters"]["ANALOG"]["RATE"]["value"][0]
    [start, stop_stance, stop] = Get_Event(file)

    T = 1 / freq * (int(stop) - int(start) + 1)
    T_stance = 1 / freq * (int(stop_stance) - int(start) + 1)
    T_swing = 1 / freq * (int(stop) - int(stop_stance) + 1)
    return T, T_stance, T_swing


def GetTime_stance(file):
    measurements = c3d(file)
    freq = measurements["parameters"]["ANALOG"]["RATE"]["value"][0]
    [start, stop_stance, stop] = Get_Event(file)
    T_stance_tot = 1 / freq * (int(stop_stance) - int(start) + 1)

    CoP = ComputeCoP(file)
    ixd_platform = find_platform(file)

    idx_2_contact = np.where(CoP[int(start) : int(stop_stance), 1, (ixd_platform - 1) ** 2] == 0)[0][
        0
    ]  # toe of of the left leg (no signal on the PF)
    a = -CoP[int(start) : (int(stop_stance) - 20), 1, ixd_platform]
    idx_1_contact = np.where(a == a.max())[0][0]  # max before the left leg move forward

    T_Heel = 1 / freq * (int(idx_2_contact))
    T_2_contact = 1 / freq * (int(idx_1_contact) - int(idx_2_contact))
    T_Forefoot = 1 / freq * ((int(stop_stance) - int(start)) - int(idx_1_contact) + 1)
    T_stance = [T_Heel, T_2_contact, T_Forefoot]

    # plot
    #     plt.plot(-CoP[int(start): int(stop_stance), 1, ixd_platform], '+')
    #     plt.plot(-CoP[int(start): int(stop_stance), 1, (ixd_platform - 1) ** 2], '+')
    #     plt.plot([idx_2_contact, idx_2_contact], [np.min(-CoP[int(start): int(stop_stance), 1, (ixd_platform - 1) ** 2]),
    #                                               np.max(-CoP[int(start): int(stop_stance), 1, ixd_platform])], 'k--')
    #     plt.plot([idx_1_contact, idx_1_contact], [np.min(-CoP[int(start): int(stop_stance), 1, (ixd_platform - 1) ** 2]),
    #                                               np.max(-CoP[int(start): int(stop_stance), 1, ixd_platform])], 'k--')

    return T_stance


def load_data_GRF(name_subject, biorbd_model, n_shooting_points):
    # Load c3d file and get the muscular excitation from emg
    file = "../../DonneesMouvement/" + name_subject + "_out.c3d"
    nbNoeuds = n_shooting_points

    # GET GROUND REACTION WRENCHES
    GRW = GetGroundReactionForces(file)

    # time
    [start, stop_stance, stop] = Get_Event(file)
    T, T_stance, T_swing = GetTime(file)

    # FIND FORCE PLATFORM FOR RIGHT FOOT -- GET FORCES FOR MODEL
    idx_platform = find_platform(file)
    GRF = GRW[:, :, idx_platform].T

    # INTERPOLATE AND GET REAL FORCES FOR SHOOTING POINT FOR THE GAIT CYCLE PHASE
    t_stance = np.linspace(0, T_stance, int(stop_stance - start) + 1)
    node_t_stance = np.linspace(0, T_stance, nbNoeuds + 1)
    f_stance = interp1d(t_stance, GRF[:, int(start) : int(stop_stance) + 1], kind="cubic")
    GRF_real = f_stance(node_t_stance)

    # t_stance = GetTime_stance(file)

    return GRF_real, T, T_stance, T_swing


def load_muscularExcitation(emg_ref):
    # Create initial vector for muscular excitation (nbNoeuds x nbMus)
    # Based on EMG from the c3d file

    # INPUT
    # U_real          = muscular excitation from the c3d file

    # OUTPUT
    # U0             = initial guess for muscular excitation (3 x nbNoeuds)

    nbNoeuds = len(emg_ref[0, :])
    nbMus = len(emg_ref[:, 0])

    excitation_ref = np.zeros((nbMus + 7, nbNoeuds))

    excitation_ref[0, :] = emg_ref[0, :]  # glut_max1_r
    excitation_ref[1, :] = emg_ref[0, :]  # glut_max2_r
    excitation_ref[2, :] = emg_ref[0, :]  # glut_max3_r
    excitation_ref[3, :] = emg_ref[1, :]  # glut_med1_r
    excitation_ref[4, :] = emg_ref[1, :]  # glut_med2_r
    excitation_ref[5, :] = emg_ref[1, :]  # glut_med3_r
    excitation_ref[6, :] = emg_ref[2, :]  # semimem_r
    excitation_ref[7, :] = emg_ref[2, :]  # semiten_r
    excitation_ref[8, :] = emg_ref[3, :]  # bi_fem_r
    excitation_ref[9, :] = emg_ref[4, :]  # rectus_fem_r
    excitation_ref[10, :] = emg_ref[5, :]  # vas_med_r
    excitation_ref[11, :] = emg_ref[5, :]  # vas_int_r
    excitation_ref[12, :] = emg_ref[5, :]  # vas_lat_r
    excitation_ref[13, :] = emg_ref[6, :]  # gas_med_r
    excitation_ref[14, :] = emg_ref[7, :]  # gas_lat_r
    excitation_ref[15, :] = emg_ref[8, :]  # soleus_r
    excitation_ref[16, :] = emg_ref[9, :]  # tib_ant_r

    return excitation_ref
