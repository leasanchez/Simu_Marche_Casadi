from ezc3d import c3d
import numpy as np
from scipy.interpolate import interp1d
import scipy.io as sio

def Get_Event(file):
    # Find event from c3d file : heel strike (HS) and toe off (TO)
    # Determine the indexes of the beginning and end of each phases

    measurements = c3d(file)
    time         = measurements['parameters']['EVENT']['TIMES']['value'][1, :]
    labels_time  = measurements['parameters']['EVENT']['LABELS']['value']
    freq         = measurements['parameters']['POINT']['RATE']['value'][0]
    get_indexes  = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]

    RHS = time[get_indexes('RHS', labels_time)]
    RTO = time[get_indexes('RTO', labels_time)]
    if len(RTO) > 1:
        RTO = max(RTO)
    else:
        RTO = RTO[0]

    start       = round(RHS[0] * freq) + 1
    stop_stance = round(RTO * freq) + 1
    stop        = round(RHS[1] * freq) + 1

    return start, stop_stance, stop

def GetForces(file):
    measurements  = c3d(file)
    analog        = measurements['data']['analogs']
    labels_analog = measurements['parameters']['ANALOG']['LABELS']['value']
    nbPF          = measurements['parameters']['FORCE_PLATFORM']['USED']['value'][0][0]

    F = np.zeros((len(analog[0, 0,:]), 3, nbPF))
    for p in range(nbPF):
        F[:, 0, p] = analog[0, labels_analog.index("Fx" + str(p + 1)), :].squeeze()  # Fx
        F[:, 1, p] = analog[0, labels_analog.index("Fy" + str(p + 1)), :].squeeze()  # Fy
        F[:, 2, p] = analog[0, labels_analog.index("Fz" + str(p + 1)), :].squeeze()  # Fz

    return F

def GetGroundReactionForces(file):
    measurements  = c3d(file)
    analog        = measurements['data']['analogs']
    labels_analog = measurements['parameters']['ANALOG']['LABELS']['value']
    nbPF          = measurements['parameters']['FORCE_PLATFORM']['USED']['value'][0][0]

    GRF = np.zeros((len(analog[0, 0,:]), 3, nbPF))
    for p in range(nbPF):
        GRF[:, 0, p] = analog[0, labels_analog.index("Fx" + str(p + 1)), :].squeeze()    # Fx
        GRF[:, 1, p] = analog[0, labels_analog.index("Fy" + str(p + 1)), :].squeeze()    # Fy
        GRF[:, 2, p] = - analog[0, labels_analog.index("Fz" + str(p + 1)), :].squeeze()  # Fz
    return GRF

def load_data_markers(name_subject, biorbd_model, final_time, n_shooting_points, GaitPhase):
    # Load c3d file and get the muscular excitation from emg
    file = "../../DonneesMouvement/" + name_subject + "_out.c3d"
    nbMarker = biorbd_model.nbMarkers()

    # LOAD C3D FILE
    measurements   = c3d(file)
    points         = measurements['data']['points']
    labels_markers = measurements['parameters']['POINT']['LABELS']['value']

    # GET THE TIME OF TOE OFF & HEEL STRIKE
    [start, stop_stance, stop] = Get_Event(file)

    # GET THE MARKERS POSITION (X, Y, Z) AT EACH POINT
    markers = np.zeros((3, nbMarker, len(points[0, 0,:])))

    # pelvis markers
    markers[:, 0, :]  = points[:3, labels_markers.index("L_IAS"), :] * 1e-3                           # L_IAS
    markers[:, 1, :]  = points[:3, labels_markers.index("L_IPS"), :] * 1e-3                           # L_IPS
    markers[:, 2, :]  = points[:3, labels_markers.index("R_IPS"), :] * 1e-3                           # R_IPS
    markers[:, 3, :]  = points[:3, labels_markers.index("R_IAS"), :] * 1e-3                           # R_IAS
    # femur R markers
    markers[:, 4, :]  = points[:3, labels_markers.index("R_FTC"), :] * 1e-3                           # R_FTC
    markers[:, 5, :]  = points[:3, labels_markers.index("R_Thigh_Top"), :] * 1e-3                     # R_Thigh_Top
    markers[:, 6, :]  = points[:3, labels_markers.index("R_Thigh_Down"), :] * 1e-3                    # R_Thigh_Down
    markers[:, 7, :]  = points[:3, labels_markers.index("R_Thigh_Front"), :] * 1e-3                   # R_Thigh_Front
    markers[:, 8, :]  = points[:3, labels_markers.index("R_Thigh_Back"), :] * 1e-3                    # R_Thigh_Back
    markers[:, 9, :]  = points[:3, labels_markers.index("R_FLE"), :] * 1e-3                           # R_FLE
    markers[:, 10, :] = points[:3, labels_markers.index("R_FME"), :] * 1e-3                           # R_FME
    #  tibia R markers
    markers[:, 11, :] = points[:3, labels_markers.index("R_FAX"), :] * 1e-3                           # R_FAX
    markers[:, 12, :] = points[:3, labels_markers.index("R_TTC"), :] * 1e-3                           # R_TTC
    markers[:, 13, :] = points[:3, labels_markers.index("R_Shank_Top"), :] * 1e-3                     # R_Shank_Top
    markers[:, 14, :] = points[:3, labels_markers.index("R_Shank_Down"), :] * 1e-3                    # R_Shank_Down
    markers[:, 15, :] = points[:3, labels_markers.index("R_Shank_Front"), :] * 1e-3                   # R_Shank_Front
    markers[:, 16, :] = points[:3, labels_markers.index("R_Shank_Tibia"), :] * 1e-3                   # R_Shank_Tibia
    markers[:, 17, :] = points[:3, labels_markers.index("R_FAL"), :] * 1e-3                           # R_FAL
    markers[:, 18, :] = points[:3, labels_markers.index("R_TAM"), :] * 1e-3                           # R_TAM
    #  foot R markers
    markers[:, 19, :] = points[:3, labels_markers.index("R_FCC"), :] * 1e-3                           # R_FCC
    markers[:, 20, :] = points[:3, labels_markers.index("R_FM1"), :] * 1e-3                           # R_FM1
    markers[:, 21, :] = points[:3, labels_markers.index("R_FMP1"), :] * 1e-3                          # R_FMP1
    markers[:, 22, :] = points[:3, labels_markers.index("R_FM2"), :] * 1e-3                           # R_FM2
    markers[:, 23, :] = points[:3, labels_markers.index("R_FMP2"), :] * 1e-3                          # R_FMP2
    markers[:, 24, :] = points[:3, labels_markers.index("R_FM5"), :] * 1e-3                           # R_FM5
    markers[:, 25, :] = points[:3, labels_markers.index("R_FMP5"), :] * 1e-3                          # R_FMP5

    # INTERPOLATE AND GET REAL POSITION FOR SHOOTING POINT FOR THE SWING PHASE
    if GaitPhase == 'stance':
        t = np.linspace(0, final_time, int(stop_stance - start) + 1)
        node_t = np.linspace(0, final_time, n_shooting_points + 1)
        f = interp1d(t, markers[:, :, int(start): int(stop_stance) + 1], kind='cubic')
        markers_ref = f(node_t)
    elif GaitPhase == 'swing':
        t = np.linspace(0, final_time, int(stop - stop_stance) + 1)
        node_t = np.linspace(0, final_time, n_shooting_points + 1)
        f = interp1d(t, markers[:, :, int(stop_stance): int(stop) + 1], kind='cubic')
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
    Q_real = kalman['Q2']

    [start, stop_stance, stop] = Get_Event(c3d_file)

    # INTERPOLATE AND GET KALMAN JOINT POSITION FOR SHOOTING POINT FOR THE CYCLE PHASE
    if GaitPhase == 'stance':
        t      = np.linspace(0, final_time, int(stop_stance - start) + 1)
        node_t = np.linspace(0, final_time, n_shooting_points + 1)
        f      = interp1d(t, Q_real[:, int(start): int(stop_stance) + 1], kind='cubic')
        q_ref  = f(node_t)
    elif GaitPhase == 'swing':
        t      = np.linspace(0, final_time, int(stop - stop_stance) + 1)
        node_t = np.linspace(0, final_time, n_shooting_points + 1)
        f      = interp1d(t, Q_real[:, int(stop_stance): int(stop) + 1], kind='cubic')
        q_ref  = f(node_t)
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
    Q_real = kalman['Q2']
    Qdot = np.zeros((biorbd_model.nbQ(), Q_real.shape[1] - 1))
    dt = final_time/n_shooting_points
    for i in range(biorbd_model.nbQ()):
        Qdot[i, :] = np.diff(Q_real[i, :])/dt

    [start, stop_stance, stop] = Get_Event(c3d_file)

    # INTERPOLATE AND GET KALMAN JOINT POSITION FOR SHOOTING POINT FOR THE CYCLE PHASE
    if GaitPhase == 'stance':
        t      = np.linspace(0, final_time, int(stop_stance - start) + 1)
        node_t = np.linspace(0, final_time, n_shooting_points + 1)
        f      = interp1d(t, Qdot[:, int(start): int(stop_stance) + 1], kind='cubic')
        qdot_ref  = f(node_t)
    elif GaitPhase == 'swing':
        t      = np.linspace(0, final_time, int(stop - stop_stance) + 1)
        node_t = np.linspace(0, final_time, n_shooting_points + 1)
        f      = interp1d(t, Qdot[:, int(stop_stance): int(stop) + 1], kind='cubic')
        qdot_ref  = f(node_t)
    else:
        raise RuntimeError("Gaitphase doesn't exist")

    return qdot_ref

def load_data_emg(name_subject, biorbd_model, final_time, n_shooting_points, GaitPhase):
    # Load c3d file and get the muscular excitation from emg
    file = "../../DonneesMouvement/" + name_subject + "_out.c3d"
    nbMuscle = biorbd_model.nbMuscleTotal()

    # LOAD C3D FILE
    measurements  = c3d(file)
    points        = measurements['data']['points']
    labels_points = measurements['parameters']['POINT']['LABELS']['value']

    # GET THE TIME OF TOE OFF & HEEL STRIKE
    [start, stop_stance, stop] = Get_Event(file)

    # GET THE MUSCULAR EXCITATION FROM EMG (NOT ALL MUSCLES)
    EMG = np.zeros(((nbMuscle - 7), len(points[0, 0, :])))

    EMG[9, :] = points[0, labels_points.index("R_Tibialis_Anterior"), :].squeeze()          # R_Tibialis_Anterior
    EMG[8, :] = points[0, labels_points.index("R_Soleus"), :].squeeze()                     # R_Soleus
    EMG[7, :] = points[0, labels_points.index("R_Gastrocnemius_Lateralis"), :].squeeze()    # R_Gastrocnemius_Lateralis
    EMG[6, :] = points[0, labels_points.index("R_Gastrocnemius_Medialis"), :].squeeze()     # R_Gastrocnemius_Medialis
    EMG[5, :] = points[0, labels_points.index("R_Vastus_Medialis"), :].squeeze()            # R_Vastus_Medialis
    EMG[4, :] = points[0, labels_points.index("R_Rectus_Femoris"), :].squeeze()             # R_Rectus_Femoris
    EMG[3, :] = points[0, labels_points.index("R_Biceps_Femoris"), :].squeeze()             # R_Biceps_Femoris
    EMG[2, :] = points[0, labels_points.index("R_Semitendinosus"), :].squeeze()             # R_Semitendinous
    EMG[1, :] = points[0, labels_points.index("R_Gluteus_Medius"), :].squeeze()             # R_Gluteus_Medius
    EMG[0, :] = points[0, labels_points.index("R_Gluteus_Maximus"), :].squeeze()            # R_Gluteus_Maximus

    # INTERPOLATE AND GET REAL MUSCULAR EXCITATION FOR SHOOTING POINT FOR THE GAIT CYCLE PHASE
    if GaitPhase == 'stance':
        t = np.linspace(0, final_time, int(stop_stance - start) + 1)
        node_t = np.linspace(0, final_time, n_shooting_points + 1)
        f = interp1d(t, EMG[:, int(start): int(stop_stance) + 1], kind='cubic')
        emg_ref = f(node_t)
    elif GaitPhase == 'swing':
        t = np.linspace(0, final_time, int(stop - stop_stance) + 1)
        node_t = np.linspace(0, final_time, n_shooting_points + 1)
        f = interp1d(t, EMG[:, int(stop_stance): int(stop) + 1], kind='cubic')
        emg_ref = f(node_t)
    else:
        raise RuntimeError("Gaitphase doesn't exist")

    # RECTIFY EMG VALUES BETWEEN 0 & 1
    emg_ref[emg_ref < 0]  = 1e-3
    emg_ref[emg_ref == 0] = 1e-3
    emg_ref[emg_ref > 1]  = 1

    return emg_ref

def load_data_GRF(name_subject, biorbd_model, n_shooting_points):
    # Load c3d file and get the muscular excitation from emg
    file = "../../DonneesMouvement/" + name_subject + "_out.c3d"
    nbNoeuds = n_shooting_points

    # LOAD C3D FILE
    measurements = c3d(file)
    freq         = measurements['parameters']['ANALOG']['RATE']['value'][0]

    # GET GROUND REACTION WRENCHES
    GRW = GetGroundReactionForces(file)

    [start, stop_stance, stop] = Get_Event(file)

    # time
    T        = 1/freq * (int(stop) - int(start) + 1)
    T_stance = 1/freq * (int(stop_stance) - int(start) + 1)  # point stop stance inclus
    T_swing  = 1/freq * (int(stop) - int(stop_stance) + 1)

    # FIND FORCE PLATFORM FOR RIGHT FOOT -- GET FORCES FOR MODEL
    P1 = sum(GRW[int(start): int(stop_stance) + 1, 2, 0])
    P2 = sum(GRW[int(start): int(stop_stance) + 1, 2, 1])

    if P1 > P2 :
        GRF = GRW[:, :, 0].T
    else:
        GRF = GRW[:, :, 1].T

    # INTERPOLATE AND GET REAL FORCES FOR SHOOTING POINT FOR THE GAIT CYCLE PHASE
    t_stance = np.linspace(0, T_stance, int(stop_stance - start) + 1)
    node_t_stance = np.linspace(0, T_stance, nbNoeuds + 1)
    f_stance = interp1d(t_stance, GRF[:, int(start): int(stop_stance) + 1], kind='cubic')
    GRF_real = f_stance(node_t_stance)

    return GRF_real, T, T_stance, T_swing