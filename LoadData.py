from ezc3d import c3d
import numpy as np
from Platform_Force import GetForces, GetMoment, GetGroundReactionForces
from scipy.interpolate import interp1d


def Get_Event(file):
    # Find event from c3d file : heel strike (HS) and toe off (TO)
    # Determine the indexes of the beginning and end of each phases

    measurements = c3d(file)
    time         = np.reshape(np.reshape(measurements['parameters']['EVENT']['TIMES']['value'], (7 * 2, 1)), (7, 2))[:, 1]
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


def load_data_markers(params, GaitPhase):
    # Load c3d file and get the muscular excitation from emg
    file     = params.file
    nbMarker = params.nbMarker
    T_stance = params.T_stance
    nbNoeuds_stance = params.nbNoeuds_stance
    T_swing = params.T_swing
    nbNoeuds_swing = params.nbNoeuds_swing

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

    # INTERPOLATE AND GET REAL POSITION FOR SHOOTING POINT FOR THE WHOLE CYCLE
    if GaitPhase == 'stance':
        t = np.linspace(0, T_stance, int(stop_stance - start + 1))
        node_t = np.linspace(0, T_stance, nbNoeuds_stance + 1)
        f = interp1d(t, markers[:, :, int(start): int(stop_stance) + 1], kind='cubic')
        M_real = f(node_t)

    elif GaitPhase == 'swing':
        t = np.linspace(0, T_swing, int(stop - stop_stance) + 1)
        node_t = np.linspace(0, T_swing, nbNoeuds_swing + 1)
        f = interp1d(t, markers[:, :, int(stop_stance): int(stop) + 1], kind='cubic')
        M_real = f(node_t)
    return M_real





def load_data_emg(params, GaitPhase):
    # Load c3d file and get the muscular excitation from emg

    file     = params.file
    nbMuscle = params.nbMus
    T_stance = params.T_stance
    nbNoeuds_stance = params.nbNoeuds_stance
    T_swing = params.T_swing
    nbNoeuds_swing = params.nbNoeuds_swing

    # LOAD C3D FILE
    measurements  = c3d(file)
    points        = measurements['data']['points']
    labels_points = measurements['parameters']['POINT']['LABELS']['value']

    # GET THE TIME OF TOE OFF & HEEL STRIKE
    [start, stop_stance, stop] = Get_Event(file)

    # GET THE MUSCULAR EXCITATION FROM EMG (NOT ALL MUSCLES)
    EMG = np.zeros(((nbMuscle - 7), len(points[0, 0, :])))

    EMG[0, :] = points[0, labels_points.index("R_Tibialis_Anterior"), :].squeeze()          # R_Tibialis_Anterior
    EMG[1, :] = points[0, labels_points.index("R_Soleus"), :].squeeze()                     # R_Soleus
    EMG[2, :] = points[0, labels_points.index("R_Gastrocnemius_Lateralis"), :].squeeze()    # R_Gastrocnemius_Lateralis
    EMG[3, :] = points[0, labels_points.index("R_Gastrocnemius_Medialis"), :].squeeze()     # R_Gastrocnemius_Medialis
    EMG[4, :] = points[0, labels_points.index("R_Vastus_Medialis"), :].squeeze()            # R_Vastus_Medialis
    EMG[5, :] = points[0, labels_points.index("R_Rectus_Femoris"), :].squeeze()             # R_Rectus_Femoris
    EMG[6, :] = points[0, labels_points.index("R_Biceps_Femoris"), :].squeeze()             # R_Biceps_Femoris
    EMG[7, :] = points[0, labels_points.index("R_Semitendinosus"), :].squeeze()             # R_Semitendinous
    EMG[8, :] = points[0, labels_points.index("R_Gluteus_Medius"), :].squeeze()             # R_Gluteus_Medius
    EMG[9, :] = points[0, labels_points.index("R_Gluteus_Maximus"), :].squeeze()            # R_Gluteus_Maximus

    # INTERPOLATE AND GET REAL MUSCULAR EXCITATION FOR SHOOTING POINT FOR THE GAIT CYCLE PHASE
    if GaitPhase == 'stance':
        t      = np.linspace(0, T_stance, int(stop_stance - start) + 1)
        node_t = np.linspace(0, T_stance, nbNoeuds_stance + 1)
        f      = interp1d(t, EMG[:, int(start): int(stop_stance) + 1], kind='cubic')
        U_real = f(node_t)

    elif GaitPhase == 'swing':
        t      = np.linspace(0, T_swing, int(stop - stop_stance) + 1)
        node_t = np.linspace(0, T_swing, nbNoeuds_swing + 1)
        f = interp1d(t, EMG[:, int(stop_stance): int(stop) + 1], kind='cubic')
        U_real = f(node_t)

    # RECTIFY EMG VALUES BETWEEN 0 & 1
    U_real[U_real < 0]  = 1e-3
    U_real[U_real == 0] = 1e-3
    U_real[U_real > 1]  = 1

    return U_real

def load_data_GRF(params, GaitPhase):
    # Load c3d file and get the Ground Reaction Forces from teh force platform
    # based on the vertical GRF estimation of each phase time

    file            = params.file
    nbNoeuds_stance = params.nbNoeuds_stance
    nbNoeuds_swing  = params.nbNoeuds_swing
    nbNoeuds        = params.nbNoeuds

    # LOAD C3D FILE
    measurements = c3d(file)
    freq         = measurements['parameters']['ANALOG']['RATE']['value'][0]

    # GET FORCES & MOMENTS FROM PLATFORM
    F = GetForces(file)
    M = GetMoment(file)

    # GET GROUND REACTION WRENCHES
    GRW = GetGroundReactionForces(file)

    # GET THE TIME OF TOE OFF & HEEL STRIKE
    time        = np.reshape(np.reshape(measurements['parameters']['EVENT']['TIMES']['value'], (7 * 2, 1)), (7, 2))[:, 1]
    labels_time = measurements['parameters']['EVENT']['LABELS']['value']
    get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]

    RHS         = time[get_indexes('RHS', labels_time)]
    RTO         = time[get_indexes('RTO', labels_time)]
    if len(RTO) > 1:
        RTO = max(RTO)
    else:
        RTO = RTO[0]

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
    if GaitPhase == 'swing':
        t_swing = np.linspace(0, T_swing, int(stop - stop_stance) + 1)
        node_t_swing = np.linspace(0, T_swing, nbNoeuds_swing + 1)
        f_swing = interp1d(t_swing, GRF[:, int(stop_stance): int(stop) + 1], kind='cubic')
        GRF_real = f_swing(node_t_swing)

    elif GaitPhase == 'stance':
        t_stance = np.linspace(0, T_stance, int(stop_stance - start) + 1)
        node_t_stance = np.linspace(0, T_stance, nbNoeuds_stance + 1)
        f_stance = interp1d(t_stance, GRF[:, int(start): int(stop_stance) + 1], kind='cubic')
        GRF_real = f_stance(node_t_stance)

    elif GaitPhase == 'cycle':
        t_stance = np.linspace(0, T_stance, int(stop_stance - start) + 1)
        node_t_stance = np.linspace(0, T_stance, nbNoeuds_stance + 1)
        f_stance = interp1d(t_stance, GRF[:, int(start): int(stop_stance) + 1], kind='cubic')
        GRF_real_stance = f_stance(node_t_stance)

        t_swing = np.linspace(0, T_swing, int(stop - stop_stance) + 1)
        node_t_swing = np.linspace(0, T_swing, nbNoeuds_swing + 1)
        f_swing = interp1d(t_swing, GRF[:, int(stop_stance): int(stop) + 1], kind='cubic')
        GRF_real_swing = f_swing(node_t_swing)

        GRF_real = np.hstack([GRF_real_stance[:, :-1], GRF_real_swing])

    return GRF_real, T, T_stance, T_swing


