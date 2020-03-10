from ezc3d import c3d
import biorbd
import numpy as np
from Platform_Force import *
from pylab import *
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


def load_data_markers(file, T, nbNoeuds, nbMarker, GaitPhase):
    # Load c3d file and get the muscular excitation from emg

    # INPUT
    # file            = name and path of the c3d file
    # T               = phase time
    # nbNoeuds        = shooting point for the phase
    # nbMarker        = markers number for the model
    # GaitPhase       = phase of the cycle : stance, swing

    # OUTPUT
    # M_real          = 3 x nMarker x nbNoeuds : muscular excitation from emg

    # LOAD C3D FILE
    measurements   = c3d(file)
    points         = measurements['data']['points']
    labels_markers = measurements['parameters']['POINT']['LABELS']['value']
    freq           = measurements['parameters']['POINT']['RATE']['value'][0]

    # GET THE TIME OF TOE OFF & HEEL STRIKE
    [start, stop_stance, stop] = Get_Event(file)

    # GET THE MARKERS POSITION (X, Y, Z) AT EACH POINT
    markers = np.zeros((3, nbMarker, len(points[0, 0,:])))
    # markers = [coord, marker, points de mesures] en mm

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
        # T = T_stance
        t = np.linspace(0, T, int(stop_stance - start + 1))
        node_t = np.linspace(0, T, nbNoeuds)
        f = interp1d(t, markers[:, :, int(start): int(stop_stance) + 1], kind='cubic')
        M_real = f(node_t)

    elif GaitPhase == 'swing':
        # T = T_swing
        t = np.linspace(0, T + 1/freq, int(stop - stop_stance) + 1)
        node_t = np.linspace(0, T + 1/freq, nbNoeuds + 1)
        f = interp1d(t, markers[:, :, int(stop_stance) + 1: int(stop) + 2], kind='cubic')
        M_real = f(node_t)
    return M_real





def load_data_emg(file, T, nbNoeuds, nbMuscle, GaitPhase):
    # Load c3d file and get the muscular excitation from emg

    # INPUT
    # file            = name and path of the c3d file
    # T               = phase time
    # nbNoeuds        = shooting point for the phase
    # nbMuscle        = muscle number for the model
    # GaitPhase       = phase of the cycle : stance, swing

    # OUTPUT
    # U_real          = (nMus - 7) x nbNoeuds : muscular excitation from emg

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
        # T = T_stance
        t      = np.linspace(0, T, int(stop_stance - start + 1))
        node_t = np.linspace(0, T, nbNoeuds)
        f      = interp1d(t, EMG[:, int(start): int(stop_stance) + 1], kind='cubic')
        U_real = f(node_t)

    elif GaitPhase == 'swing':
        # T = T_swing
        t      = np.linspace(0, T, int(stop - stop_stance))
        node_t = np.linspace(0, T, nbNoeuds)
        f = interp1d(t, EMG[:, int(stop_stance) + 1: int(stop) + 1], kind='cubic')
        U_real = f(node_t)

    # RECTIFY EMG VALUES BETWEEN 0 & 1
    U_real[U_real < 0]  = 1e-3
    U_real[U_real == 0] = 1e-3
    U_real[U_real > 1]  = 1

    return U_real



def find_TO(input, idx_in):
    idx_TO = idx_in + 1
    while input[idx_TO + 1] < input[idx_TO]:
        idx_TO = idx_TO + 1
    return idx_TO

def find_HS(input, idx_in):
    idx_HS = idx_in - 1
    while input[idx_HS - 1] < input[idx_HS]:
        idx_HS = idx_HS - 1
    return idx_HS


def load_data_GRF(file, nbNoeuds_stance, nbNoeuds_swing, GaitPhase):
    # Load c3d file and get the Ground Reaction Forces from teh force platform
    # based on the vertical GRF estimation of each phase time

    # INPUT
    # file            = name and path of the c3d file
    # nbNoeuds_stance = shooting point for the stance phase
    # nbNoeuds_swing  = shooting point for the swing phase
    # GaitPhase       = phase of the cycle : stance, swing, cycle

    # OUTPUT
    # GRF_real        = 3 x nbNoeuds with the Ground Reaction Forces values
    # T               = gaitcycle time
    # T_stance        = stance phase time

    # LOAD C3D FILE
    measurements = c3d(file)
    freq         = measurements['parameters']['ANALOG']['RATE']['value'][0]

    # GET FORCES & MOMENTS FROM PLATFORM
    F = GetForces(file)
    M = GetMoment(file)

    # GET GROUND REACTION WRENCHES
    GRW = GetGroundReactionForces(file)

    # CENTER OF PRESSURE
    # CoP     = ComputeCoP(file)
    # corners = np.reshape(np.reshape(measurements['parameters']['FORCE_PLATFORM']['CORNERS']['value']*1e-3, (3 * 4 * 2, 1)), (2, 4, 3))                                                                                    # coins de la plateforme

    # # Affichage CoP on force platform
    # plt.figure()
    # plt.plot(corners[0, :, 0],corners[0, :, 1],'o')
    # plt.plot(corners[1, :, 0],corners[1, :, 1],'o')
    # plt.plot(CoP[:, 0, 0], CoP[:, 1, 0], 'r+')
    # plt.plot(CoP[:, 0, 1], CoP[:, 1, 1], 'g+')
    # plt.axis('equal')


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
    T_swing  = 1/freq * (int(stop) - int(stop_stance))

    # FIND FORCE PLATFORM FOR RIGHT FOOT -- GET FORCES FOR MODEL
    P1 = sum(GRW[int(start): int(stop_stance) + 1, 2, 0])
    P2 = sum(GRW[int(start): int(stop_stance) + 1, 2, 1])

    if P1 > P2 :
        GRF = GRW[:, :, 0].T
    else:
        GRF = GRW[:, :, 1].T

    # # AFFICHAGE GROUND REACTION FORCES
    # plt.figure()
    # t_init = np.linspace(0, len(GRW[:, 0, 0])*1/freq, len(GRW[:, 0, 0]))
    # plt.plot(t_init, GRW[:, :, 0])
    # plt.plot(t_init, GRW[:, :, 1], '--')
    # plt.plot([RHS[0], RHS[1], RTO], [GRW[int(start + 1), 2, 0], GRW[int(stop + 1), 2, 0], GRW[int(stop_stance + 1), 2, 0]], '+')
    #
    # # Affichage ground reaction forces ajustées
    # t = np.linspace(0, T, int(stop - start) + 1)
    # plt.figure()
    # plt.plot(t, GRW[int(start): int(stop) + 1, :, 0])
    # plt.plot(t, GRW[int(start): int(stop) + 1, :, 1], '--')

    # INTERPOLATE AND GET REAL FORCES FOR SHOOTING POINT FOR THE GAIT CYCLE PHASE
    # Stance
    t_stance        = np.linspace(0, T_stance, int(stop_stance - start) + 1)
    node_t_stance   = np.linspace(0, T_stance, nbNoeuds_stance)
    f_stance        = interp1d(t_stance, GRF[:, int(start): int(stop_stance + 1)], kind ='cubic')
    GRF_real_stance = f_stance(node_t_stance)

    # Swing
    t_swing        = np.linspace(0, T_swing + 1/freq, int(stop - stop_stance))
    node_t_swing   = np.linspace(0, T_swing, nbNoeuds_swing)
    f_swing        = interp1d(t_swing, GRF[:, int(stop_stance) + 1: int(stop) + 1], kind ='cubic')
    GRF_real_swing = f_swing(node_t_swing)


    if GaitPhase == 'swing':
        GRF_real = GRF_real_swing

    elif GaitPhase == 'stance':
        GRF_real = GRF_real_stance

    elif GaitPhase == 'cycle':
        # ! Y = mvt !
        GRF_real = np.hstack([GRF_real_stance, GRF_real_swing])

    # # Afichage GRF_real
    # plt.figure()
    # node_t = np.hstack([node_t_stance, node_t_stance[-1] + node_t_swing])
    # plt.plot(node_t, GRF_real[2, :].T, '+-')
    # plt.plot([T_stance, T_stance], [0, 800], 'k--')

    return GRF_real, T, T_stance


