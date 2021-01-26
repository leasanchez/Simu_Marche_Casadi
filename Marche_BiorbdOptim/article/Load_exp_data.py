from ezc3d import c3d
import numpy as np


def GetMarkers_Position(c3d_file, nb_marker):
    """
    get markers trajectories
    :param c3d_file : path of the c3d file (exp data)
    :param nb_marker : number of markers in the model
    :return: markers trajectories
    """

    # LOAD C3D FILE
    measurements = c3d(c3d_file)
    points = measurements["data"]["points"]
    labels_markers = measurements["parameters"]["POINT"]["LABELS"]["value"]

    # GET THE MARKERS POSITION (X, Y, Z) AT EACH POINT
    markers = np.zeros((3, nb_marker, len(points[0, 0, :])))

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
    return markers

def Get_Q(Q_file, nb_q):
    """
    get joints trajectories computed using Kalman filter
    :param Q_file: path of the text file (Q Kalman)
    :return: joint trajectories (Q)
    """
    Q = np.loadtxt(Q_file)
    nb_frame = int(len(Q) / nb_q)
    q = np.zeros((nb_q, nb_frame))
    for n in range(nb_frame):
        q[:, n] = Q[n * nb_q : n * nb_q + nb_q]
    return q

def GetForces(c3d_file):
    """
    get ground reaction forces from force platform
    :param c3d_file: path of the c3d file (exp data)
    :return: ground reaction forces
    """
    measurements = c3d(c3d_file, extract_forceplat_data=True)
    platform = measurements["data"]["platform"][0]
    F = platform["force"]
    return F

def GetMoment(c3d_file):
    """
    get moments value expressed at the center of pression
    from force platform
    :param c3d_file: path of the c3d file (exp data)
    :return: Moments
    """
    measurements = c3d(c3d_file, extract_forceplat_data=True)
    platform = measurements["data"]["platform"][0]
    M_CoP = platform["Tz"] * 1e-3
    return M_CoP

def GetCoP(c3d_file):
    """
    get the trajectory of the center of pression (CoP)
    from force platform
    :param c3d_file: path of the c3d file (exp data)
    :return: CoP trajectory
    """

    measurements = c3d(c3d_file, extract_forceplat_data=True)
    platform = measurements["data"]["platform"][0]
    CoP = platform["center_of_pressure"] * 1e-3
    return CoP

def Get_Event(c3d_file):
    """
    find event from c3d file : heel strike (HS) and toe off (TO)
    determine the indexes of the beginning and end of the cycle
    :param c3d_file: path of the c3d file (exp data)
    :return: list events
    """

    measurements = c3d(c3d_file)
    time = measurements["parameters"]["EVENT"]["TIMES"]["value"][1, :]
    labels_time = measurements["parameters"]["EVENT"]["LABELS"]["value"]
    get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]

    RHS = time[get_indexes("RHS", labels_time)]
    RTO = time[get_indexes("RTO", labels_time)]
    if len(RTO) > 1:
        RTO = max(RTO)
    else:
        RTO = RTO[0]

    return RHS, RTO

def Get_indexes(c3d_file):
    """
    find phase indexes
    indexes corresponding to the event that defines phases :
    - start : heel strike
    - 2 contacts : toes on the ground
    - heel rise : rising of the heel
    - stop stance : foot off the ground
    - stop : second heel strike
    :param c3d_file: path of the c3d file (exp data)
    :return: list of indexes
    """
    measurements = c3d(c3d_file)
    freq = measurements["parameters"]["POINT"]["RATE"]["value"][0]
    seuil = 0.04

    # get events for start and stop of the cycle
    RHS, RTO = Get_Event(c3d_file)
    idx_start = int(round(RHS[0] * freq) + 1)
    idx_stop_stance = int(round(RTO * freq) + 1)
    idx_stop = int(round(RHS[1] * freq) + 1)

    # get markers position
    markers = GetMarkers_Position(c3d_file, nb_marker=30)
    Heel = markers[:, 19, idx_start: idx_stop_stance]
    Meta1 = markers[:, 20, idx_start: idx_stop_stance]
    Meta5 = markers[:, 24, idx_start: idx_stop_stance]

    # Heel rise
    idx_heel = np.where(Heel[2, :] > seuil)
    idx_heel_rise = idx_start + int(idx_heel[0][0])

    # forefoot
    idx_Meta1 = np.where(Meta1[2, :] < seuil)
    idx_Meta5 = np.where(Meta5[2, :] < seuil)
    idx_2_contacts = idx_start + np.max([idx_Meta5[0][0], idx_Meta1[0][0]])
    return [idx_start, idx_2_contacts, idx_heel_rise, idx_stop_stance, idx_stop]


def GetTime(c3d_file):
    """
    find phase duration
    :param c3d_file: path of the c3d file (exp data)
    :return: list of phase duration
    """
    measurements = c3d(c3d_file)
    freq = measurements["parameters"]["ANALOG"]["RATE"]["value"][0]

    index = Get_indexes(c3d_file)
    phase_time = []
    for i in range(len(index) - 1):
        phase_time.append((1 / freq * (index[i + 1] - index[i] + 1)))
    return phase_time

def dispatch_data(c3d_file, data, nb_shooting):
    """
    divide and adjust data dimensions to match number of shooting point for each phase
    :param c3d_file: path of the c3d file (exp data)
    :param data: data to adjust (ex : q, markers trajectories, grf ...)
    :param nb_shooting: list of number of shooting point for each phase
    :return: list of adjusted data
    """

    index = Get_indexes(c3d_file)
    DATA = []
    for i in range(len(nb_shooting)):
        a = (index[i + 1] + 1 - index[i]) / (nb_shooting[i] + 1)
        if len(data.shape) == 3:
            if a.is_integer():
                x = data[:, :, index[i]:index[i + 1] + 1]
                DATA.append(x[:, :, 0::int(a)])
            else:
                x = data[:, :, index[i]:index[i + 1]]
                DATA.append(x[:, :, 0::int(a)])

        else:
            if a.is_integer():
                x = data[:, index[i]:index[i + 1] + 1]
                DATA.append(x[:, 0::int(a)])
            else:
                x = data[:, index[i]:index[i + 1]]
                DATA.append(x[:, 0::int(a)])
    return DATA
