from ezc3d import c3d
import numpy as np
from scipy.interpolate import interp1d
import scipy.io as sio
from matplotlib import pyplot as plt


class Data_to_track:
    def __init__(self, name_subject, model, multiple_contact=False, two_leg=False):
        PROJET = "/home/leasanchez/programmation/Simu_Marche_Casadi/"
        self.name_subject = name_subject
        self.file = PROJET + "DonneesMouvement/" + name_subject + "_out.c3d"
        self.kalman_file = PROJET + "DonneesMouvement/" + name_subject + "_out_MOD5000_leftHanded_GenderF_Florent_.Q2"
        self.two_leg = two_leg
        self.multiple_contact = multiple_contact
        self.start_leg, self.idx_start, self.idx_stop_stance, self.idx_stop = self.Get_Event()
        self.idx_platform = self.Find_platform()
        if self.two_leg:
            self.Q_KalmanFilter_file = PROJET + "DonneesMouvement/" + name_subject + "_q_KalmanFilter_2legs.txt"
            self.Qdot_KalmanFilter_file = PROJET + "DonneesMouvement/" + name_subject + "_qdot_KalmanFilter_2legs.txt"
            self.Qddot_KalmanFilter_file = PROJET + "DonneesMouvement/" + name_subject + "_qddot_KalmanFilter_2legs.txt"
        else:
            self.Q_KalmanFilter_file = PROJET + "DonneesMouvement/" + name_subject + "_q_KalmanFilter.txt"
            self.Qdot_KalmanFilter_file = PROJET + "DonneesMouvement/" + name_subject + "_qdot_KalmanFilter.txt"
            self.Qddot_KalmanFilter_file = PROJET + "DonneesMouvement/" + name_subject + "_qddot_KalmanFilter.txt"
        self.model = model
        self.nb_marker = model.nbMarkers()
        self.nb_q = model.nbQ()
        self.nb_mus = model.nbMuscleTotal()




    def Get_Event(self):
        # Find event from c3d file : heel strike (HS) and toe off (TO)
        # Determine the indexes of the beginning and end of each phases

        measurements = c3d(self.file)
        time = measurements["parameters"]["EVENT"]["TIMES"]["value"][1, :]
        labels_time = measurements["parameters"]["EVENT"]["LABELS"]["value"]
        freq = measurements["parameters"]["POINT"]["RATE"]["value"][0]
        get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]

        if self.two_leg:
            RHS = time[get_indexes("RHS", labels_time)]
            LHS = time[get_indexes("LHS", labels_time)]
            RTO = time[get_indexes("RTO", labels_time)]
            LTO = time[get_indexes("LTO", labels_time)]

            if RHS[0]>LHS[0]:
                start_leg = 'L'
                # always start by the rigth leg
                start = [int(round(RHS[0] * freq) + 1), int(round(min(RTO) * freq) + 1)] # debute au moment ou les orteils de la jambe 2 quitte le sol
                stop_stance = [int(round(max(RTO) * freq) + 1), int(round(max(LTO) * freq) + 1)]
                stop = [int(round(RHS[1] * freq) + 1), int(round(LHS[1] * freq) + 1)]
            else:
                start_leg = 'R'
                # always start by the rigth leg
                start = [int(round(min(LTO) * freq) + 1), int(round(LHS[0] * freq) + 1)]
                stop_stance = [int(round(max(RTO) * freq) + 1), int(round(max(LTO) * freq) + 1)]
                stop = [int(round(RHS[1] * freq) + 1), int(round(LHS[1] * freq) + 1)]

        else:
            RHS = time[get_indexes("RHS", labels_time)]
            RTO = time[get_indexes("RTO", labels_time)]
            if len(RTO) > 1:
                RTO = max(RTO)
            else:
                RTO = RTO[0]
            start = int(round(RHS[0] * freq) + 1)
            stop_stance = int(round(RTO * freq) + 1)
            stop = int(round(RHS[1] * freq) + 1)
            start_leg = 'R'
        return start_leg, start, stop_stance, stop

        start = round(RHS[0] * freq) + 1
        stop_stance = round(RTO * freq) + 1
        stop = round(RHS[1] * freq) + 1
        return int(start), int(stop_stance), int(stop)

    def Find_platform(self):
        GRF = self.GetForces()
        P = np.zeros(self.nbPF)
        for p in range(self.nbPF):
            P[p] = (sum(GRF[p][2, int(self.idx_start) : int(self.idx_stop_stance) + 1]))
        idx_platform = np.where(P == P.max())[0][0]
        return idx_platform

    def GetTime(self):
        measurements = c3d(self.file)
        freq = measurements["parameters"]["ANALOG"]["RATE"]["value"][0]

        T = 1 / freq * (self.idx_stop - self.idx_start + 1)
        T_stance = 1 / freq * (self.idx_stop_stance - self.idx_start + 1)
        T_swing = 1 / freq * (self.idx_stop - self.idx_stop_stance + 1)

        if self.multiple_contact:
            T_stance = self.GetTime_stance()
        return T, T_stance, T_swing

    def GetMarkers_Position(self):
     # LOAD C3D FILE
     measurements = c3d(self.file)
     points = measurements["data"]["points"]
     labels_markers = measurements["parameters"]["POINT"]["LABELS"]["value"]

     # GET THE MARKERS POSITION (X, Y, Z) AT EACH POINT
     markers = np.zeros((3, self.nb_marker, len(points[0, 0, :])))

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

     if self.two_leg:
         # --- Left Leg ---
         # femur L markers
         markers[:, 26, :] = points[:3, labels_markers.index("L_FTC"), :] * 1e-3  # L_FTC
         markers[:, 27, :] = points[:3, labels_markers.index("L_Thigh_Top"), :] * 1e-3  # L_Thigh_Top
         markers[:, 28, :] = points[:3, labels_markers.index("L_Thigh_Down"), :] * 1e-3  # L_Thigh_Down
         markers[:, 29, :] = points[:3, labels_markers.index("L_Thigh_Front"), :] * 1e-3  # L_Thigh_Front
         markers[:, 30, :] = points[:3, labels_markers.index("L_Thigh_Back"), :] * 1e-3  # L_Thigh_Back
         markers[:, 31, :] = points[:3, labels_markers.index("L_FLE"), :] * 1e-3  # L_FLE
         markers[:, 32, :] = points[:3, labels_markers.index("L_FME"), :] * 1e-3  # L_FME
         #  tibia L markers
         markers[:, 33, :] = points[:3, labels_markers.index("L_FAX"), :] * 1e-3  # L_FAX
         markers[:, 34, :] = points[:3, labels_markers.index("L_TTC"), :] * 1e-3  # L_TTC
         markers[:, 35, :] = points[:3, labels_markers.index("L_Shank_Top"), :] * 1e-3  # L_Shank_Top
         markers[:, 36, :] = points[:3, labels_markers.index("L_Shank_Down"), :] * 1e-3  # L_Shank_Down
         markers[:, 37, :] = points[:3, labels_markers.index("L_Shank_Front"), :] * 1e-3  # L_Shank_Front
         markers[:, 38, :] = points[:3, labels_markers.index("L_Shank_Tibia"), :] * 1e-3  # L_Shank_Tibia
         markers[:, 39, :] = points[:3, labels_markers.index("L_FAL"), :] * 1e-3  # L_FAL
         markers[:, 40, :] = points[:3, labels_markers.index("L_TAM"), :] * 1e-3  # L_TAM
         #  foot L markers
         markers[:, 41, :] = points[:3, labels_markers.index("L_FCC"), :] * 1e-3  # L_FCC
         markers[:, 42, :] = points[:3, labels_markers.index("L_FM1"), :] * 1e-3  # L_FM1
         markers[:, 43, :] = points[:3, labels_markers.index("L_FMP1"), :] * 1e-3  # L_FMP1
         markers[:, 44, :] = points[:3, labels_markers.index("L_FM2"), :] * 1e-3  # L_FM2
         markers[:, 45, :] = points[:3, labels_markers.index("L_FMP2"), :] * 1e-3  # L_FMP2
         markers[:, 46, :] = points[:3, labels_markers.index("L_FM5"), :] * 1e-3  # L_FM5
         markers[:, 47, :] = points[:3, labels_markers.index("L_FMP5"), :] * 1e-3  # L_FMP5
     return markers


    def GetTime_stance(self):
        """
        Get the different times that divides the stance phase :
        1. heel strike -> 1 contact at the heel
        2. flat foot -> 2 contacts to the ground
        3. forefoot -> heel rise hence 1 contact forefoot
        """
        measurements = c3d(self.file)
        freq = measurements["parameters"]["ANALOG"]["RATE"]["value"][0]

        # get markers position
        markers = self.GetMarkers_Position()
        Heel = markers[:, 19, self.idx_start : self.idx_stop_stance]
        Meta1 = markers[:, 20, self.idx_start : self.idx_stop_stance]
        Meta5 = markers[:, 24, self.idx_start : self.idx_stop_stance]

        # Heel rise -- z > 0.02
        idx_heel = np.where(Heel[2, :] > 0.023)
        self.idx_heel_rise = self.idx_start + int(idx_heel[0][0])

        # forefoot -- z < 0.02
        idx_Meta1 = np.where(Meta1[2, :] < 0.023)
        idx_Meta5 = np.where(Meta5[2, :] < 0.023)
        self.idx_2_contacts = self.idx_start + np.max([idx_Meta5[0][0], idx_Meta1[0][0]])

        T_Heel = 1 / freq * (self.idx_2_contacts - self.idx_start + 1)
        T_2_contact = 1 / freq * (self.idx_heel_rise - self.idx_2_contacts + 1)
        T_Forefoot = 1 / freq * (self.idx_stop_stance - self.idx_heel_rise + 1)
        T_stance = [T_Heel, T_2_contact, T_Forefoot]

        # plt.figure('foot position')
        # plt.plot(Heel[2, :], 'g')
        # plt.plot(Meta1[2, :], 'r')
        # plt.plot(Meta5[2, :], 'b')
        # plt.legend(['heel', 'meta1', 'meta5'])
        # plt.title('foot markers position during stance phase')
        # plt.ylabel('z position (mm)')
        # plt.ylim([0, 0.05])
        # plt.xlim([0, len(Heel[2, :])])
        # plt.plot([0, len(Heel[2, :])], [0.023, 0.023], "k--", linewidth=0.7)
        # plt.plot([np.max([idx_Meta5[0][0], idx_Meta1[0][0]]), np.max([idx_Meta5[0][0], idx_Meta1[0][0]])], [0, 0.05], "k--", linewidth=0.7)
        # plt.plot([idx_heel[0][0], idx_heel[0][0]], [0, 0.05], "k--", linewidth=0.7)
        return T_stance

    def GetForces(self):
        measurements = c3d(self.file, extract_forceplat_data=True)
        self.nbPF = len(measurements["data"]["platform"])
        F = []
        for p in range(self.nbPF):
            platform = measurements["data"]["platform"][p]
            force = platform["force"]
            F.append(force)
        return F

    def GetMoment(self):
        measurements = c3d(self.file, extract_forceplat_data=True)
        M = []
        corners = []
        for p in range(self.nbPF):
            platform = measurements["data"]["platform"][p]
            moment = platform["moment"] * 1e-3
            c = platform["corners"] * 1e-3
            M.append(moment)
            corners.append(c)
        return M

    def GetMoment_at_CoP(self):
        measurements = c3d(self.file, extract_forceplat_data=True)
        M_CoP = []
        for p in range(self.nbPF):
            platform = measurements["data"]["platform"][p]
            moment = platform["Tz"] * 1e-3
            M_CoP.append(moment)
        return M_CoP

    def load_data_Moment(self, biorbd_model, final_time, n_shooting_points):
        # GET MOMENT
        M_real = self.GetMoment()
        M_real = M_real[self.idx_platform]

        # INTERPOLATE AND GET REAL FORCES FOR SHOOTING POINT FOR THE GAIT CYCLE PHASE
        if self.multiple_contact:
            M_ref = []
            idx = [self.idx_start, self.idx_2_contacts, self.idx_heel_rise, self.idx_stop_stance]
            for i in range(len(final_time)):
                t_stance = np.linspace(0, final_time[i], (idx[i + 1] - idx[i]) + 1)
                node_t_stance = np.linspace(0, final_time[i], n_shooting_points[i] + 1)
                f_stance = interp1d(t_stance, M_real[:, idx[i] : (idx[i + 1] + 1)], kind="cubic")
                M = f_stance(node_t_stance)
                M_ref.append(M)
        else:
            t = np.linspace(0, final_time, (self.idx_stop_stance - self.idx_start + 1))
            node_t = np.linspace(0, final_time, n_shooting_points + 1)
            f = interp1d(t, M_real[:, self.idx_start: (self.idx_stop_stance + 1)], kind="cubic")
            M_ref = f(node_t)
        return M_ref

    def load_data_Moment_at_CoP(self, biorbd_model, final_time, n_shooting_points):
        # GET MOMENT
        M_real = self.GetMoment_at_CoP()
        M_real = M_real[self.idx_platform]
        M_real = np.nan_to_num(M_real)

        # INTERPOLATE AND GET REAL FORCES FOR SHOOTING POINT FOR THE GAIT CYCLE PHASE
        if self.multiple_contact:
            M_CoP = []
            idx = [self.idx_start, self.idx_2_contacts, self.idx_heel_rise, self.idx_stop_stance]
            for i in range(len(final_time)):
                t_stance = np.linspace(0, final_time[i], (idx[i + 1] - idx[i]) + 1)
                node_t_stance = np.linspace(0, final_time[i], n_shooting_points[i] + 1)
                f_stance = interp1d(t_stance, M_real[:, idx[i] : (idx[i + 1] + 1)], kind="cubic")
                M = f_stance(node_t_stance)
                M_CoP.append(M)
        else:
            t = np.linspace(0, final_time, (self.idx_stop_stance - self.idx_start + 1))
            node_t = np.linspace(0, final_time, n_shooting_points + 1)
            f = interp1d(t, M_real[:, self.idx_start : (self.idx_stop_stance + 1)], kind="cubic")
            M_CoP = f(node_t)
        return M_CoP

    def ComputeCoP(self):
        measurements = c3d(self.file, extract_forceplat_data=True)
        CoP = []
        corners = []
        for p in range(self.nbPF):
            platform = measurements["data"]["platform"][p]
            cop = platform["center_of_pressure"] * 1e-3
            corner = platform["corners"] * 1e-3
            CoP.append(cop)
            corners.append(corner)

        # plt.figure('CoP')
        # markers = self.GetMarkers_Position()
        # plt.plot(corners[0][0, :], corners[0][1, :], 'k+')
        # plt.plot(corners[0][0, 2] + (corners[0][0, 1] - corners[0][0, 2]) / 2, (corners[0][1, 0] - corners[0][1, 1]) / 2, 'k+')
        # plt.plot(CoP[0][0, self.idx_start: self.idx_2_contacts],
        #          CoP[0][1, self.idx_start: self.idx_2_contacts], "g+")
        # plt.plot(CoP[0][0, self.idx_2_contacts: self.idx_heel_rise],
        #          CoP[0][1, self.idx_2_contacts: self.idx_heel_rise], "b+")
        # plt.plot(CoP[0][0, self.idx_heel_rise: self.idx_stop_stance - 1],
        #          CoP[0][1, self.idx_heel_rise: self.idx_stop_stance - 1], "r+")
        #
        # plt.plot(np.mean(markers[0, 19, self.idx_start:self.idx_heel_rise]), np.mean(markers[1, 19, self.idx_start:self.idx_heel_rise]), 'go')
        # plt.plot(np.mean(markers[0, 20, self.idx_2_contacts:self.idx_stop_stance]),
        #          np.mean(markers[1, 20, self.idx_2_contacts:self.idx_stop_stance]), 'ro')
        # plt.plot(np.mean(markers[0, 25, self.idx_2_contacts:self.idx_stop_stance]),
        #          np.mean(markers[1, 25, self.idx_2_contacts:self.idx_stop_stance]), 'bo')
        #
        # plt.title("CoP evolution during stance phase")
        # plt.xlabel("x (m)")
        # plt.ylabel("y (m)")
        return CoP

    def load_data_CoP(self, biorbd_model, final_time, n_shooting_points):
        # GET MOMENT
        CoP_real = self.ComputeCoP()
        CoP_real = CoP_real[self.idx_platform]
        CoP_real = np.nan_to_num(CoP_real)

        # INTERPOLATE AND GET REAL FORCES FOR SHOOTING POINT FOR THE GAIT CYCLE PHASE
        if self.multiple_contact:
            CoP = []
            idx = [self.idx_start, self.idx_2_contacts, self.idx_heel_rise, self.idx_stop_stance]
            for i in range(len(final_time)):
                t_stance = np.linspace(0, final_time[i], (idx[i + 1] - idx[i]) + 1)
                node_t_stance = np.linspace(0, final_time[i], n_shooting_points[i] + 1)
                f_stance = interp1d(t_stance, CoP_real[:, idx[i] : (idx[i + 1] + 1)], kind="cubic")
                cop = f_stance(node_t_stance)
                CoP.append(cop)
        else:
            t = np.linspace(0, final_time, (self.idx_stop_stance - self.idx_start + 1))
            node_t = np.linspace(0, final_time, n_shooting_points + 1)
            f = interp1d(t, CoP_real[:, self.idx_start : (self.idx_stop_stance + 1)], kind="cubic")
            CoP = f(node_t)
        return CoP

    def load_data_markers(self, biorbd_model, final_time, n_shooting_points, GaitPhase):
        markers = self.GetMarkers_Position()

        # INTERPOLATE AND GET REAL POSITION FOR SHOOTING POINT FOR THE SWING PHASE
        if GaitPhase == "stance":
            if self.multiple_contact:
                markers_ref = []
                idx = [self.idx_start, self.idx_2_contacts, self.idx_heel_rise, self.idx_stop_stance]
                for i in range(len(final_time)):
                    t_stance = np.linspace(0, final_time[i], (idx[i + 1] - idx[i]) + 1)
                    node_t_stance = np.linspace(0, final_time[i], n_shooting_points[i] + 1)
                    f_stance = interp1d(t_stance, markers[:, :, idx[i] : (idx[i + 1] + 1)], kind="cubic")
                    markers_ref.append(f_stance(node_t_stance))
            else:
                t = np.linspace(0, final_time, (self.idx_stop_stance - self.idx_start + 1))
                node_t = np.linspace(0, final_time, n_shooting_points + 1)
                f = interp1d(t, markers[:, :, self.idx_start : (self.idx_stop_stance + 1)], kind="cubic")
                markers_ref = f(node_t)

        elif GaitPhase == "swing":
            t = np.linspace(0, final_time, (self.idx_stop - self.idx_stop_stance + 1))
            node_t = np.linspace(0, final_time, n_shooting_points + 1)
            f = interp1d(t, markers[:, :, self.idx_stop_stance : (self.idx_stop + 1)], kind="cubic")
            markers_ref = f(node_t)
        else:
            raise RuntimeError("Gaitphase doesn't exist")
        return markers_ref

    def load_q_kalman(self, biorbd_model, final_time, n_shooting_points, GaitPhase):
        Q = np.loadtxt(self.Q_KalmanFilter_file)
        nb_q = biorbd_model.nbQ()
        nb_frame = int(len(Q) / nb_q)
        q_init = np.zeros((nb_q, nb_frame))
        for n in range(nb_frame):
            q_init[:, n] = Q[n * nb_q : n * nb_q + nb_q]

        # INTERPOLATE AND GET KALMAN JOINT POSITION FOR SHOOTING POINT FOR THE CYCLE PHASE
        if GaitPhase == "stance":
            if self.multiple_contact:
                q_ref = []
                idx = [self.idx_start, self.idx_2_contacts, self.idx_heel_rise, self.idx_stop_stance]
                for i in range(len(final_time)):
                    t_stance = np.linspace(0, final_time[i], (idx[i + 1] - idx[i]) + 1)
                    node_t_stance = np.linspace(0, final_time[i], n_shooting_points[i] + 1)
                    f_stance = interp1d(t_stance, q_init[:, idx[i] : (idx[i + 1] + 1)], kind="cubic")
                    q_ref.append(f_stance(node_t_stance))
            else:
                t = np.linspace(0, final_time, (self.idx_stop_stance - self.idx_start + 1))
                node_t = np.linspace(0, final_time, n_shooting_points + 1)
                f = interp1d(t, q_init[:, self.idx_start : (self.idx_stop_stance + 1)], kind="cubic")
                q_ref = f(node_t)
        elif GaitPhase == "swing":
            t = np.linspace(0, final_time, (self.idx_stop - self.idx_stop_stance) + 1)
            node_t = np.linspace(0, final_time, n_shooting_points + 1)
            f = interp1d(t, q_init[:, self.idx_stop_stance : (self.idx_stop + 1)], kind="cubic")
            q_ref = f(node_t)
        else:
            raise RuntimeError("Gaitphase doesn't exist")
        return q_ref

    def load_qdot_kalman(self, biorbd_model, final_time, n_shooting_points, GaitPhase):
        Q = np.loadtxt(self.Qdot_KalmanFilter_file)
        nb_q = biorbd_model.nbQ()
        nb_frame = int(len(Q) / nb_q)
        qdot_init = np.zeros((nb_q, nb_frame))
        for n in range(nb_frame):
            qdot_init[:, n] = Q[n * nb_q : n * nb_q + nb_q]

        # INTERPOLATE AND GET KALMAN JOINT POSITION FOR SHOOTING POINT FOR THE CYCLE PHASE
        if GaitPhase == "stance":
            if self.multiple_contact:
                qdot_ref = []
                idx = [self.idx_start, self.idx_2_contacts, self.idx_heel_rise, self.idx_stop_stance]
                for i in range(len(final_time)):
                    t_stance = np.linspace(0, final_time[i], (idx[i + 1] - idx[i]) + 1)
                    node_t_stance = np.linspace(0, final_time[i], n_shooting_points[i] + 1)
                    f_stance = interp1d(t_stance, qdot_init[:, idx[i] : (idx[i + 1] + 1)], kind="cubic")
                    qdot_ref.append(f_stance(node_t_stance))
            else:
                t = np.linspace(0, final_time, (self.idx_stop_stance - self.idx_start + 1))
                node_t = np.linspace(0, final_time, n_shooting_points + 1)
                f = interp1d(t, qdot_init[:, self.idx_start : (self.idx_stop_stance + 1)], kind="cubic")
                qdot_ref = f(node_t)
        elif GaitPhase == "swing":
            t = np.linspace(0, final_time, (self.idx_stop - self.idx_stop_stance) + 1)
            node_t = np.linspace(0, final_time, n_shooting_points + 1)
            f = interp1d(t, qdot_init[:, self.idx_stop_stance : (self.idx_stop + 1)], kind="cubic")
            qdot_ref = f(node_t)
        else:
            raise RuntimeError("Gaitphase doesn't exist")
        return qdot_ref

    def load_data_q(self, biorbd_model, final_time, n_shooting_points, GaitPhase):
        # Create initial vector for joint position (nbNoeuds x nbQ)
        # Based on Kalman filter??

        # # LOAD MAT FILE FOR GENERALIZED COORDINATES
        kalman = sio.loadmat(self.kalman_file)
        Q_real = kalman["Q2"]

        # INTERPOLATE AND GET KALMAN JOINT POSITION FOR SHOOTING POINT FOR THE CYCLE PHASE
        if GaitPhase == "stance":
            if self.multiple_contact:
                q_ref = []
                idx = [self.idx_start, self.idx_2_contacts, self.idx_heel_rise, self.idx_stop_stance]
                for i in range(len(final_time)):
                    t_stance = np.linspace(0, final_time[i], (idx[i + 1] - idx[i]) + 1)
                    node_t_stance = np.linspace(0, final_time[i], n_shooting_points[i] + 1)
                    f_stance = interp1d(t_stance, Q_real[:, idx[i] : (idx[i + 1] + 1)], kind="cubic")
                    q_ref.append(f_stance(node_t_stance))
            else:
                t = np.linspace(0, final_time, (self.idx_stop_stance - self.idx_start + 1))
                node_t = np.linspace(0, final_time, n_shooting_points + 1)
                f = interp1d(t, Q_real[:, self.idx_start : (self.idx_stop_stance + 1)], kind="cubic")
                q_ref = f(node_t)
        elif GaitPhase == "swing":
            t = np.linspace(0, final_time, (self.idx_stop - self.idx_stop_stance) + 1)
            node_t = np.linspace(0, final_time, n_shooting_points + 1)
            f = interp1d(t, Q_real[:, self.idx_stop_stance : (self.idx_stop + 1)], kind="cubic")
            q_ref = f(node_t)
        else:
            raise RuntimeError("Gaitphase doesn't exist")
        return q_ref

    def load_data_emg(self, biorbd_model, final_time, n_shooting_points, GaitPhase):
        # Load c3d file and get the muscular excitation from emg
        nbMuscle = biorbd_model.nbMuscleTotal()

        # LOAD C3D FILE
        measurements = c3d(self.file)
        points = measurements["data"]["points"]
        labels_points = measurements["parameters"]["POINT"]["LABELS"]["value"]

        # GET THE TIME OF TOE OFF & HEEL STRIKE
        [start, stop_stance, stop] = self.Get_Event()

        # GET THE MUSCULAR EXCITATION FROM EMG (NOT ALL MUSCLES)
        EMG = np.zeros(((nbMuscle - 7), len(points[0, 0, :])))

        EMG[9, :] = points[0, labels_points.index("R_Tibialis_Anterior"), :].squeeze()  # R_Tibialis_Anterior
        EMG[8, :] = points[0, labels_points.index("R_Soleus"), :].squeeze()  # R_Soleus
        EMG[7, :] = points[
            0, labels_points.index("R_Gastrocnemius_Lateralis"), :
        ].squeeze()  # R_Gastrocnemius_Lateralis
        EMG[6, :] = points[0, labels_points.index("R_Gastrocnemius_Medialis"), :].squeeze()  # R_Gastrocnemius_Medialis
        EMG[5, :] = points[0, labels_points.index("R_Vastus_Medialis"), :].squeeze()  # R_Vastus_Medialis
        EMG[4, :] = points[0, labels_points.index("R_Rectus_Femoris"), :].squeeze()  # R_Rectus_Femoris
        EMG[3, :] = points[0, labels_points.index("R_Biceps_Femoris"), :].squeeze()  # R_Biceps_Femoris
        EMG[2, :] = points[0, labels_points.index("R_Semitendinosus"), :].squeeze()  # R_Semitendinous
        EMG[1, :] = points[0, labels_points.index("R_Gluteus_Medius"), :].squeeze()  # R_Gluteus_Medius
        EMG[0, :] = points[0, labels_points.index("R_Gluteus_Maximus"), :].squeeze()  # R_Gluteus_Maximus

        # INTERPOLATE AND GET REAL MUSCULAR EXCITATION FOR SHOOTING POINT FOR THE GAIT CYCLE PHASE
        if GaitPhase == "stance":
            if self.multiple_contact:
                emg_ref = []
                idx = [self.idx_start, self.idx_2_contacts, self.idx_heel_rise, self.idx_stop_stance]
                for i in range(len(final_time)):
                    t_stance = np.linspace(0, final_time[i], (idx[i + 1] - idx[i]) + 1)
                    node_t_stance = np.linspace(0, final_time[i], n_shooting_points[i] + 1)
                    f_stance = interp1d(t_stance, EMG[:, idx[i] : (idx[i + 1] + 1)], kind="cubic")
                    emg_ref.append(f_stance(node_t_stance))

                    # RECTIFY EMG VALUES BETWEEN 0 & 1
                    emg_ref[i][emg_ref[i] < 0] = 1e-3
                    emg_ref[i][emg_ref[i] == 0] = 1e-3
                    emg_ref[i][emg_ref[i] > 1] = 1
            else:
                t = np.linspace(0, final_time, int(stop_stance - start) + 1)
                node_t = np.linspace(0, final_time, n_shooting_points + 1)
                f = interp1d(t, EMG[:, int(start) : int(stop_stance) + 1], kind="cubic")
                emg_ref = f(node_t)

                # RECTIFY EMG VALUES BETWEEN 0 & 1
                emg_ref[emg_ref < 0] = 1e-3
                emg_ref[emg_ref == 0] = 1e-3
                emg_ref[emg_ref > 1] = 1

        elif GaitPhase == "swing":
            t = np.linspace(0, final_time, int(stop - stop_stance) + 1)
            node_t = np.linspace(0, final_time, n_shooting_points + 1)
            f = interp1d(t, EMG[:, int(stop_stance) : int(stop) + 1], kind="cubic")
            emg_ref = f(node_t)

            # RECTIFY EMG VALUES BETWEEN 0 & 1
            emg_ref[emg_ref < 0] = 1e-3
            emg_ref[emg_ref == 0] = 1e-3
            emg_ref[emg_ref > 1] = 1
        else:
            raise RuntimeError("Gaitphase doesn't exist")

        return emg_ref

    def load_data_GRF(self, biorbd_model, final_time, n_shooting_points):
        # Load c3d file and get the muscular excitation from emg

        # GET GROUND REACTION WRENCHES
        GRF = self.GetForces()
        GRF = GRF[self.idx_platform]

        # INTERPOLATE AND GET REAL FORCES FOR SHOOTING POINT FOR THE GAIT CYCLE PHASE
        if self.multiple_contact:
            GRF_real = []
            idx = [self.idx_start, self.idx_2_contacts, self.idx_heel_rise, self.idx_stop_stance]
            for i in range(len(final_time)):
                t_stance = np.linspace(0, final_time[i], (idx[i + 1] - idx[i]) + 1)
                node_t_stance = np.linspace(0, final_time[i], n_shooting_points[i] + 1)
                f_stance = interp1d(t_stance, GRF[:, idx[i] : (idx[i + 1] + 1)], kind="cubic")
                G = f_stance(node_t_stance)
                GRF_real.append(G)
        else:
            t_stance = np.linspace(0, final_time, (self.idx_stop_stance - self.idx_start) + 1)
            node_t_stance = np.linspace(0, final_time, n_shooting_points + 1)
            f_stance = interp1d(t_stance, GRF[:, self.idx_start : (self.idx_stop_stance + 1)], kind="cubic")
            G = f_stance(node_t_stance)
            GRF_real = G
        return GRF_real

    def load_muscularExcitation(self, emg_ref):
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
