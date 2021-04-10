from ezc3d import c3d
import numpy as np
from scipy.interpolate import interp1d
from casadi import MX, Function


def markers_func_casadi(model):
    symbolic_q = MX.sym("q", model.nbQ(), 1)
    markers_func = []
    for m in range(model.nbMarkers()):
        markers_func.append(Function(
            "ForwardKin",
            [symbolic_q], [model.marker(symbolic_q, m).to_mx()],
            ["q"],
            ["markers"],
        ).expand())
    return markers_func

class C3dData:
    def __init__(self, file_path):
        self.c3d = c3d(file_path, extract_forceplat_data=True)
        self.marker_names = [
            "L_IAS",
            "L_IPS",
            "R_IPS",
            "R_IAS",
            "R_FTC",
            "R_Thigh_Top",
            "R_Thigh_Down",
            "R_Thigh_Front",
            "R_Thigh_Back",
            "R_FLE",
            "R_FME",
            "R_FAX",
            "R_TTC",
            "R_Shank_Top",
            "R_Shank_Down",
            "R_Shank_Front",
            "R_Shank_Tibia",
            "R_FAL",
            "R_TAM",
            "R_FCC",
            "R_FM1",
            "R_FMP1",
            "R_FM2",
            "R_FMP2",
            "R_FM5",
            "R_FMP5",
        ]
        self.muscle_names = [
            "R_Tibialis_Anterior",
            "R_Soleus",
            "R_Gastrocnemius_Lateralis",
            "R_Gastrocnemius_Medialis",
            "R_Vastus_Medialis",
            "R_Rectus_Femoris",
            "R_Biceps_Femoris",
            "R_Semitendinosus",
            "R_Gluteus_Medius",
            "R_Gluteus_Maximus",
        ]

        self.trajectories = self.get_marker_trajectories(self.c3d, self.marker_names)
        self.forces = self.get_forces(self.c3d)
        self.moments = self.get_moment(self.c3d)
        self.cop = self.get_cop(self.c3d)
        self.emg = self.get_emg(self.c3d, self.muscle_names)
        self.indices = self.get_indices()
        self.phase_time = self.get_time()

    @staticmethod
    def get_marker_trajectories(loaded_c3d, marker_names):
        """
        get markers trajectories
        """

        # LOAD C3D FILE
        points = loaded_c3d["data"]["points"]
        labels_markers = loaded_c3d["parameters"]["POINT"]["LABELS"]["value"]

        # GET THE MARKERS POSITION (X, Y, Z) AT EACH POINT
        markers = np.zeros((3, len(marker_names), len(points[0, 0, :])))

        # pelvis markers
        for i, name in enumerate(marker_names):
            markers[:, i, :] = points[:3, labels_markers.index(name), :] * 1e-3
        return markers

    @staticmethod
    def get_emg(loaded_c3d, muscle_names):
        points = loaded_c3d["data"]["points"]
        labels_muscles= loaded_c3d["parameters"]["POINT"]["LABELS"]["value"]

        emg = np.zeros((len(muscle_names), len(points[0, 0, :])))

        for (i, name) in enumerate(muscle_names):
            emg[i, :] = points[0,labels_muscles.index(name), :].squeeze()
        return emg

    @staticmethod
    def get_forces(loaded_c3d):
        """
        get ground reaction forces from force platform
        """
        platform = loaded_c3d["data"]["platform"][0]
        return platform["force"]

    @staticmethod
    def get_moment(loaded_c3d):
        """
        get moments value expressed at the center of pression
        from force platform
        """
        platform = loaded_c3d["data"]["platform"][0]
        return platform["Tz"] * 1e-3

    @staticmethod
    def get_cop(loaded_c3d):
        """
        get the trajectory of the center of pressure (cop)
        from force platform
        """
        platform = loaded_c3d["data"]["platform"][0]
        return platform["center_of_pressure"] * 1e-3

    @staticmethod
    def get_event_rhs_rto(loaded_c3d):
        """
        find event from c3d file : heel strike (HS) and toe off (TO)
        determine the indexes of the beginning and end of the cycle
        """

        time = loaded_c3d["parameters"]["EVENT"]["TIMES"]["value"][1, :]
        labels_time = loaded_c3d["parameters"]["EVENT"]["LABELS"]["value"]

        def get_indices(name, time):
            return [i for (y, i) in zip(time, range(len(time))) if name == y]

        rhs = time[get_indices("RHS", labels_time)]
        rto = time[get_indices("RTO", labels_time)]
        if len(rto) > 1:
            rto = max(rto)
        else:
            rto = rto[0]

        return rhs, rto

    @staticmethod
    def get_rhs_rto_from_forces(forces):
        """
        find heel strike (HS) and toe off (TO) from ground reaction forces
        determine the indexes of the beginning and end of the cycle
        """
        idx = np.where(forces[2, :] > 5)
        return idx[0][0], idx[0][-1]

    def get_indices(self):
        """
        find phase indexes
        indexes corresponding to the event that defines phases :
        - start : heel strike
        - 2 contacts : toes on the ground
        - heel rise : rising of the heel
        - stop stance : foot off the ground
        - stop : second heel strike
        """
        freq = self.c3d["parameters"]["POINT"]["RATE"]["value"][0]
        threshold = 0.04

        # get events for start and stop of the cycle
        rhs, rto = C3dData.get_event_rhs_rto(self.c3d)
        idx_start = int(round(rhs[0] * freq) + 1)
        idx_stop_stance = int(round(rto * freq) + 1)
        idx_stop = int(round(rhs[1] * freq) + 1)

        # get markers position
        markers = C3dData.get_marker_trajectories(self.c3d, self.marker_names)
        heel = markers[:, 19, idx_start:idx_stop_stance]
        meta1 = markers[:, 20, idx_start:idx_stop_stance]
        meta5 = markers[:, 24, idx_start:idx_stop_stance]

        # Heel rise
        idx_heel = np.where(heel[2, :] > threshold)
        idx_heel_rise = idx_start + int(idx_heel[0][0])

        # forefoot
        idx_meta1 = np.where(meta1[2, :] < threshold)
        idx_meta5 = np.where(meta5[2, :] < threshold)
        idx_2_contacts = idx_start + np.max([idx_meta5[0][0], idx_meta1[0][0]])
        return [idx_start, idx_2_contacts, idx_heel_rise, idx_stop_stance, idx_stop]

    def get_indices_four_phases(self):
        """
        find phase indexes
        indexes corresponding to the event that defines phases :
        - start : heel strike
        - 2 contacts : toes on the ground
        - heel rise : rising of the heel
        - stop stance : foot off the ground
        - stop : second heel strike
        """
        freq = self.c3d["parameters"]["POINT"]["RATE"]["value"][0]
        threshold = 0.025

        # get events for start and stop of the cycle
        rhs, rto = C3dData.get_event_rhs_rto(self.c3d)
        idx_start = int(round(rhs[0] * freq) + 1)
        idx_stop_stance = int(round(rto * freq) + 1)
        idx_stop = int(round(rhs[1] * freq) + 1)

        # get markers position
        markers = C3dData.get_marker_trajectories(self.c3d, self.marker_names)
        heel = markers[:, 19, idx_start:idx_stop_stance]
        meta1 = markers[:, 20, idx_start:idx_stop_stance]
        meta5 = markers[:, 24, idx_start:idx_stop_stance]

        # Heel rise
        idx_heel = np.where(heel[2, :] > threshold)
        idx_heel_rise = idx_start + int(idx_heel[0][0])

        # forefoot
        idx_meta1 = np.where(meta1[2, :] < threshold)
        idx_meta5 = np.where(meta5[2, :] < threshold)
        idx_2_contacts = idx_start + np.min([idx_meta5[0][0], idx_meta1[0][0]])

        # toe
        idx_meta1 = np.where(meta1[2, :] < threshold)
        idx_meta5 = np.where(meta5[2, :] < threshold)
        idx_toe = idx_start + np.max([idx_meta5[0][-1], idx_meta1[0][-1]])
        return [idx_start, idx_2_contacts, idx_heel_rise, idx_toe, idx_stop_stance, idx_stop]


    def get_indices_from_forces(self):
        """
        find phase indexes
        indexes corresponding to the event that defines phases :
        - start : heel strike
        - 2 contacts : toes on the ground
        - heel rise : rising of the heel
        - stop stance : foot off the ground
        - stop : second heel strike
        """
        freq = self.c3d["parameters"]["POINT"]["RATE"]["value"][0]
        threshold = 0.04

        # get events for start and stop of the cycle
        rhs, rto = C3dData.get_event_rhs_rto(self.c3d)
        idx_start, idx_stop_stance = C3dData.get_rhs_rto_from_forces(self.forces)
        idx_stop = int(round(rhs[1] * freq) + 1)

        # get markers position
        markers = C3dData.get_marker_trajectories(self.c3d, self.marker_names)
        heel = markers[:, 19, idx_start:idx_stop_stance]
        meta1 = markers[:, 20, idx_start:idx_stop_stance]
        meta5 = markers[:, 24, idx_start:idx_stop_stance]

        # Heel rise
        idx_heel = np.where(heel[2, :] > threshold)
        idx_heel_rise = idx_start + int(idx_heel[0][0])

        # forefoot
        idx_meta1 = np.where(meta1[2, :] < threshold)
        idx_meta5 = np.where(meta5[2, :] < threshold)
        idx_2_contacts = idx_start + np.max([idx_meta5[0][0], idx_meta1[0][0]])
        return [idx_start, idx_2_contacts, idx_heel_rise, idx_stop_stance, idx_stop]

    def get_time(self):
        """
        find phase duration
        """
        freq = self.c3d["parameters"]["ANALOG"]["RATE"]["value"][0]

        index = self.indices
        phase_time = []
        for i in range(len(index) - 1):
            phase_time.append((1 / freq * (index[i + 1] - index[i] + 1)))
        return phase_time

    def get_time_from_forces(self):
        """
        find phase duration
        """
        freq = self.c3d["parameters"]["ANALOG"]["RATE"]["value"][0]

        index = self.get_indices_from_forces()
        phase_time = []
        for i in range(len(index) - 1):
            phase_time.append((1 / freq * (index[i + 1] - index[i])))
        return phase_time


class LoadData:
    def __init__(self, model, c3d_file, q_file, qdot_file, dt, interpolation=False):
        def load_txt_file(file_path, size):
            data_tp = np.loadtxt(file_path)
            nb_frame = int(len(data_tp) / size)
            out = np.zeros((size, nb_frame))
            for n in range(nb_frame):
                out[:, n] = data_tp[n * size : n * size + size]
            return out

        self.model = model
        self.nb_q = model.nbQ()
        self.nb_qdot = model.nbQdot()
        self.nb_markers = model.nbMarkers()
        self.nb_mus = model.nbMuscleTotal()

        # files path
        self.c3d_data = C3dData(c3d_file)
        self.q = load_txt_file(q_file, self.nb_q)
        self.qdot = load_txt_file(qdot_file, self.nb_qdot)
        self.emg = self.dispatch_muscle_activation(self.c3d_data.emg)

        # dispatch data
        self.dt = dt
        self.phase_time = self.c3d_data.get_time()
        self.number_shooting_points = self.get_shooting_numbers()
        if interpolation:
            self.q_ref = self.dispatch_data_interpolation(data=self.q)
            self.qdot_ref = self.dispatch_data_interpolation(data=self.qdot)
            self.markers_ref = self.dispatch_data_interpolation(data=self.c3d_data.trajectories)
            self.grf_ref = self.dispatch_data_interpolation(data=self.c3d_data.forces)
            self.moments_ref = self.dispatch_data_interpolation(data=self.c3d_data.moments)
            self.cop_ref = self.dispatch_data_interpolation(data=self.c3d_data.cop)
            self.excitation_ref = self.dispatch_data_interpolation(data=self.emg)
        else:
            self.q_ref = self.dispatch_data(data=self.q)
            self.qdot_ref = self.dispatch_data(data=self.qdot)
            self.markers_ref = self.dispatch_data(data=self.c3d_data.trajectories)
            self.grf_ref = self.dispatch_data(data=self.c3d_data.forces)
            self.moments_ref = self.dispatch_data(data=self.c3d_data.moments)
            self.cop_ref = self.dispatch_data(data=self.c3d_data.cop)
            self.excitation_ref = self.dispatch_data(data=self.emg)

    def dispatch_data(self, data):
        """
        divide and adjust data dimensions to match number of shooting point for each phase
        """

        index = self.c3d_data.get_indices()
        out = []
        for i in range(len(self.number_shooting_points)):
            a = (index[i + 1] + 1 - index[i]) / (self.number_shooting_points[i] + 1)
            if len(data.shape) == 3:
                if a.is_integer():
                    x = data[:, :, index[i] : index[i + 1] + 1]
                    out.append(x[:, :, 0 :: int(a)])
                else:
                    x = data[:, :, index[i] : index[i + 1]]
                    out.append(x[:, :, 0 :: int(a)])

            else:
                if a.is_integer():
                    x = data[:, index[i] : index[i + 1] + 1]
                    out.append(x[:, 0 :: int(a)])
                else:
                    x = data[:, index[i] : index[i + 1]]
                    out.append(x[:, 0 :: int(a)])
        return out

    def dispatch_data_interpolation(self, data):
        """
        divide and adjust data dimensions to match number of shooting point for each phase
        """
        index = self.c3d_data.indices
        out = []
        for (i, time) in enumerate(self.phase_time):
            t = np.linspace(0, time, (index[i + 1] - index[i]) + 1)
            node_t = np.linspace(0, time, self.number_shooting_points[i] + 1)
            if len(data.shape)==3:
                f = interp1d(t, data[:, :, index[i]: (index[i + 1] + 1)], kind="cubic")
            else:
                d = data[:, index[i]: (index[i + 1] + 1)]
                d[np.isnan(d)] = 0.0
                f = interp1d(t, d, kind="cubic")
            out.append(f(node_t))
        return out

    def get_indices_from_kalman(self):
        indices = self.c3d_data.get_indices()
        markers_func = markers_func_casadi(self.model)
        marker_pos = np.empty((3, self.model.nbMarkers(), self.q.shape[1]))
        for m in range(self.model.nbMarkers()):
            for n in range(self.q.shape[1]):
                marker_pos[:, m, n:n + 1] = markers_func[m](self.q[:, n])
        from matplotlib import pyplot as plt
        # plt.figure()
        # plt.plot(marker_pos[2, -1, indices[0]: indices[-1] + 1], "g")
        # plt.plot(marker_pos[2, -2, indices[0]: indices[-1] + 1], "r")
        # plt.plot(marker_pos[2, 24, indices[0]: indices[-1] + 1], "r--")
        # plt.plot(marker_pos[2, -3, indices[0]: indices[-1] + 1], "b")
        # plt.plot(marker_pos[2, 20, indices[0]: indices[-1] + 1], "b--")
        # plt.plot(marker_pos[2, 19, indices[0]: indices[-1] + 1], "k--")
        # for idx in indices:
        #     plt.plot([idx - indices[0], idx- indices[0]], [0.0, 0.15], "k--")
        #
        # plt.figure()
        # plt.plot(self.c3d_data.forces[2, indices[0]: indices[-1] + 1], "b")
        # for idx in indices:
        #     plt.plot([idx - indices[0], idx- indices[0]], [0.0, 820], "k--")
        # indices.append(2)
        return indices

    def dispatch_muscle_activation(self, data):
        excitation_ref = np.zeros((self.nb_mus, data.shape[1]))

        excitation_ref[0, :] = data[9, :]  # glut_max1_r
        excitation_ref[1, :] = data[9, :]  # glut_max2_r
        excitation_ref[2, :] = data[9, :]  # glut_max3_r
        excitation_ref[3, :] = data[8, :]  # glut_med1_r
        excitation_ref[4, :] = data[8, :]  # glut_med2_r
        excitation_ref[5, :] = data[8, :]  # glut_med3_r
        excitation_ref[6, :] = np.zeros(data.shape[1])  # iliacus_r
        excitation_ref[7, :] = np.zeros(data.shape[1])  # psoas_r
        excitation_ref[8, :] = np.zeros(data.shape[1])  # semimem_r
        excitation_ref[9, :] = data[7, :]  # semiten_r
        excitation_ref[10, :] = data[6, :]  # bi_fem_r
        excitation_ref[11, :] = data[5, :]  # rectus_fem_r
        excitation_ref[12, :] = data[4, :]  # vas_med_r
        excitation_ref[13, :] = np.zeros(data.shape[1])  # vas_int_r
        excitation_ref[14, :] = np.zeros(data.shape[1])  # vas_lat_r
        excitation_ref[15, :] = data[3, :]  # gas_med_r
        excitation_ref[16, :] = data[2, :]  # gas_lat_r
        excitation_ref[17, :] = data[1, :]  # soleus_r
        excitation_ref[18, :] = data[0, :]  # tib_ant_r
        return excitation_ref

    def get_shooting_numbers(self):
        number_shooting_points = []
        for time in self.phase_time:
            number_shooting_points.append(int(time / self.dt))
        return number_shooting_points