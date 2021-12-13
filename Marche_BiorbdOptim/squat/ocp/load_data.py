import numpy as np
from scipy.interpolate import interp1d

def adjust_muscle_activation(activation):
    idx = [16, 16, 16, 14, 14, 14, 18, 12, 12, 12, 10, 8, 8, 8, 0, 0, 2, 6, 4, 17, 17, 17, 15, 15, 15, 19, 13, 13, 13, 11, 9, 9, 9, 1, 1, 3, 7, 5]
    new_activation = np.array([activation[i, :]/100 for i in idx])
    new_activation[new_activation > 1] = 1.0
    return new_activation

def interpolate_data(final_time, nb_shooting, data):
    t_init = np.linspace(0, final_time, data.shape[-1])
    t_node = np.linspace(0, final_time, nb_shooting + 1)
    f = interp1d(t_init, data, kind="cubic")
    return f(t_node)

class data:
    @staticmethod
    def get_q(name, title):
        q = np.load("/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/squat/Data_test/" + name + "/kalman/" + title + "_mean.npy")
        return q

    @staticmethod
    def get_muscle_activation(name, title):
        activation = np.load("/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/squat/Data_test/" + name + "/EMG/" + title + "_mean.npy")
        activation = adjust_muscle_activation(activation)
        return activation

    @staticmethod
    def get_markers_position(name, title):
        markers_position = np.load("/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/squat/Data_test/" + name + "/MARKERS/" + title + "_mean.npy")
        return markers_position

    @staticmethod
    def data_per_phase(name, title, final_time, nb_shooting, name_data):
        match name_data:
            case 'q':
                q = data.get_q(name, title)
                idx = int(q.shape[-1] / 2)
                q_ref = []
                q_ref.append(interpolate_data(final_time[0], nb_shooting[0], q[:, 1:idx + 1]))
                q_ref.append(interpolate_data(final_time[1], nb_shooting[1], q[:, idx:]))
                return q_ref

            case 'activation':
                activation = data.get_muscle_activation(name, title)
                idx = int(activation.shape[-1] / 2)
                activation_ref = []
                activation_ref.append(interpolate_data(final_time[0], nb_shooting[0], activation[:, 1:idx + 1]))
                activation_ref.append(interpolate_data(final_time[1], nb_shooting[1], activation[:, idx:]))
                return activation_ref

            case 'marker':
                marker = data.get_markers_position(name, title)
                idx = int(marker.shape[-1] / 2)
                marker_ref = []
                marker_ref.append(interpolate_data(final_time[0], nb_shooting[0], marker[:, :, 1:idx + 1]))
                marker_ref.append(interpolate_data(final_time[1], nb_shooting[1], marker[:, :, idx:]))
                return marker_ref

