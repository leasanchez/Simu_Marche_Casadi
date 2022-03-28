import numpy as np
from scipy.interpolate import interp1d
from compute_exp_data.UTILS import utils
from compute_exp_data.MARKERS import markers
from compute_exp_data.KINEMATIC import kinematic
from compute_exp_data.EMG import emg

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
    def get_q(name, higher_foot, condition):
        kin = kinematic(name, higher_foot)
        idx_condition = kin.list_exp_files.index(condition)
        return kin.q_mean[idx_condition]

    @staticmethod
    def get_muscle_activation(name, higher_foot, condition):
        e = emg(name, higher_foot)
        idx_condition = e.list_exp_files.index(condition)
        activation = e.mean[idx_condition]
        activation = adjust_muscle_activation(activation) # complete value for muscle without emg data
        return activation

    @staticmethod
    def get_markers_position(name, condition):
        mark = markers(name)
        idx_condition = mark.list_exp_files.index(condition)
        markers_position = np.array([m[idx_condition] for m in mark.mean_markers_position])
        markers_position = np.moveaxis(markers_position, 0, 1) # correct dimension for the matrix
        return markers_position

    @staticmethod
    # A MODIFIER COMMENT SONT DIVISE LES PHASES
    def data_per_phase(name, higher_foot, condition, final_time, nb_shooting, name_data):
        match name_data:
            case 'q':
                q = data.get_q(name, higher_foot, condition)
                idx = int(q.shape[-1] / 2)
                q_ref = []
                q_ref.append(interpolate_data(final_time[0], nb_shooting[0], q[:, 1:idx + 3]))
                q_ref.append(interpolate_data(final_time[1], nb_shooting[1], q[:, idx + 2:]))
                return q_ref

            case 'activation':
                activation = data.get_muscle_activation(name, higher_foot, condition)
                idx = int(activation.shape[-1] / 2)
                activation_ref = []
                activation_ref.append(interpolate_data(final_time[0], nb_shooting[0], activation[:, 1:idx + 3]))
                activation_ref.append(interpolate_data(final_time[1], nb_shooting[1], activation[:, idx + 2:]))
                return activation_ref

            case 'marker':
                marker = data.get_markers_position(name, condition)
                idx = int(marker.shape[-1] / 2)
                marker_ref = []
                marker_ref.append(interpolate_data(final_time[0], nb_shooting[0], marker[:, :, 1:idx + 3]))
                marker_ref.append(interpolate_data(final_time[1], nb_shooting[1], marker[:, :, idx + 2:]))
                return marker_ref

    @staticmethod
    def data_one_phase(name, higher_foot, condition, final_time, nb_shooting, name_data):
        match name_data:
            case 'q':
                q = data.get_q(name, higher_foot, condition)
                q_ref = interpolate_data(final_time, nb_shooting, q)
                return q_ref

            case 'activation':
                activation = data.get_muscle_activation(name, higher_foot, condition)
                activation_ref = interpolate_data(final_time, nb_shooting, activation)
                return activation_ref

            case 'marker':
                marker = data.get_markers_position(name, condition)
                marker_ref = interpolate_data(final_time, nb_shooting, marker)
                return marker_ref

