import numpy as np

def adjust_muscle_activation(activation):
    idx = [16, 16, 16, 14, 14, 14, 18, 12, 12, 12, 10, 8, 8, 8, 0, 0, 2, 6, 4, 17, 17, 17, 15, 15, 15, 19, 13, 13, 13, 11, 9, 9, 9, 1, 1, 3, 7, 5]
    new_activation = np.array([activation[i, :]/100 for i in idx])
    new_activation[new_activation > 1] = 1.0
    return new_activation

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