import numpy as np

class data:
    @staticmethod
    def get_q(name, title):
        q = np.load("/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/squat/Data_test/" + name + "/kalman/" + title + "_mean.npy")
        return q

    @staticmethod
    def get_muscle_activation(name, title):
        activation = np.load("/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/squat/Data_test/" + name + "/EMG/" + title + "_mean.npy")
        return activation