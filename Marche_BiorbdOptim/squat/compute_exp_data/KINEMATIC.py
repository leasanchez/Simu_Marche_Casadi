import matplotlib.pyplot as plt
import numpy as np
import biorbd
from scipy import interpolate
from ezc3d import c3d
from MARKERS import markers

def get_q_name(model):
    q_name = []
    for q in range(model.nbSegment()):
        for d in range(model.segment(q).nbQ()):
            q_name.append(f"{model.segment(q).name().to_string()}_{model.segment(q).nameDof(d).to_string()}")
    return q_name

def load_txt_file(file_path, size):
    data_tp = np.loadtxt(file_path)
    nb_frame = int(len(data_tp) / size)
    out = np.zeros((size, nb_frame))
    for n in range(nb_frame):
        out[:, n] = data_tp[n * size: n * size + size]
    return out

def divide_squat_repetition(x, index):
    x_squat = []
    for idx in range(int(len(index)/2)):
        x_squat.append(x[:, index[2*idx]:index[2*idx + 1]])
    return x_squat

def interpolate_repetition(x):
    x_interp = np.zeros((len(x), x[0].shape[0], 200))
    for (i, e) in enumerate(x):
        start = np.arange(0, e[0].shape[0])
        interp = np.linspace(0, start[-1], 200)
        for m in range(x[0].shape[0]):
            f = interpolate.interp1d(start, e[m, :])
            x_interp[i, m, :] = f(interp)
    return x_interp

def compute_mean(x):
    x_interp = interpolate_repetition(x)
    mean = np.mean(x_interp, axis=0)
    std = np.std(x_interp, axis=0)
    return mean, std

def compute_symetry_ratio(emg):
    emg_sym = []
    for i in range(int(len(emg)/2)):
        emg_sym.append((emg[2*i] + 100)/(emg[2*i + 1] + 100))
    return emg_sym

class kinematic:
    def __init__(self, name, higher_foot='R'):
        self.name = name
        self.higher_foot = higher_foot
        self.path = '../Data_test/' + name
        self.model = biorbd.Model(self.path + "/" + name + ".bioMod")
        self.list_exp_files = ['squat_controle', 'squat_3cm', 'squat_4cm', 'squat_5cm', 'squat_controle_post']
        self.q_name = get_q_name(self.model)
        self.events = markers(self.path).get_events()
        self.q = self.get_q()
        self.q_mean, self.q_std = self.get_mean(self.q)
        self.qdot = self.get_qdot()

    def get_q(self):
        q = []
        for (i, file) in enumerate(self.list_exp_files):
            q_kalman = load_txt_file(self.path + "/kalman/" + file + "_q_KalmanFilter.txt", self.model.nbQ())
            q.append(divide_squat_repetition(q_kalman, self.events[i]))
        return q

    def get_qdot(self):
        qdot = []
        for (i, file) in enumerate(self.list_exp_files):
            qdot_kalman = load_txt_file(self.path + "/kalman/" + file + "_qdot_KalmanFilter.txt", self.model.nbQ())
            qdot.append(divide_squat_repetition(qdot_kalman, self.events[i]))
        return qdot

    def get_mean(self, data):
        mean = []
        std = []
        for (i, d) in enumerate(data):
            a = compute_mean(d)
            mean.append(a[0])
            std.append(a[1])
        return mean, std

    