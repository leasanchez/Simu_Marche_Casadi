import matplotlib.pyplot as plt
import numpy as np
import biorbd_casadi as biorbd
from casadi import MX
from scipy import interpolate
from .MARKERS import markers
from .UTILS import utils

def find_pic(data, index):
    pic=[np.max(np.abs(d[index, :])) * 180/np.pi for (i, d) in enumerate(data)]
    return np.mean(np.array(pic))

def find_amplitude(data, index):
    amp=[(np.max(np.abs(d[index, :])) - np.min(np.abs(d[index, :]))) * 180/np.pi for (i, d) in enumerate(data)]
    return np.mean(np.array(amp))

def compute_speed(com_z, time):
    com_displacement = np.max(com_z) - np.min(com_z)
    speed = 2 * com_displacement/time[0]
    speed_fall = com_displacement/time[1]
    speed_climb = com_displacement/time[2]
    return [speed, speed_fall, speed_climb]

def find_indices(com_position, com_anato):
    com_init = np.mean(com_anato[2, 100:500])
    norm_com = com_position[2, :] / com_init
    index_anato = np.where(norm_com > 0.99)[0]
    discontinuities_idx = np.where(np.gradient(index_anato) > 1)[0]

    indices = index_anato[discontinuities_idx]
    a = []
    for i in range(int(len(discontinuities_idx)/2)):
        min = np.min(norm_com[indices[2*i] : indices[2*i+1]])
        if min < 0.64:
            a.append(indices[2*i])
            a.append(indices[2*i+1])

    # plt.figure()
    # plt.plot(com_position[2, :]/com_init)
    # plt.plot([0, len(com_position[2, :])], [1.0, 1.0], 'k--')
    # for idx in indices:
    #     plt.plot([idx, idx], [0.5, 1.2], 'k--')
    # for idx_a in a:
    #     plt.plot([idx_a, idx_a], [0.5, 1.2], 'r--')
    # plt.show()
    return a

def find_indices_speed(com_speed):
    index_anato = np.where(com_speed[2, :] > 0)[0]
    discontinuities_idx = np.where(np.gradient(index_anato) > 1)[0]
    indices = []

    for idx in index_anato[discontinuities_idx]:
        a = np.where((com_speed[2, idx:idx + 51] - com_speed[2, idx]) < 0)[0]
        b = np.where((com_speed[2, idx - 50:idx + 1] - com_speed[2, idx]) < 0)[0]
        if (a.size > 40) or (b.size > 40):
            indices.append(idx)

    plt.figure()
    plt.plot(com_speed[2, :])
    plt.plot([0, len(com_speed[2, :])], [0.0, 0.0], 'k--')
    for idx in indices:
        plt.plot([idx, idx], [-1, 1], 'k--')
    plt.show()
    return indices

def plot_events(q_kalman, q_idx, events):
    plt.figure()
    plt.plot(q_kalman[q_idx, :]*180/np.pi)
    plt.plot([0, len(q_kalman[0, :])], [0, 0], 'k--')
    for idx in events:
        plt.plot([idx, idx], [np.min(q_kalman[q_idx, :]*180/np.pi), np.max(q_kalman[q_idx, :]*180/np.pi)], 'k--')

def plot_mean_std(ax, m, sd, color):
    abs = np.linspace(0, 100, 200)
    ax.plot(abs, m, color=color)
    ax.fill_between(abs, m - sd, m + sd, color=color, alpha=0.2)

def plot_repet(ax, data, idx, color, trans=False):
    dt = 1/100
    for d in data:
        abs = (np.linspace(0, d.shape[1] * dt, d.shape[1]) / (d.shape[1] * dt)) * 100
        if trans:
            ax.plot(abs, d[idx, :], color=color)
        else:
            if (idx == 15) or (idx == 16) or (idx == 19): # left side
                ax.plot(abs, -d[idx, :] * 180/np.pi, color=color)
            else:
                ax.plot(abs, d[idx, :] * 180 / np.pi, color=color)

class kinematic:
    def __init__(self, name, higher_foot='R'):
        self.name = name
        self.higher_foot = higher_foot
        self.path = '/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/squat/Data_test/' + name
        self.model = biorbd.Model(self.path + "/" + name + ".bioMod")
        self.list_exp_files = ['squat_controle', 'squat_3cm', 'squat_4cm', 'squat_5cm', 'squat_controle_post']
        self.q_name = utils.get_q_name(self.model)
        self.label_q = ["pelvis Tx", "pelvis Ty", "pelvis Tz",
                        "pelvis Rx", "pelvis Ry", "pelvis Rz",
                        "tronc Rx", "tronc Ry", "tronc Rz",
                        "hanche Rx", "hanche Ry", "hanche Rz",
                        "genou Rz",
                        "cheville Rx", "cheville Rz"]

        # get events from markers position
        mark = markers(self.name)
        self.events = mark.get_events()[0]
        self.time = mark.get_time()
        # load kinematic from kalman filter
        self.q_kalman = self.get_q_kalman()
        self.qdot_kalman = self.get_qdot_kalman()
        self.com_position = self.compute_com(self.q_kalman)
        self.com_speed = self.compute_com_speed(self.q_kalman, self.qdot_kalman)
        # divide repetition and compute mean
        self.q = self.get_q()
        self.q_mean, self.q_std = self.get_mean(self.q_kalman)
        self.qdot_mean, self.qdot_std = self.get_mean(self.qdot_kalman)
        self.com_mean_position = self.compute_com(self.q_mean)
        # compute speed and max angle
        self.speed = self.get_speed()
        idx_hip = [11, 17]
        idx_knee = [12, 18]
        idx_ankle = [14, 20]
        if self.higher_foot == 'R':
            # control leg first : left one and right elevated
            self.pic_flexion_hip = [self.get_pic_value(index=idx_hip[1]), self.get_pic_value(index=idx_hip[0])]
            self.pic_flexion_knee = [self.get_pic_value(index=idx_knee[1]), self.get_pic_value(index=idx_knee[0])]
            self.pic_flexion_ankle = [self.get_pic_value(index=idx_ankle[1]), self.get_pic_value(index=idx_ankle[0])]
        else:
            # control leg first : right one and left elevated
            self.pic_flexion_hip = [self.get_pic_value(index=idx_hip[0]), self.get_pic_value(index=idx_hip[1])]
            self.pic_flexion_knee = [self.get_pic_value(index=idx_knee[0]), self.get_pic_value(index=idx_knee[1])]
            self.pic_flexion_ankle = [self.get_pic_value(index=idx_ankle[0]), self.get_pic_value(index=idx_ankle[1])]


    def get_q_kalman(self):
        q_kalman = [utils.load_txt_file(self.path + "/kalman/" + file + "_q_KalmanFilter.txt", self.model.nbQ()) for (i, file) in enumerate(self.list_exp_files)]
        return q_kalman

    def get_qdot_kalman(self):
        qd_kalman = [utils.load_txt_file(self.path + "/kalman/" + file + "_qdot_KalmanFilter.txt", self.model.nbQ()) for (i, file) in enumerate(self.list_exp_files)]
        return qd_kalman

    def get_events(self):
        q_anato = utils.load_txt_file(self.path + "/kalman/anato_q_KalmanFilter.txt", self.model.nbQ())
        com_anato = self.compute_com([q_anato])[0]
        events = find_indices(self.com_position[0], com_anato)
        # plot_events(self.q_kalman[0], q_idx=12, events=events)
        # plot_events(self.com_speed[0], q_idx=2, events=events)
        # plt.show()
        return events

    def get_events_speed(self):
        events = find_indices_speed(self.com_speed[0])
        # plot_events(self.q_kalman[0], q_idx=12, events=events)
        # plt.show()
        return events

    def get_q(self):
        return [utils.divide_squat_repetition(q_kal, self.events[i]) for (i, q_kal) in enumerate(self.q_kalman)]

    def get_qdot(self):
        return [utils.divide_squat_repetition(qd_kal, self.events[i]) for (i, qd_kal) in enumerate(self.qdot_kalman)]

    def get_mean(self, data):
        mean = [utils.compute_mean(d, self.events[i], 100)[0] for (i, d) in enumerate(data)]
        std = [utils.compute_mean(d, self.events[i], 100)[1] for (i, d) in enumerate(data)]
        return mean, std

    def get_pic_value(self, index):
        return [find_pic(q, index) for q in self.q]

    def get_amp_value(self, index):
        return [find_amplitude(q, index) for q in self.q]

    def compute_com(self, q_data):
        compute_CoM = biorbd.to_casadi_func("CoM", self.model.CoM, MX.sym("q", self.model.nbQ(), 1))
        return [np.array(compute_CoM(q)) for q in q_data]

    def compute_com_speed(self, q_data, qdot_data):
        compute_CoM_speed = biorbd.to_casadi_func("CoM_speed",
                                                  self.model.CoMdot,
                                                  MX.sym("q", self.model.nbQ(), 1),
                                                  MX.sym("qdot", self.model.nbQ(), 1))
        return [np.array(compute_CoM_speed(q, qdot_data[i])) for (i, q) in enumerate(q_data)]

    def get_speed(self):
        return [compute_speed(com[2, :], self.time[i]) for (i, com) in enumerate(self.com_position)]


    def plot_q_repet(self, idx_condition):
        fig, axes = plt.subplots(5, 3, sharex=True)
        fig.suptitle('repet joint angles')
        # pelvis
        axes[0, 0].set_ylabel('pelvic translation [m]')
        axes[1, 0].set_ylabel('pelvic rotation [°]')
        for i in range(3):
            # pelvis translation
            axes[0, i].set_title(self.label_q[i])
            plot_repet(axes[0, i], self.q[idx_condition], idx=i, color='g', trans=True)
            # pelvis rotation
            axes[1, i].set_title(self.label_q[i + 3])
            plot_repet(axes[1, i], self.q[idx_condition], idx=i+3, color='g')

            # hip
            axes[2, 0].set_ylabel('hip rotation [°]')
            for i in range(3):
                axes[2, i].set_title(self.label_q[i + 9])
                plot_repet(axes[2, i], self.q[idx_condition], idx=i + 9, color='r') # right side
                plot_repet(axes[2, i], self.q[idx_condition], idx=i + 15, color='b')  # left side

            # knee
            axes[3, 0].set_ylabel('knee rotation [°]')
            axes[3, 0].set_title(self.label_q[12])
            plot_repet(axes[3, 0], self.q[idx_condition], idx=12, color='r')  # right side
            plot_repet(axes[3, 0], self.q[idx_condition], idx=18, color='b')  # left side

            # ankle
            axes[4, 0].set_ylabel('ankle rotation [°]')
            for i in range(2):
                axes[4, i].set_title(self.label_q[i + 13])
                plot_repet(axes[4, i], self.q[idx_condition], idx=i + 13, color='r')  # right side
                plot_repet(axes[4, i], self.q[idx_condition], idx=i + 19, color='b')  # left side

        axes[4, 0].set_xlabel('time [%]')
        axes[4, 1].set_xlabel('time [%]')
        axes[4, 2].set_xlabel('time [%]')
        axes[4, 1].set_xlim([0, 100])
        plt.show()


    def plot_q_mean(self, idx_condition):
        fig, axes = plt.subplots(5, 3, sharex=True)
        fig.suptitle('mean joint angles')
        # pelvis
        axes[0, 0].set_ylabel('pelvic translation [m]')
        axes[1, 0].set_ylabel('pelvic rotation [°]')
        for i in range(3):
            # pelvis translation
            axes[0, i].set_title(self.label_q[i])
            plot_mean_std(axes[0, i], self.q_mean[idx_condition][i, :], self.q_std[idx_condition][i, :], color='g')

            # pelvis rotation
            axes[1, i].set_title(self.label_q[i + 3])
            plot_mean_std(axes[1, i], self.q_mean[idx_condition][i + 3, :] * 180 / np.pi,
                          self.q_std[idx_condition][i + 3, :] * 180 / np.pi, color='g')

            # hip
            axes[2, 0].set_ylabel('hip rotation [°]')
            for i in range(3):
                axes[2, i].set_title(self.label_q[i + 9])
                # right side
                m_r = self.q_mean[idx_condition][i + 9, :] * 180 / np.pi
                sd_r = self.q_std[idx_condition][i + 9, :] * 180 / np.pi
                plot_mean_std(axes[2, i], m_r, sd_r, color='r')

                # left side
                m_l = self.q_mean[idx_condition][i + 15, :] * 180 / np.pi if (i==2) else -self.q_mean[idx_condition][i + 15, :] * 180 / np.pi
                sd_l = self.q_std[idx_condition][i + 15, :] * 180 / np.pi
                plot_mean_std(axes[2, i], m_l, sd_l, color='b')

            # knee
            axes[3, 0].set_ylabel('knee rotation [°]')
            axes[3, 0].set_title(self.label_q[12])
            # right side
            plot_mean_std(axes[3, 0], self.q_mean[idx_condition][12, :] * 180 / np.pi, self.q_std[idx_condition][12, :] * 180 / np.pi, color='r')
            # left side
            plot_mean_std(axes[3, 0], self.q_mean[idx_condition][18, :] * 180 / np.pi, self.q_std[idx_condition][18, :] * 180 / np.pi, color='b')

            # ankle
            axes[4, 0].set_ylabel('ankle rotation [°]')
            for i in range(2):
                axes[4, i].set_title(self.label_q[i + 13])
                # right side
                plot_mean_std(axes[4, i], self.q_mean[idx_condition][i + 13, :] * 180 / np.pi, self.q_std[idx_condition][i + 13, :] * 180 / np.pi, color='r')
                # left side
                m_l = self.q_mean[idx_condition][i + 19, :] * 180 / np.pi if (i == 1) else -self.q_mean[idx_condition][i + 19, :] * 180 / np.pi
                plot_mean_std(axes[4, i], m_l, self.q_std[idx_condition][i + 19, :] * 180 / np.pi, color='b')

        axes[4, 0].set_xlabel('time [%]')
        axes[4, 1].set_xlabel('time [%]')
        axes[4, 2].set_xlabel('time [%]')
        axes[4, 1].set_xlim([0, 100])
        plt.show()