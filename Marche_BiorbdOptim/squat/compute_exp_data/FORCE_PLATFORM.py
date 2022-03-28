import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from .MARKERS import markers
from .UTILS import utils


def get_cop(loaded_c3d):
    cop = []
    platform = loaded_c3d["data"]["platform"]
    for p in platform:
        cop.append(p["center_of_pressure"] * 1e-3)
    return cop

def get_corners_position(loaded_c3d):
    corners = []
    platform = loaded_c3d["data"]["platform"]
    for p in platform:
        corners.append(p["corners"] * 1e-3)
    return corners

def get_forces(loaded_c3d):
    b, a = utils.define_butterworth_filter(fs=1000, fc=15)
    force = []
    platform = loaded_c3d["data"]["platform"]
    for p in platform:
        f = p["force"] # raw data
        f_filt = np.vstack([signal.filtfilt(b, a, p["force"][i, :]) for i in range(3)]) # apply filter
        force.append(f_filt)
    return force

def get_moments(loaded_c3d):
    moment = []
    platform = loaded_c3d["data"]["platform"]
    for p in platform:
        moment.append(p["moment"] * 1e-3)
    return moment

def get_moments_at_cop(loaded_c3d):
    Tz = []
    platform = loaded_c3d["data"]["platform"]
    for p in platform:
        Tz.append(p["Tz"] * 1e-3)
    return Tz


class force_platform:
    def __init__(self, name):
        self.name = name
        self.path = '../Data_test/' + name
        self.freq = 1000
        marker = markers(self.path)
        self.events = marker.events
        self.time = marker.time
        self.list_exp_files = ['squat_controle', 'squat_3cm', 'squat_4cm', 'squat_5cm']
        self.loaded_c3d = utils.load_c3d(self.path + '/Squats/', self.list_exp_files, extract_forceplat_data=True)
        self.force = [get_forces(c3d_file) for c3d_file in self.loaded_c3d]
        self.moments = [get_moments_at_cop(c3d_file) for c3d_file in self.loaded_c3d]
        self.cop = [get_cop(c3d_file) for c3d_file in self.loaded_c3d]
        self.mean_force, self.std_force = self.get_mean(self.force)
        self.mean_moment, self.std_moment = self.get_mean(self.moments)
        self.mean_cop, self.mean_cop = self.get_mean(self.cop)

    def get_mean(self, data):
        mean_right = [utils.compute_mean(d[0], [e * 10 for e in self.events[i]], 1000)[0] for (i, d) in enumerate(data)] # right leg # frequence 10 fois plus elevé
        std_right = [utils.compute_mean(d[0], [e * 10 for e in self.events[i]], 1000)[1] for (i, d) in enumerate(data)]
        mean_left = [utils.compute_mean(d[1], [e * 10 for e in self.events[i]], 1000)[0] for (i, d) in enumerate(data)]  # left leg
        std_left = [utils.compute_mean(d[1], [e * 10 for e in self.events[i]], 1000)[1] for (i, d) in enumerate(data)]
        return [mean_right, mean_left], [std_right, std_left]

    def plot_force_mean(self, idx_condition):
        abs = np.linspace(0, 100, 2000)
        colors = ['r', 'g', 'b']
        fig, axes = plt.subplots(3, 2, sharex=True)
        fig.suptitle('mean forces')
        axes[0, 0].set_title('Right leg')
        axes[0, 1].set_title('Left leg')
        for i in range(3):
            axes[i, 0].plot(abs, self.mean_force[0][idx_condition][i, :], c=colors[i])
            axes[i, 0].fill_between(abs,
                                    self.mean_force[0][idx_condition][i, :] - self.std_force[0][idx_condition][i, :],
                                    self.mean_force[0][idx_condition][i, :] + self.std_force[0][idx_condition][i, :],
                                    color=colors[i], alpha=0.2)
            axes[i, 1].plot(abs, self.mean_force[1][idx_condition][i, :], color=colors[i])
            axes[i, 1].fill_between(abs,
                                    self.mean_force[1][idx_condition][i, :] - self.std_force[1][idx_condition][i, :],
                                    self.mean_force[1][idx_condition][i, :] + self.std_force[1][idx_condition][i, :],
                                    color=colors[i], alpha=0.2)
        axes[0, 0].set_ylabel('forces in X [N]')
        axes[1, 0].set_ylabel('forces in Y [N]')
        axes[2, 0].set_ylabel('forces in Z [N]')
        axes[2, 0].set_xlabel('time [%]')
        axes[2, 1].set_xlabel('time [%]')
        axes[2, 1].set_xlim([0, 100])
        plt.show()

    def plot_force_repet(self, idx_condition):
        dt = 1/1000
        repet_right = utils.divide_squat_repetition(self.force[idx_condition][0], [e * 10 for e in self.events[idx_condition]])
        repet_left = utils.divide_squat_repetition(self.force[idx_condition][1], [e * 10 for e in self.events[idx_condition]])

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        fig, axes = plt.subplots(3, 2, sharex=True)
        fig.suptitle('forces for each repetition')
        axes[0, 0].set_title('Right leg')
        axes[0, 1].set_title('Left leg')
        for i in range(len(repet_right)):
            abs = (np.linspace(0, repet_right[i].shape[1]*dt, repet_right[i].shape[1])/(repet_right[i].shape[1]*dt))*100
            for j in range(3):
                axes[j, 0].plot(abs, repet_right[i][j, :], c=colors[i])
                axes[j, 1].plot(abs, repet_left[i][j, :], c=colors[i])
        axes[0, 0].set_ylabel('forces in X [N]')
        axes[1, 0].set_ylabel('forces in Y [N]')
        axes[2, 0].set_ylabel('forces in Z [N]')
        axes[2, 0].set_xlabel('time [%]')
        axes[2, 1].set_xlabel('time [%]')
        axes[2, 1].set_xlim([0, 100])
        plt.show()
