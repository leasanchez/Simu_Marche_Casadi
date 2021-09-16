import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, signal
from ezc3d import c3d
from MARKERS import markers


def get_cop(loaded_c3d):
    """
    get the trajectory of the center of pressure (cop)
    from force platform
    """
    cop = []
    platform = loaded_c3d["data"]["platform"]
    for p in platform:
        cop.append(p["center_of_pressure"] * 1e-3)
    return cop


def get_corners_position(loaded_c3d):
    """
    get platform corners position
    """
    corners = []
    platform = loaded_c3d["data"]["platform"]
    for p in platform:
        corners.append(p["corners"] * 1e-3)
    return corners

def define_butterworth_filter(fs, fc):
    """
    define filter parameters
    input : fs : sample frequency, fc : cut frequency
    output : signal parameter for low pass filter
    """
    w = fc / (fs / 2)
    b, a = signal.butter(4, w, 'low')
    return b, a

def get_forces(loaded_c3d):
    """
    get the ground reaction forces
    from force platform
    """
    b, a = define_butterworth_filter(fs=1000, fc=15)
    force = []
    platform = loaded_c3d["data"]["platform"]
    for p in platform:
        f = np.ndarray((3, p["force"].shape[1]))
        for i in range(3):
            f[i, :] = signal.filtfilt(b, a, p["force"][i, :])
        force.append(f)
    return force


def get_moments(loaded_c3d):
    """
    get the ground reaction moments
    from force platform
    """
    moment = []
    platform = loaded_c3d["data"]["platform"]
    for p in platform:
        moment.append(p["moment"] * 1e-3)
    return moment


def get_moments_at_cop(loaded_c3d):
    """
    get the ground reaction moments at cop
    from force platform
    """
    Tz = []
    platform = loaded_c3d["data"]["platform"]
    for p in platform:
        Tz.append(p["Tz"] * 1e-3)
    return Tz

def load_c3d(c3d_path):
    return c3d(c3d_path, extract_forceplat_data=True)

def divide_squat_repetition(data, index):
    data_divide = []
    for d in data:
        squat = []
        for idx in range(int(len(index)/2)):
            squat.append(d[:, index[2*idx]*10: index[2*idx + 1]*10])
        data_divide.append(squat)
    return data_divide

def interpolate_squat_repetition(data, index):
    data_divide = divide_squat_repetition(data, index)
    data_interp = []
    for d in data_divide:
        squat_interp = np.zeros((int(len(index) / 2), 3, 2000))
        for (i, s) in enumerate(d):
            x_start = np.arange(0, s[0].shape[0])
            x_interp = np.linspace(0, x_start[-1], 2000)
            for m in range(3):
                f = interpolate.interp1d(x_start, s[m, :])
                squat_interp[i, m, :] = f(x_interp)
        data_interp.append(squat_interp)
    return data_interp

def compute_mean_squat_repetition(data, index):
    data_interp = interpolate_squat_repetition(data, index)
    M = []
    SD = []
    for d in data_interp:
        mean = np.zeros((3, 2000))
        std = np.zeros((3, 2000))
        for i in range(3):
            mean[i, :] = np.mean(d[:, i, :], axis=0)
            std[i, :] = np.std(d[:, i, :], axis=0)
        M.append(mean)
        SD.append(std)
    return M, SD


class force_platform:
    def __init__(self, name):
        self.name = name
        self.path = '../Data_test/' + name
        self.events = markers(self.path).get_events()
        self.list_exp_files = ['squat_controle.c3d', 'squat_3cm.c3d', 'squat_4cm.c3d', 'squat_5cm.c3d']
        self.loaded_c3d = []
        self.force = []
        self.moments = []
        self.cop = []
        for (i, file) in enumerate(self.list_exp_files):
            self.loaded_c3d.append(load_c3d(self.path + '/Squats/' + file))
            self.force.append(get_forces(self.loaded_c3d[i]))
            self.moments.append(get_moments_at_cop(self.loaded_c3d[i]))
            self.cop.append(get_cop(self.loaded_c3d[i]))
        self.mean_force, self.std_force = self.get_mean(self.force)
        self.mean_moment, self.std_moment = self.get_mean(self.moments)
        self.mean_cop, self.mean_cop = self.get_mean(self.cop)


    def interpolate_data(self, data, index):
        data_interp = []
        for d in data:
            interp = np.zeros((len(index)-1, 3, 5000))
            for i in range(len(index)-1):
                x_start = np.arange(0, d[0, index[i]:index[i+1]].shape[0])
                x_interp = np.linspace(0, x_start[-1], 5000)
                for m in range(3):
                    f = interpolate.interp1d(x_start, d[m, index[i]:index[i+1]])
                    interp[i, m, :] = f(x_interp)
            data_interp.append(interp)
        return data_interp

    def get_mean(self, data):
        mean = []
        std = []
        for i in range(len(data)):
            A = compute_mean_squat_repetition(data[i], self.events[i])
            mean.append(A[0])
            std.append(A[1])
        return mean, std

    def plot_cop(self, file_path, index):
        corners = self.get_corners_position(self.load_c3d(file_path[0]))
        color = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
        mean = []
        for (i, file) in enumerate(file_path):
            cop = self.interpolate_data(data=self.get_cop(self.load_c3d(file)), index=index[i])
            m, std = self.mean_data(cop)
            mean.append(m)

        plt.figure()
        plt.suptitle('cop trajectories comparison')
        for (i, c) in enumerate(mean):
            plt.scatter(-c[0][1, :], c[0][0, :], c=color[i], alpha=0.5, label=file_path[i])
            plt.scatter(-c[1][1, :], c[1][0, :], c=color[i], alpha=0.5, label=file_path[i])
        plt.scatter(-corners[0][1, :], corners[0][0, :], c='k', marker='+') # platform 1 --> left
        plt.scatter(-corners[1][1, :], corners[1][0, :], c='y', marker='+')  # platform 2 --> right
        plt.xlabel('y (m)')
        plt.ylabel('x (m)')
        plt.legend()

    def plot_force_repetition(self, title=None):
        label = ['x', 'y', 'z']
        if title is not None:
            idx = self.list_exp_files.index(title)
            data_interp = interpolate_squat_repetition(self.force[idx], self.events[idx])
            fig, axes = plt.subplots(len(data_interp), 3)
            axes = axes.flatten()
            fig.suptitle(self.name + "\n ground reaction forces repetition " + title)
            for i in range(3):
                axes[i].set_title(f"platform 1 : {label[i]}")
                axes[i].plot(np.linspace(0, 100, 2000), data_interp[0][:, i, :].T)
                axes[i].plot([0, 100], [0, 0], 'k--')
                axes[i].set_xlim([0, 100])
            if len(data_interp) > 1:
                for i in range(3):
                    axes[i + 3].set_title(f"platform 2 : {label[i]}")
                    axes[i + 3].plot(np.linspace(0, 100, 2000), data_interp[1][:, i, :].T)
                    axes[i + 3].plot([0, 100], [0, 0], 'k--')
                    axes[i + 3].set_xlim([0, 100])
        else:
            for (t, title) in enumerate(self.list_exp_files):
                data_interp = interpolate_squat_repetition(self.force[t], self.events[t])
                fig, axes = plt.subplots(len(data_interp), 3)
                axes = axes.flatten()
                fig.suptitle(self.name + "\n ground reaction forces repetition " + title)
                for i in range(3):
                    axes[i].set_title(f"platform 1 : {label[i]}")
                    axes[i].plot(np.linspace(0, 100, 2000), data_interp[0][:, i, :].T)
                    axes[i].plot([0, 100], [0, 0], 'k--')
                    axes[i].set_xlim([0, 100])
                if len(data_interp) > 1:
                    for i in range(3):
                        axes[i + 3].set_title(f"platform 2 : {label[i]}")
                        axes[i + 3].plot(np.linspace(0, 100, 2000), data_interp[1][:, i, :].T)
                        axes[i + 3].plot([0, 100], [0, 0], 'k--')
                        axes[i + 3].set_xlim([0, 100])

    def plot_force_mean(self, title=None):
        label = ['x', 'y', 'z']
        abscisse = np.linspace(0, 100, 2000)
        if title is not None:
            idx = self.list_exp_files.index(title)
            fig, axes = plt.subplots(1, 3)
            axes = axes.flatten()
            fig.suptitle(self.name + "\nground reaction forces mean " + title)
            for i in range(3):
                axes[i].set_title(label[i])
                axes[i].plot(abscisse, self.mean_force[idx][0][i, :], 'r')
                axes[i].plot(abscisse, self.mean_force[idx][1][i, :], 'b')
                axes[i].fill_between(abscisse,
                                     self.mean_force[idx][0][i, :] - self.std_force[idx][0][i, :],
                                     self.mean_force[idx][0][i, :] + self.std_force[idx][0][i, :], color='r', alpha=0.2)
                axes[i].fill_between(abscisse,
                                     self.mean_force[idx][1][i, :] - self.std_force[idx][1][i, :],
                                     self.mean_force[idx][1][i, :] + self.std_force[idx][1][i, :], color='b', alpha=0.2)
                axes[i].set_xlim([0, 100])
                axes[i].set_xlabel('temps')
            axes[0].set_ylabel('forces')
            plt.legend(['right', 'left'])
        else:
            for (t, title) in enumerate(self.list_exp_files):
                fig, axes = plt.subplots(1, 3)
                axes = axes.flatten()
                fig.suptitle(self.name + "\nground reaction forces mean " + title)
                for i in range(3):
                    axes[i].set_title(label[i])
                    axes[i].plot(abscisse, self.mean_force[t][0][i, :], 'r')
                    if (i < 2):
                        axes[i].plot(abscisse, -self.mean_force[t][1][i, :], 'b')
                        axes[i].fill_between(abscisse,
                                             -(self.mean_force[t][1][i, :] - self.std_force[t][1][i, :]),
                                             -(self.mean_force[t][1][i, :] + self.std_force[t][1][i, :]), color='b',
                                             alpha=0.2)
                    else:
                        axes[i].plot(abscisse, self.mean_force[t][1][i, :], 'b')
                        axes[i].fill_between(abscisse,
                                             (self.mean_force[t][1][i, :] - self.std_force[t][1][i, :]),
                                             (self.mean_force[t][1][i, :] + self.std_force[t][1][i, :]), color='b',
                                             alpha=0.2)
                    axes[i].fill_between(abscisse,
                                         self.mean_force[t][0][i, :] - self.std_force[t][0][i, :],
                                         self.mean_force[t][0][i, :] + self.std_force[t][0][i, :], color='r',
                                         alpha=0.2)

                    axes[i].set_xlim([0, 100])
                    axes[i].set_xlabel('temps')
                axes[0].set_ylabel('forces')
                plt.legend(['right', 'left'])

    def plot_force_comparison(self):
        label = ['x', 'y', 'z']
        fig, axes = plt.subplots(2, 3)
        axes = axes.flatten()
        fig.suptitle(self.name + "\nground reaction forces comparison controle vs 5cm")
        for i in range(3):
            axes[i].set_title(f"platform 1 : {label[i]}")
            axes[i].plot(np.linspace(0, 100, 2000), self.mean_force[0][0][i, :], 'b')
            axes[i].plot(np.linspace(0, 100, 2000), self.mean_force[3][0][i, :], 'r')
            axes[i].plot([0, 100], [0, 0], 'k--')
            axes[i].set_xlim([0, 100])
        for i in range(3):
            axes[i + 3].set_title(f"platform 2 : {label[i]}")
            axes[i + 3].plot(np.linspace(0, 100, 2000), self.mean_force[0][1][i, :], 'b')
            axes[i + 3].plot(np.linspace(0, 100, 2000), self.mean_force[3][1][i, :], 'r')
            axes[i + 3].plot([0, 100], [0, 0], 'k--')
            axes[i + 3].set_xlim([0, 100])
        plt.legend(['controle', '5cm'])