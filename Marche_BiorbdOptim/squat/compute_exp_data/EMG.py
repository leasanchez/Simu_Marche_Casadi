import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import interpolate
from ezc3d import c3d
from pyomeca import Analogs
from MARKERS import markers

def get_mvc_files(path):
    return os.listdir(path+'/MVC/')

def get_exp_files(path):
    return os.listdir(path + '/Squats/')

def get_labels_muscles(c3d_path):
    loaded_c3d = c3d(c3d_path)
    a = loaded_c3d["parameters"]["ANALOG"]["LABELS"]["value"].index("Voltage.GM_r")
    b = loaded_c3d["parameters"]["ANALOG"]["LABELS"]["value"].index("Voltage.LA_l")
    return loaded_c3d["parameters"]["ANALOG"]["LABELS"]["value"][a:b+1]

def find_muscle_mvc_value(emg, idx_muscle, freq):
    a = []
    for e in emg:
        a = np.concatenate([a, e[idx_muscle].data])
    return np.mean(np.sort(a)[-int(freq):])

class emg:
    def __init__(self, path):
        self.path = path
        self.list_mvc_files = get_mvc_files(path)
        self.list_exp_files = get_exp_files(path)
        self.label_muscles_analog = get_labels_muscles(path + '/MVC/' + self.list_mvc_files[0])
        self.label_muscles = ['Gastrocnemien medial',
                              'Soleaire',
                              'Long fibulaire',
                              'Tibial anterieur',
                              'Vaste medial',
                              'Droit anterieur',
                              'Semitendineux',
                              'Moyen fessier',
                              'Grand fessier',
                              'Long adducteur']
        self.nb_mus = len(self.label_muscles_analog)

        self.emg_filtered = []
        self.emg_filtered_mvc = []
        self.emg_filtered_exp = []
        self.emg_normalized_exp = []
        for file in self.list_mvc_files:
            self.emg_filtered.append(self.get_filtered_emg(file_path=self.path + '/MVC/' + file))
            self.emg_filtered_mvc.append(self.get_filtered_emg(file_path=self.path + '/MVC/' + file))
        for file in self.list_exp_files:
            self.emg_filtered.append(self.get_filtered_emg(file_path=self.path + '/Squats/' + file))
            self.emg_filtered_exp.append(self.get_filtered_emg(file_path=self.path + '/Squats/' + file))

        self.mvc_value = self.get_mvc_value()

        for file in self.list_exp_files:
            self.emg_normalized_exp.append(self.get_normalized_emg(file_path=self.path + '/Squats/' + file))

        self.events = markers(path).get_events()


    def get_raw_emg(self, file_path):
        emg = Analogs.from_c3d(file_path, usecols=self.label_muscles_analog)
        return emg

    def get_filtered_emg(self, file_path):
        emg = Analogs.from_c3d(file_path, usecols=self.label_muscles_analog)
        self.freq = 1/np.array(emg.time)[1]
        emg_process = (
            emg.meca.band_pass(order=2, cutoff=[10, 425])
                .meca.center()
                .meca.abs()
                .meca.low_pass(order=4, cutoff=6, freq=emg.rate)
        )
        return emg_process

    def get_normalized_emg(self, file_path):
        emg = self.get_raw_emg(file_path)
        emg_norm = []
        for (i, e) in enumerate(emg):
            emg_norm.append(
                e.meca.band_pass(order=2, cutoff=[10, 425])
                 .meca.center()
                 .meca.abs()
                 .meca.low_pass(order=4, cutoff=6, freq=emg.rate)
                 .meca.normalize(self.mvc_value[i])
            )
        return emg_norm

    def get_mvc_value(self):
        mvc_value=[]
        for i in range(self.nb_mus):
            mvc_value.append(find_muscle_mvc_value(self.emg_filtered, idx_muscle=i, freq=self.freq))
        return mvc_value

    def compute_mean_emg_squat_repetition(self, file_path, index):
        emg_squat_interp = self.interpolate_emg_squat_repetition(file_path, index)
        mean_emg = np.zeros((self.nb_mus, 5000))
        std_emg = np.zeros((self.nb_mus, 5000))
        for m in range(self.nb_mus):
            mean_emg[m, :] = np.mean(emg_squat_interp[:, m, :], axis=0)
            std_emg[m, :] = np.std(emg_squat_interp[:, m, :], axis=0)
        return mean_emg, std_emg

    def plot_mvc_data(self, emg_data):
        fig, axes = plt.subplots(4, 5)
        axes = axes.flatten()
        fig.suptitle('MVC')
        for i in range(self.nb_mus):
            axes[i].set_title(self.label_muscles_analog[i])
            for emg in emg_data:
                axes[i].plot(emg[i].time.data, emg[i].data)

    def plot_squat(self, emg_data, title):
        fig, axes = plt.subplots(2, 5)
        axes = axes.flatten()
        fig.suptitle(title)
        for i in range(int(self.nb_mus/2)):
            axes[i].set_title(self.label_muscles[i])
            axes[i].plot(emg_data[2*i].time.data, emg_data[2*i].data)
            axes[i].plot(emg_data[2*i + 1].time.data, emg_data[2*i + 1].data)
            axes[i].set_ylim([0, 100])
        plt.legend(['right', 'left'])

    def plot_squat_repetition(self, file_path, index):
        emg_squat_interp = self.interpolate_emg_squat_repetition(file_path, index)
        fig, axes = plt.subplots(4, 5)
        axes = axes.flatten()
        fig.suptitle(file_path)
        for i in range(self.nb_mus):
            axes[i].set_title(self.label_muscles_analog[i])
            axes[i].plot(np.linspace(0, 100, 5000), emg_squat_interp[:, i, :].T)
            axes[i].set_xlim([0, 100])
            axes[i].set_ylim([0, 100])

    def plot_squat_mean(self, file_path, index):
        mean, std = self.compute_mean_emg_squat_repetition(file_path, index)
        fig, axes = plt.subplots(3, 3)
        axes = axes.flatten()
        fig.suptitle(file_path)
        for i in range(int(self.nb_mus/2)):
            axes[i].set_title(self.label_muscles[i])
            axes[i].plot(np.linspace(0, 100, 5000), mean[2*i, :], 'r')
            axes[i].plot(np.linspace(0, 100, 5000), mean[2*i + 1, :], 'b')

            axes[i].fill_between(np.linspace(0, 100, 5000), mean[2*i, :] - std[2*i, :], mean[2 * i, :] + std[2 * i, :], color='r', alpha=0.2)
            axes[i].fill_between(np.linspace(0, 100, 5000), mean[2*i + 1] - std[2*i + 1, :], mean[2 * i + 1] + std[2 * i + 1, :], color='b', alpha=0.2)
            axes[i].set_xlim([0, 100])
            axes[i].set_ylim([0, 100])
        plt.legend(['right', 'left'])

    def plot_squat_comparison(self, file_path, index):
        mean = []
        for (i, file) in enumerate(file_path):
            m, std = self.compute_mean_emg_squat_repetition(file, index[i])
            mean.append(m)
        fig, axes = plt.subplots(4, 5)
        axes = axes.flatten()
        fig.suptitle('comparison')
        for i in range(self.nb_mus):
            axes[i].set_title(self.label_muscles_analog[i])
            axes[i].plot(np.linspace(0, 100, 5000), mean[0][i, :], color='#d62728', linestyle='-')
            axes[i].plot(np.linspace(0, 100, 5000), mean[1][i, :], color='#ff7f0e', linestyle='--')
            axes[i].plot(np.linspace(0, 100, 5000), mean[2][i, :], color='#2ca02c', linestyle='--')
            axes[i].plot(np.linspace(0, 100, 5000), mean[3][i, :], color='#1f77b4', linestyle='--')
            axes[i].set_xlim([0, 100])
            axes[i].set_ylim([0, 100])
        axes[self.nb_mus-1].legend(file_path)



