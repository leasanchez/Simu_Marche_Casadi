import matplotlib.pyplot as plt
import numpy as np
import os
from pyomeca import Analogs
from .MARKERS import markers
from .UTILS import utils

def get_mvc_files(path):
    return os.listdir(path+'/MVC/')

def find_muscle_mvc_value(emg, idx_muscle, freq):
    a = []
    for e in emg:
        a = np.concatenate([a, e[idx_muscle].data])
    return np.mean(np.sort(a)[-int(freq):])

def correct_mvc_value(name, mvc_value):
    if name == 'AmeCeg':
        mvc_value[2] = mvc_value[3]  # AC
        mvc_value[12] = mvc_value[13]
    elif name == 'AnaLau':
        mvc_value[2] = mvc_value[3]  # AL
        mvc_value[7] = mvc_value[6]
        mvc_value[15] = mvc_value[14]
    elif name == 'BeaMoy':
        mvc_value[5] = mvc_value[4]  # BM
        mvc_value[8] = mvc_value[9]
    elif name == 'EriHou':
        mvc_value[0] = mvc_value[1]  # EH
        mvc_value[9] = mvc_value[8]
    elif name == 'LudArs':
        mvc_value[14] = mvc_value[15]  # LA
        mvc_value[2] = mvc_value[3]

    # self.mvc_value[3] = self.mvc_value[2] #GD - no mvc
    # self.mvc_value[19] = self.mvc_value[18]

    # self.mvc_value[0] = self.mvc_value[1] #JD - no mvc
    # self.mvc_value[2] = self.mvc_value[3]
    return mvc_value

class emg:
    def __init__(self, name, higher_foot='R'):
        self.name = name
        self.higher_foot = higher_foot
        self.path = '/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/squat/Data_test/' + name
        self.list_mvc_files = get_mvc_files(self.path)
        self.list_exp_files = ['squat_controle', 'squat_3cm', 'squat_4cm', 'squat_5cm', 'squat_controle_post']
        self.nb_mus = 20
        self.label_muscles_analog = ['Voltage.GM_r', 'Voltage.GM_l', # gastrocnemiem medial
                                     'Voltage.SOL_r', 'Voltage.SOL_l', # soleaire
                                     'Voltage.LF_r', 'Voltage.LF_l', # long fibulaire
                                     'Voltage.TA_r', 'Voltage.TA_l', # tibial anterieur
                                     'Voltage.VM_r', 'Voltage.VM_l', # vaste medial
                                     'Voltage.RF_r', 'Voltage.RF_l', # rectus femoris
                                     'Voltage.ST_r', 'Voltage.ST_l', # semi tendineux
                                     'Voltage.MF_r', 'Voltage.MF_l', # moyen fessier
                                     'Voltage.GF_r', 'Voltage.GF_l', # grand fessier
                                     'Voltage.LA_r', 'Voltage.LA_l'] # long adducteur
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
        self.emg_filtered_mvc = [self.get_filtered_emg(file_path=self.path + '/MVC/' + file) for file in self.list_mvc_files]
        self.emg_filtered_exp = [self.get_filtered_emg(file_path=self.path + '/Squats/' + file + ".c3d")for file in self.list_exp_files]
        self.mvc_value = self.get_mvc_value()
        self.mvc_value = correct_mvc_value(name, self.mvc_value) # MVC correction for some subjects
        self.emg_normalized_exp = [self.get_normalized_emg(file_path=self.path + '/Squats/' + file + ".c3d") for file in self.list_exp_files]

        self.events = markers(self.name).events
        self.mean, self.std = self.get_mean()


    def get_raw_emg(self, file_path):
        emg = Analogs.from_c3d(file_path, usecols=self.label_muscles_analog)
        return emg

    def get_filtered_emg(self, file_path):
        emg = Analogs.from_c3d(file_path, usecols=self.label_muscles_analog)
        self.freq = int(1/np.array(emg.time)[1])
        emg_process = (
            emg.meca.band_pass(order=2, cutoff=[10, 425])
                .meca.center()
                .meca.abs()
                .meca.low_pass(order=4, cutoff=6, freq=emg.rate)
        )
        return emg_process

    def get_normalized_emg(self, file_path):
        emg = Analogs.from_c3d(file_path, usecols=self.label_muscles_analog)
        emg_norm = []
        for (i, e) in enumerate(emg):
            emg_norm.append(
                e.meca.band_pass(order=2, cutoff=[10, 425])
                 .meca.center()
                 .meca.abs()
                 .meca.low_pass(order=4, cutoff=6, freq=emg.rate)
                 .meca.normalize(self.mvc_value[i])
                 .meca.abs()
            )
        return np.vstack(emg_norm)

    def get_mvc_value(self):
        return [find_muscle_mvc_value(self.emg_filtered_mvc + self.emg_filtered_exp, idx_muscle=i, freq=self.freq) for i in range(self.nb_mus)]

    def get_mean(self):
        mean = [utils.compute_mean(emg, [e * int(self.freq/100) for e in self.events[i]], self.freq)[0] for (i, emg) in enumerate(self.emg_normalized_exp)]
        std = [utils.compute_mean(emg, [e * int(self.freq / 100) for e in self.events[i]], self.freq)[1] for (i, emg) in enumerate(self.emg_normalized_exp)]
        return mean, std

    def plot_mvc_data(self, emg_data):
        fig, axes = plt.subplots(4, 5)
        axes = axes.flatten()
        fig.suptitle('MVC')
        for i in range(self.nb_mus):
            axes[i].set_title(self.label_muscles_analog[i])
            for emg in emg_data:
                axes[i].plot(emg[i].time.data, emg[i].data)

    def plot_mean_activation(self, idx_condition):
        abs = np.linspace(0, 100, 2*self.freq)
        fig, axes = plt.subplots(int(self.nb_mus/2), 2, sharex=True)
        fig.suptitle('mean muscle activation')
        axes[0, 0].set_title('Right leg')
        axes[0, 1].set_title('Left leg')
        for i in range(int(self.nb_mus/2)):
            axes[i, 0].plot(abs, self.mean[idx_condition][2*i, :], 'r')
            axes[i, 0].fill_between(abs,
                                    self.mean[idx_condition][2*i, :] - self.std[idx_condition][2*i, :],
                                    self.mean[idx_condition][2*i, :] + self.std[idx_condition][2*i, :],
                                    color='r', alpha=0.2)
            axes[i, 1].plot(abs, self.mean[idx_condition][2*i + 1, :], 'b')
            axes[i, 1].fill_between(abs,
                                    self.mean[idx_condition][2*i + 1, :] - self.std[idx_condition][2*i + 1, :],
                                    self.mean[idx_condition][2*i + 1, :] + self.std[idx_condition][2*i + 1, :],
                                    color='b', alpha=0.2)
            axes[i, 0].set_ylabel(self.label_muscles[i])
            axes[i, 0].set_ylim([0, 100])
            axes[i, 1].set_ylim([0, 100])
        axes[9, 0].set_xlabel('time [%]')
        axes[9, 1].set_xlabel('time [%]')
        axes[9, 1].set_xlim([0, 100])
        plt.show()

    def plot_repet_activation(self, idx_condition):
        dt = 1 / self.freq
        ev = [e * int(self.freq/100) for e in self.events[idx_condition]]
        repet_right = utils.divide_squat_repetition(self.emg_normalized_exp[idx_condition][0::2, :], ev)
        repet_left = utils.divide_squat_repetition(self.emg_normalized_exp[idx_condition][1::2, :], ev)

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        fig, axes = plt.subplots(int(self.nb_mus/2), 2, sharex=True)
        fig.suptitle('forces for each repetition')
        axes[0, 0].set_title('Right leg')
        axes[0, 1].set_title('Left leg')
        for i in range(len(repet_right)):
            abs = (np.linspace(0, repet_right[i].shape[1] * dt, repet_right[i].shape[1]) / (
                        repet_right[i].shape[1] * dt)) * 100
            for j in range(int(self.nb_mus/2)):
                axes[j, 0].plot(abs, repet_right[i][j, :], c=colors[i])
                axes[j, 1].plot(abs, repet_left[i][j, :], c=colors[i])
        for j in range(int(self.nb_mus / 2)):
            axes[j, 0].set_ylabel(self.label_muscles[j])
            axes[j, 0].set_ylim([0, 100])
            axes[j, 1].set_ylim([0, 100])
        axes[9, 0].set_xlabel('time [%]')
        axes[9, 1].set_xlabel('time [%]')
        axes[9, 1].set_xlim([0, 100])
        plt.show()
