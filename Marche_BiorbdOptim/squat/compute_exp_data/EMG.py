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

def divide_squat_repetition(emg, index, freq):
    emg_squat = []
    for idx in range(int(len(index)/2)):
        e = []
        for m in range(len(emg)):
            e.append(emg[m].data[int(index[2*idx]*freq/100): int(index[2*idx + 1]*freq/100)])
        emg_squat.append(e)
    return emg_squat

def interpolate_squat_repetition(emg, index, freq):
    emg_squat_interp = np.zeros((int(len(index)/2), len(emg), int(2*freq)))
    emg_squat = divide_squat_repetition(emg, index, freq)
    for (i, e) in enumerate(emg_squat):
        x_start = np.arange(0, e[0].shape[0])
        x_interp = np.linspace(0, x_start[-1], int(2*freq))
        for m in range(len(emg)):
            f = interpolate.interp1d(x_start, e[m])
            emg_squat_interp[i, m, :] = f(x_interp)
    return emg_squat_interp

def compute_mean_squat_repetition(emg, index, freq):
    emg_squat_interp = interpolate_squat_repetition(emg, index, freq)
    mean_emg = np.zeros((len(emg), emg_squat_interp.shape[2]))
    std_emg = np.zeros((len(emg), emg_squat_interp.shape[2]))
    for m in range(len(emg)):
        mean_emg[m, :] = np.mean(emg_squat_interp[:, m, :], axis=0)
        std_emg[m, :] = np.std(emg_squat_interp[:, m, :], axis=0)
    return mean_emg, std_emg

def compute_symetry_ratio(emg):
    emg_sym = []
    for i in range(int(len(emg)/2)):
        emg_sym.append((emg[2*i] + 100)/(emg[2*i + 1] + 100))
    return emg_sym

def sort_data(emg, index, freq):
    sorted_data = []
    divide_repet = divide_squat_repetition(emg, index, freq)
    for repet in divide_repet:
        s=[]
        for r in repet:
            s = np.concatenate([s, r])
        sorted_data.append(np.sort(s))
    return sorted_data

def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis=-1), arr.std(axis=-1)



class emg:
    def __init__(self, name, higher_foot='R'):
        self.name = name
        self.higher_foot = higher_foot
        self.path = '../Data_test/' + name
        self.list_mvc_files = get_mvc_files(self.path)
        # self.list_exp_files = get_exp_files(path)
        self.list_exp_files = ['squat_controle.c3d', 'squat_3cm.c3d', 'squat_4cm.c3d', 'squat_5cm.c3d', 'squat_controle_post.c3d']
        # self.label_muscles_analog = get_labels_muscles(path + '/MVC/' + self.list_mvc_files[0])
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
        self.mvc_value[2] = self.mvc_value[3]
        self.mvc_value[12] = self.mvc_value[13]

        for file in self.list_exp_files:
            self.emg_normalized_exp.append(self.get_normalized_emg(file_path=self.path + '/Squats/' + file))

        self.events = markers(self.path).get_events()
        self.mid_events = markers(self.path).get_mid_events()
        self.mean, self.std = self.get_mean()
        self.mean_sym, self.std_sym = self.get_mean(symetry=True)
        self.RMSE = self.get_RMSE()
        self.DIFF = self.get_diff()
        self.R2 = self.get_R2()
        self.sym_phases = self.get_value_sym_per_phase()


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
        return emg_norm

    def get_mvc_value(self):
        mvc_value=[]
        for i in range(self.nb_mus):
            mvc_value.append(find_muscle_mvc_value(self.emg_filtered, idx_muscle=i, freq=self.freq))
        return mvc_value

    def get_mean(self, symetry=False):
        mean = []
        std = []
        for i in range(len(self.emg_normalized_exp)):
            if symetry:
                emg_sym = compute_symetry_ratio(self.emg_normalized_exp[i])
                A = compute_mean_squat_repetition(emg_sym, self.events[i], self.freq)
                mean.append(A[0])
                std.append(A[1])
            else:
                A = compute_mean_squat_repetition(self.emg_normalized_exp[i], self.events[i], self.freq)
                mean.append(A[0])
                std.append(A[1])
        return mean, std

    def get_RMSE(self):
        RMSE = []
        idx_control = self.list_exp_files.index('squat_controle.c3d')
        for m in self.mean:
            rmse = np.zeros(self.nb_mus)
            for i in range(self.nb_mus):
             rmse[i] = np.sqrt(np.mean((m[i, :] - self.mean[idx_control][i, :])**2))
            RMSE.append(rmse)
        return RMSE

    def get_R2(self):
        R2 = []
        idx_control = self.list_exp_files.index('squat_controle.c3d')
        for m in self.mean:
            r2 = np.zeros(self.nb_mus)
            for i in range(self.nb_mus):
                r2[i] = 1 - (np.sum((m[i, :] - self.mean[idx_control][i, :])**2) / np.sum((self.mean[idx_control][i, :] - np.mean(self.mean[idx_control][i, :]))**2))
            R2.append(r2)
        return R2

    def get_diff(self):
        DIFF = []
        idx_control = self.list_exp_files.index('squat_controle.c3d')
        for m in self.mean:
            diff = np.zeros(self.nb_mus)
            for i in range(self.nb_mus):
             diff[i] = np.mean(m[i, :] - self.mean[idx_control][i, :])
            DIFF.append(diff)
        return

    def get_value_sym_per_phase(self):
        sym_phases = []
        for (i, msym) in enumerate(self.mean_sym):
            sym_phases.append([np.mean(msym[:, :int(self.freq)], axis=1), np.mean(msym[:, int(1000):], axis=1)])
        return sym_phases

    def plot_sort_activation(self):
        mean_controle, std_controle = tolerant_mean(sort_data(self.emg_normalized_exp[0], self.events[0], self.freq))
        mean_perturbation, std_perturbation = tolerant_mean(sort_data(self.emg_normalized_exp[3], self.events[3], self.freq))

        plt.figure()
        plt.plot(np.linspace(0, 100, mean_controle.shape[0]), mean_controle, 'b')
        plt.fill_between(np.linspace(0, 100, mean_controle.shape[0]), mean_controle - std_controle, mean_controle + std_controle, color='b', alpha=0.2)

        plt.plot(np.linspace(0, 100, mean_perturbation.shape[0]), mean_perturbation, 'r')
        plt.fill_between(np.linspace(0, 100, mean_perturbation.shape[0]), mean_perturbation - std_perturbation, mean_perturbation + std_perturbation, color='r', alpha=0.2)

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
        fig.suptitle(self.name + "   " + title)
        for i in range(int(self.nb_mus/2)):
            axes[i].set_title(self.label_muscles[i])
            axes[i].plot(emg_data[2*i].time.data, emg_data[2*i].data)
            axes[i].plot(emg_data[2*i + 1].time.data, emg_data[2*i + 1].data)
            axes[i].set_ylim([0, 100])
        plt.legend(['right', 'left'])

    def plot_squat_repetition(self, title=None):
        if title is not None:
            idx = self.list_exp_files.index(title)
            emg_squat_interp = interpolate_squat_repetition(self.emg_normalized_exp[idx], self.events[idx], self.freq)
            fig, axes = plt.subplots(4, 5)
            axes = axes.flatten()
            fig.suptitle(self.name + "\nrepetition " + title)
            for i in range(self.nb_mus):
                axes[i].set_title(self.label_muscles_analog[i])
                axes[i].plot(np.linspace(0, 100, emg_squat_interp.shape[2]), emg_squat_interp[:, i, :].T)
                axes[i].set_xlim([0, 100])
                axes[i].set_ylim([0, 100])
                if i > 14:
                    axes[i].set_xlabel('normalized time (%)')
            axes[0].set_ylabel('activation (%)')
            axes[5].set_ylabel('activation (%)')
            axes[10].set_ylabel('activation (%)')
            axes[15].set_ylabel('activation (%)')
        else:
            for (t, title) in enumerate(self.list_exp_files):
                emg_squat_interp = interpolate_squat_repetition(self.emg_normalized_exp[t], self.events[t], self.freq)
                fig, axes = plt.subplots(4, 5)
                axes = axes.flatten()
                fig.suptitle(self.name + "\nrepetition " + title)
                for i in range(self.nb_mus):
                    axes[i].set_title(self.label_muscles_analog[i])
                    axes[i].plot(np.linspace(0, 100, emg_squat_interp.shape[2]), emg_squat_interp[:, i, :].T)
                    axes[i].set_xlim([0, 100])
                    axes[i].set_ylim([0, 100])
                    if i > 14:
                        axes[i].set_xlabel('normalized time (%)')
                axes[0].set_ylabel('activation (%)')
                axes[5].set_ylabel('activation (%)')
                axes[10].set_ylabel('activation (%)')
                axes[15].set_ylabel('activation (%)')


    def plot_squat_mean(self, title=None):
        if title is not None:
            idx = self.list_exp_files.index(title)
            abscisse = np.linspace(0, 100, self.mean[idx].shape[1])
            fig, axes = plt.subplots(2, 5)
            axes = axes.flatten()
            fig.suptitle(self.name + "\nmean " + title)
            for i in range(int(self.nb_mus/2)):
                axes[i].set_title(self.label_muscles[i])
                axes[i].plot(abscisse, self.mean[idx][2*i, :], 'r')
                axes[i].plot(abscisse, self.mean[idx][2*i + 1, :], 'b')

                axes[i].fill_between(abscisse,
                                     self.mean[idx][2 * i, :] - self.std[idx][2 * i, :],
                                     self.mean[idx][2 * i, :] + self.std[idx][2 * i, :], color='r', alpha=0.2)
                axes[i].fill_between(abscisse,
                                     self.mean[idx][2 * i + 1] - self.std[idx][2 * i + 1, :],
                                     self.mean[idx][2 * i + 1] + self.std[idx][2 * i + 1, :], color='b', alpha=0.2)
                axes[i].plot([50, 50], [0, 100], 'k--')
                axes[i].set_xlim([0, 100])
                axes[i].set_ylim([0, 100])
                if i > 4:
                    axes[i].set_xlabel('normalized time (%)')
            axes[0].set_ylabel('activation (%)')
            axes[5].set_ylabel('activation (%)')
            plt.legend(['right', 'left'])
        else:
            abscisse = np.linspace(0, 100, self.mean[0].shape[1])
            for (t, title) in enumerate(self.list_exp_files):
                fig, axes = plt.subplots(2, 5)
                axes = axes.flatten()
                fig.suptitle(self.name + "\nmean " + title)
                for i in range(int(self.nb_mus / 2)):
                    axes[i].set_title(self.label_muscles[i])
                    axes[i].plot(abscisse, self.mean[t][2 * i, :], 'r')
                    axes[i].plot(abscisse, self.mean[t][2 * i + 1, :], 'b')

                    axes[i].fill_between(abscisse,
                                         self.mean[t][2 * i, :] - self.std[t][2 * i, :],
                                         self.mean[t][2 * i, :] + self.std[t][2 * i, :], color='r', alpha=0.2)
                    axes[i].fill_between(abscisse,
                                         self.mean[t][2 * i + 1] - self.std[t][2 * i + 1, :],
                                         self.mean[t][2 * i + 1] + self.std[t][2 * i + 1, :], color='b', alpha=0.2)
                    axes[i].plot([50, 50], [0, 100],'k--')
                    axes[i].set_xlim([0, 100])
                    axes[i].set_ylim([0, 100])
                    if i > 4:
                        axes[i].set_xlabel('normalized time (%)')
                axes[0].set_ylabel('activation (%)')
                axes[5].set_ylabel('activation (%)')
                plt.legend(['right', 'left'])

    def plot_squat_comparison(self):
        abscisse = np.linspace(0, 100, self.mean[0].shape[1])
        color_plot = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown']
        fig, axes = plt.subplots(4, 5)
        axes = axes.flatten()
        fig.suptitle(self.name + "\n comparison")
        for i in range(self.nb_mus):
            axes[i].set_title(self.label_muscles_analog[i])
            axes[i].text(80, 90, 'DIFF')
            for (a, m) in enumerate(self.mean):
                axes[i].plot(abscisse, m[i, :])
                axes[i].text(80, 82-7*a, str(round(self.DIFF[a][i], 2)), color=color_plot[a])
            axes[i].set_xlim([0, 100])
            axes[i].set_ylim([0, 100])
            if i > 14:
                axes[i].set_xlabel('normalized time (%)')
        axes[0].set_ylabel('activation (%)')
        axes[5].set_ylabel('activation (%)')
        axes[10].set_ylabel('activation (%)')
        axes[15].set_ylabel('activation (%)')
        fig.legend(self.list_exp_files)

    def plot_squat_comparison_5cm(self):
        abscisse = np.linspace(0, 100, self.mean[0].shape[1])
        fig, axes = plt.subplots(4, 5)
        axes = axes.flatten()
        fig.suptitle(self.name + "\n comparison")
        for i in range(self.nb_mus):
            axes[i].set_title(self.label_muscles_analog[i])
            axes[i].text(80, 90, 'RMSE')
            # axes[i].text(80, 70, 'R2')
            axes[i].plot(abscisse, self.mean[0][i, :], color='tab:blue')
            axes[i].plot(abscisse, self.mean[3][i, :], color='tab:red')
            axes[i].text(80, 82, str(round(self.RMSE[3][i], 2)), color='tab:red')
            # axes[i].text(80, 62, str(round(self.R2[3][i], 2)), color='tab:red')
            axes[i].set_xlim([0, 100])
            axes[i].set_ylim([0, 100])
            if i > 14:
                axes[i].set_xlabel('normalized time (%)')
        axes[0].set_ylabel('activation (%)')
        axes[5].set_ylabel('activation (%)')
        axes[10].set_ylabel('activation (%)')
        axes[15].set_ylabel('activation (%)')
        fig.legend(['controle', '5cm'])

    def plot_squat_mean_symetry(self, title=None):
        if title is not None:
            idx = self.list_exp_files.index(title)
            abscisse = np.linspace(0, 100, self.mean_sym[idx].shape[1])
            fig, axes = plt.subplots(2, 5)
            axes = axes.flatten()
            fig.suptitle(self.name + "\nmean ratio R/L " + title)
            for i in range(int(self.nb_mus/2)):
                axes[i].set_title(self.label_muscles[i])
                axes[i].plot(abscisse, self.mean_sym[idx][i, :], 'r')
                axes[i].fill_between(abscisse,
                                     self.mean_sym[idx][i, :] - self.std_sym[idx][i, :],
                                     self.mean_sym[idx][i, :] + self.std_sym[idx][i, :], color='r', alpha=0.2)
                axes[i].plot([50, 50], [0, 10], 'k--')
                axes[i].set_xlim([0, 100])
                axes[i].set_ylim([0, 10])
                axes[i].text(20, 5, str(round(self.sym_phases[idx][0][i], 2)))
                axes[i].text(60, 5, str(round(self.sym_phases[idx][1][i], 2)))
                if i > 4:
                    axes[i].set_xlabel('normalized time (%)')
            axes[0].set_ylabel('activation (%)')
            axes[5].set_ylabel('activation (%)')
        else:
            abscisse = np.linspace(0, 100, self.mean[0].shape[1])
            for (t, title) in enumerate(self.list_exp_files):
                fig, axes = plt.subplots(2, 5)
                axes = axes.flatten()
                fig.suptitle(self.name + "\nmean ratio R/L " + title)
                for i in range(int(self.nb_mus / 2)):
                    axes[i].set_title(self.label_muscles[i])
                    axes[i].plot(abscisse, self.mean_sym[t][i, :], 'r')
                    axes[i].fill_between(abscisse,
                                         self.mean_sym[t][i, :] - self.std_sym[t][i, :],
                                         self.mean_sym[t][i, :] + self.std_sym[t][i, :], color='r', alpha=0.2)
                    axes[i].plot([50, 50], [0, 10], 'k--')
                    axes[i].text(20, 5, str(round(self.sym_phases[t][0][i], 2)))
                    axes[i].text(60, 5, str(round(self.sym_phases[t][1][i], 2)))
                    axes[i].set_xlim([0, 100])
                    axes[i].set_ylim([0, 10])
                    if i > 4:
                        axes[i].set_xlabel('normalized time (%)')
                axes[0].set_ylabel('activation (%)')
                axes[5].set_ylabel('activation (%)')

    def plot_assymetry_comparison_5cm(self):
        abscisse = np.linspace(0, 100, self.mean[0].shape[1])
        fig, axes = plt.subplots(2, 5)
        axes = axes.flatten()
        fig.suptitle(self.name + "\n comparison symetry 5 cm")
        for i in range(int(self.nb_mus / 2)):
            axes[i].set_title(self.label_muscles[i])
            axes[i].plot(abscisse, self.mean_sym[0][i, :], 'b')
            axes[i].plot(abscisse, self.mean_sym[3][i, :], 'r')
            axes[i].plot(abscisse, self.mean_sym[3][i, :] - self.mean_sym[0][i, :], 'g')
            axes[i].plot([50, 50], [0, 3], 'k--')
            axes[i].text(20, 2, str(round(self.sym_phases[0][0][i], 2)), color='b')
            axes[i].text(60, 2, str(round(self.sym_phases[0][1][i], 2)), color='b')
            axes[i].text(20, 1.5, str(round(self.sym_phases[3][0][i], 2)), color='r')
            axes[i].text(60, 1.5, str(round(self.sym_phases[3][1][i], 2)), color='r')
            axes[i].text(20, 0.5, str(round(self.sym_phases[3][0][i] - self.sym_phases[0][0][i], 2)), color='g')
            axes[i].text(60, 0.5, str(round(self.sym_phases[3][1][i] - self.sym_phases[0][0][i], 2)), color='g')
            axes[i].set_xlim([0, 100])
            axes[i].set_ylim([0, 3])
            if i > 4:
                axes[i].set_xlabel('normalized time (%)')
        axes[0].set_ylabel('activation (%)')
        axes[5].set_ylabel('activation (%)')
        plt.legend(['controle', '5cm', 'difference'])
