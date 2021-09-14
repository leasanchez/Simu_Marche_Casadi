import matplotlib.pyplot as plt
import numpy as np
import biorbd_casadi as biorbd
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

def compute_symetry_ratio(data, higher_foot):
    # compute difference between the two legs
    # input : data : initial data for all dofs (all repetitions)
    #         higher_foot : which foot is elevated ('R' or 'L')
    # output : data_sym :  new data set where we computed the difference between the intervention leg and the control

    data_sym = []
    for d in data:
        d_sym = []
        for r in d:
            a = np.zeros((15, r.shape[1]))
            a[:9] = r[:9]
            if higher_foot == 'R':
                a[9:12] = (r[9:12] + 100) / (r[15:18] + 100) # hip
                a[12] = (r[12] + 100) / (r[18] + 100) # knee
                a[13:15] = (r[13:15] + 100) / (r[19:20] + 100)  # ankle
            else:
                a[9:12] = r[15:18] / r[9:12]  # hip
                a[12] = r[18] / r[12] # knee
                a[13:15] = r[19:20] / r[13:15]  # ankle
            d_sym.append(a)
        data_sym.append(d_sym)
    return data_sym

def find_pic(data, index):
    pic=np.zeros(len(data))
    for (i, d) in enumerate(data):
        pic[i] = np.max(np.abs(d[index, :])) * 180/np.pi
    return np.mean(pic)

class kinematic:
    def __init__(self, name, higher_foot='R'):
        self.name = name
        self.higher_foot = higher_foot
        self.path = '../Data_test/' + name
        self.model = biorbd.Model(self.path + "/" + name + ".bioMod")
        self.list_exp_files = ['squat_controle', 'squat_3cm', 'squat_4cm', 'squat_5cm', 'squat_controle_post']
        self.q_name = get_q_name(self.model)
        self.label_q = ["pelvis Tx", "pelvis Ty", "pelvis Tz", "pelvis Rx", "pelvis Ry", "pelvis Rz",
                        "tronc Rx", "tronc Ry", "tronc Rz",
                        "hanche Rx", "hanche Ry", "hanche Rz",
                        "genou Rz",
                        "cheville Rx", "cheville Rz"]
        self.events = markers(self.path).get_events()[0]
        self.q = self.get_q()
        self.q_mean, self.q_std = self.get_mean(self.q)
        self.qdiff_mean, self.qdiff_std = self.get_mean(self.q, diff=True)
        self.qdot = self.get_qdot()
        if self.higher_foot == 'R':
            self.pic_flexion_knee = [self.get_pic_value(index=12), self.get_pic_value(index=18)]
            self.pic_flexion_hip = [self.get_pic_value(index=11), self.get_pic_value(index=17)]
            self.pic_flexion_ankle = [self.get_pic_value(index=14), self.get_pic_value(index=20)]
        else :
            self.pic_flexion_knee = [self.get_pic_value(index=18), self.get_pic_value(index=12)]
            self.pic_flexion_hip = [self.get_pic_value(index=17), self.get_pic_value(index=11)]
            self.pic_flexion_ankle = [self.get_pic_value(index=20), self.get_pic_value(index=14)]

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

    def get_mean(self, data, diff=False):
        mean = []
        std = []
        if diff:
            data = compute_symetry_ratio(data, self.higher_foot)
        for (i, d) in enumerate(data):
            a = compute_mean(d)
            mean.append(a[0])
            std.append(a[1])
        return mean, std

    def get_pic_value(self, index):
        pic_value = []
        for q in self.q:
            pic_value.append(find_pic(q, index))
        return pic_value


    def plot_squat_repetition(self, title=None):
        if title is not None:
            idx = self.list_exp_files.index(title)
            squat_interp = interpolate_repetition(self.q[idx])
            fig, axes = plt.subplots(3, 7)
            axes = axes.flatten()
            fig.suptitle(self.name + "\nrepetition " + title)
            for i in range(self.model.nbQ()):
                axes[i].set_title(self.q_name[i])
                if (i > 3):
                    axes[i].plot(np.linspace(0, 100, squat_interp.shape[2]), (squat_interp[:, i, :]*180/np.pi).T)
                else:
                    axes[i].plot(np.linspace(0, 100, squat_interp.shape[2]), squat_interp[:, i, :].T)
                axes[i].set_xlim([0, 100])
                if i > 13:
                    axes[i].set_xlabel('normalized time (%)')
            axes[0].set_ylabel('q (m/degré)')
            axes[7].set_ylabel('q (m/degré)')
            axes[14].set_ylabel('q (m/degré)')
        else:
            for (t, title) in enumerate(self.list_exp_files):
                squat_interp = interpolate_repetition(self.q[t])
                fig, axes = plt.subplots(3, 7)
                axes = axes.flatten()
                fig.suptitle(self.name + "\nrepetition " + title)
                for i in range(self.model.nbQ()):
                    axes[i].set_title(self.q_name[i])
                    if (i > 3):
                        axes[i].plot(np.linspace(0, 100, squat_interp.shape[2]),(squat_interp[:, i, :] * 180 / np.pi).T)
                    else:
                        axes[i].plot(np.linspace(0, 100, squat_interp.shape[2]), squat_interp[:, i, :].T)
                    axes[i].set_xlim([0, 100])
                    if i > 13:
                        axes[i].set_xlabel('normalized time (%)')
                axes[0].set_ylabel('q (m/degré)')
                axes[7].set_ylabel('q (m/degré)')
                axes[14].set_ylabel('q (m/degré)')

    def plot_squat_mean_leg(self, title=None):
        if title is not None:
            t = self.list_exp_files.index(title)
            abscisse = np.linspace(0, 100, self.q_mean[t].shape[1])
            fig, axes = plt.subplots(2, 3)
            axes = axes.flatten()
            fig.suptitle(self.name + "\nmean " + title)
            for i in range(len(axes)):
                axes[i].set_title(self.label_q[i + 9])
                if (i==0 or i==1 or i==4):
                    axes[i].plot(abscisse, self.q_mean[t][i + 9, :] * 180 / np.pi, 'r')
                    axes[i].plot(abscisse, -self.q_mean[t][i + 9 + 6, :] * 180 / np.pi, 'b')
                    axes[i].fill_between(abscisse,
                                         self.q_mean[t][i + 9, :] * 180 / np.pi - self.q_std[t][i + 9, :] * 180 / np.pi,
                                         self.q_mean[t][i + 9, :] * 180 / np.pi + self.q_std[t][i + 9, :] * 180 / np.pi,
                                         color='r', alpha=0.2)
                    axes[i].fill_between(abscisse,
                                         -self.q_mean[t][i + 9 + 6, :] * 180 / np.pi - self.q_std[t][i + 9 + 6,:] * 180 / np.pi,
                                         -self.q_mean[t][i + 9 + 6, :] * 180 / np.pi + self.q_std[t][i + 9 + 6,:] * 180 / np.pi,
                                         color='b', alpha=0.2)
                else:
                    axes[i].plot(abscisse, self.q_mean[t][i + 9, :] * 180 / np.pi, 'r')
                    axes[i].plot(abscisse, self.q_mean[t][i + 9 + 6, :] * 180 / np.pi, 'b')
                    axes[i].fill_between(abscisse,
                                         self.q_mean[t][i + 9, :] * 180 / np.pi - self.q_std[t][i + 9, :] * 180 / np.pi,
                                         self.q_mean[t][i + 9, :] * 180 / np.pi + self.q_std[t][i + 9, :] * 180 / np.pi,
                                         color='r', alpha=0.2)
                    axes[i].fill_between(abscisse,
                                         self.q_mean[t][i + 9 + 6, :] * 180 / np.pi - self.q_std[t][i + 9 + 6, :] * 180 / np.pi,
                                         self.q_mean[t][i + 9 + 6, :] * 180 / np.pi + self.q_std[t][i + 9 + 6, :] * 180 / np.pi,
                                         color='b', alpha=0.2)
                axes[i].plot([50, 50],
                                 [min(self.q_mean[t][i + 9, :] * 180 / np.pi - self.q_std[t][i + 9, :] * 180 / np.pi),
                                  max(self.q_mean[t][i + 9, :] * 180 / np.pi + self.q_std[t][i + 9, :] * 180 / np.pi)],
                                 'k--')
                axes[i].set_xlim([0, 100])
                if i > 2:
                    axes[i].set_xlabel('normalized time (%)')
            axes[0].set_ylabel('q (m/deg)')
            axes[3].set_ylabel('q (m/deg)')
            plt.legend(['right', 'left'])
        else:
            abscisse = np.linspace(0, 100, self.q_mean[0].shape[1])
            for (t, title) in enumerate(self.list_exp_files):
                fig, axes = plt.subplots(2, 3)
                axes = axes.flatten()
                fig.suptitle(self.name + "\nmean " + title)
                for i in range(len(axes)):
                    axes[i].set_title(self.label_q[i + 9])
                    if (i==0 or i==1 or i==4):
                        axes[i].plot(abscisse, self.q_mean[t][i + 9, :] * 180 / np.pi, 'r')
                        axes[i].plot(abscisse, -self.q_mean[t][i + 9 + 6, :] * 180 / np.pi, 'b')
                        axes[i].fill_between(abscisse,
                                             self.q_mean[t][i + 9, :] * 180 / np.pi - self.q_std[t][i + 9,
                                                                                      :] * 180 / np.pi,
                                             self.q_mean[t][i + 9, :] * 180 / np.pi + self.q_std[t][i + 9,
                                                                                      :] * 180 / np.pi,
                                             color='r', alpha=0.2)
                        axes[i].fill_between(abscisse,
                                             -self.q_mean[t][i + 9 + 6, :] * 180 / np.pi - self.q_std[t][i + 9 + 6,
                                                                                           :] * 180 / np.pi,
                                             -self.q_mean[t][i + 9 + 6, :] * 180 / np.pi + self.q_std[t][i + 9 + 6,
                                                                                           :] * 180 / np.pi,
                                             color='b', alpha=0.2)
                    else:
                        axes[i].plot(abscisse, self.q_mean[t][i + 9, :] * 180 / np.pi, 'r')
                        axes[i].plot(abscisse, self.q_mean[t][i + 9 + 6, :] * 180 / np.pi, 'b')
                        axes[i].fill_between(abscisse,
                                             self.q_mean[t][i + 9, :] * 180 / np.pi - self.q_std[t][i + 9,
                                                                                      :] * 180 / np.pi,
                                             self.q_mean[t][i + 9, :] * 180 / np.pi + self.q_std[t][i + 9,
                                                                                      :] * 180 / np.pi,
                                             color='r', alpha=0.2)
                        axes[i].fill_between(abscisse,
                                             self.q_mean[t][i + 9 + 6, :] * 180 / np.pi - self.q_std[t][i + 9 + 6,
                                                                                          :] * 180 / np.pi,
                                             self.q_mean[t][i + 9 + 6, :] * 180 / np.pi + self.q_std[t][i + 9 + 6,
                                                                                          :] * 180 / np.pi,
                                             color='b', alpha=0.2)
                    axes[i].plot([50, 50],
                                 [min(self.q_mean[t][i + 9, :] * 180 / np.pi - self.q_std[t][i + 9, :] * 180 / np.pi),
                                  max(self.q_mean[t][i + 9, :] * 180 / np.pi + self.q_std[t][i + 9, :] * 180 / np.pi)],
                                 'k--')
                    axes[i].set_xlim([0, 100])
                    if i > 2:
                        axes[i].set_xlabel('normalized time (%)')
                axes[0].set_ylabel('q (m/deg)')
                axes[3].set_ylabel('q (m/deg)')
                plt.legend(['right', 'left'])

    def plot_diff_mean_leg(self, title):
        t = self.list_exp_files.index(title)
        abscisse = np.linspace(0, 100, self.qdiff_mean[t].shape[1])
        fig, axes = plt.subplots(2, 3)
        axes = axes.flatten()
        fig.suptitle(self.name + "\nmean difference with intervention feet : " + self.higher_foot + "\n" + title)
        for i in range(len(axes)):
            axes[i].set_title(self.label_q[i + 9])
            if (i == 0 or i == 1 or i == 4):
                axes[i].plot(abscisse, self.qdiff_mean[t][i + 9, :], 'k')
                axes[i].fill_between(abscisse,
                                     self.qdiff_mean[t][i + 9, :] - self.qdiff_std[t][i + 9, :],
                                     self.qdiff_mean[t][i + 9, :] + self.qdiff_std[t][i + 9, :],
                                     color='k', alpha=0.2)
            else:
                axes[i].plot(abscisse, self.qdiff_mean[t][i + 9, :], 'k')
                axes[i].fill_between(abscisse,
                                     self.qdiff_mean[t][i + 9, :] - self.qdiff_std[t][i + 9, :],
                                     self.qdiff_mean[t][i + 9, :] + self.qdiff_std[t][i + 9, :],
                                     color='k', alpha=0.2)
            axes[i].plot([50, 50],
                         [min(self.qdiff_mean[t][i + 9, :] - self.qdiff_std[t][i + 9, :]),
                          max(self.qdiff_mean[t][i + 9, :] + self.qdiff_std[t][i + 9, :])],
                         'k--')
            axes[i].set_xlim([0, 100])
            if i > 2:
                axes[i].set_xlabel('normalized time (%)')
        axes[0].set_ylabel('q (m/deg)')
        axes[3].set_ylabel('q (m/deg)')

    def plot_squat_mean_pelvis(self, title=None):
        if title is not None:
            t = self.list_exp_files.index(title)
            abscisse = np.linspace(0, 100, self.q_mean[t].shape[1])
            fig, axes = plt.subplots(2, 3)
            axes = axes.flatten()
            fig.suptitle(self.name + "\nmean " + title)
            for i in range(len(axes)):
                axes[i].set_title(self.label_q[i])
                if (i > 2):
                    axes[i].plot(abscisse, self.q_mean[t][i, :] * 180 / np.pi, 'r')
                    axes[i].fill_between(abscisse,
                                         self.q_mean[t][i, :] * 180 / np.pi - self.q_std[t][i, :] * 180 / np.pi,
                                         self.q_mean[t][i, :] * 180 / np.pi + self.q_std[t][i, :] * 180 / np.pi,
                                         color='r', alpha=0.2)
                    axes[i].plot([50, 50],
                                 [min(self.q_mean[t][i, :] * 180 / np.pi - self.q_std[t][i, :] * 180 / np.pi), max(self.q_mean[t][i, :] * 180 / np.pi + self.q_std[t][i, :] * 180 / np.pi)],
                                 'k--')
                else:
                    axes[i].plot(abscisse, self.q_mean[t][i, :], 'r')
                    axes[i].fill_between(abscisse,
                                         self.q_mean[t][i, :] - self.q_std[t][i, :],
                                         self.q_mean[t][i, :] + self.q_std[t][i, :],
                                         color='r', alpha=0.2)
                    axes[i].plot([50, 50],
                                 [min(self.q_mean[t][i, :] - self.q_std[t][i, :]), max(self.q_mean[t][i, :] + self.q_std[t][i, :])],
                                 'k--')
                axes[i].set_xlim([0, 100])
                if i > 2:
                    axes[i].set_xlabel('normalized time (%)')
            axes[0].set_ylabel('q (m/deg)')
            axes[3].set_ylabel('q (m/deg)')
        else:
            abscisse = np.linspace(0, 100, self.q_mean[0].shape[1])
            for (t, title) in enumerate(self.list_exp_files):
                fig, axes = plt.subplots(2, 3)
                axes = axes.flatten()
                fig.suptitle(self.name + "\nmean " + title)
                for i in range(len(axes)):
                    axes[i].set_title(self.label_q[i])
                    if (i > 2):
                        axes[i].plot(abscisse, self.q_mean[t][i, :] * 180 / np.pi, 'r')
                        axes[i].fill_between(abscisse,
                                             self.q_mean[t][i, :] * 180 / np.pi - self.q_std[t][i, :] * 180 / np.pi,
                                             self.q_mean[t][i, :] * 180 / np.pi + self.q_std[t][i, :] * 180 / np.pi,
                                             color='r', alpha=0.2)
                        axes[i].plot([50, 50],
                                     [min(self.q_mean[t][i, :] * 180 / np.pi - self.q_std[t][i, :] * 180 / np.pi),
                                      max(self.q_mean[t][i, :] * 180 / np.pi + self.q_std[t][i, :] * 180 / np.pi)],
                                     'k--')
                    else:
                        axes[i].plot(abscisse, self.q_mean[t][i, :], 'r')
                        axes[i].fill_between(abscisse,
                                             self.q_mean[t][i, :] - self.q_std[t][i, :],
                                             self.q_mean[t][i, :] + self.q_std[t][i, :],
                                             color='r', alpha=0.2)
                        axes[i].plot([50, 50],
                                     [min(self.q_mean[t][i, :] - self.q_std[t][i, :]),
                                      max(self.q_mean[t][i, :] + self.q_std[t][i, :])],
                                     'k--')
                    axes[i].set_xlim([0, 100])
                    if i > 2:
                        axes[i].set_xlabel('normalized time (%)')
                axes[0].set_ylabel('q (m/deg)')
                axes[3].set_ylabel('q (m/deg)')

    def plot_squat_mean_torso(self, title=None):
        if title is not None:
            t = self.list_exp_files.index(title)
            abscisse = np.linspace(0, 100, self.q_mean[t].shape[1])
            fig, axes = plt.subplots(1, 3)
            axes = axes.flatten()
            fig.suptitle(self.name + "\nmean " + title)
            for i in range(len(axes)):
                axes[i].set_title(self.label_q[i + 6])
                axes[i].plot(abscisse, self.q_mean[t][i + 6, :] * 180 / np.pi, 'r')
                axes[i].fill_between(abscisse,
                                     self.q_mean[t][i + 6, :] * 180 / np.pi - self.q_std[t][i + 6, :] * 180 / np.pi,
                                     self.q_mean[t][i + 6, :] * 180 / np.pi + self.q_std[t][i + 6, :] * 180 / np.pi,
                                     color='r', alpha=0.2)
                axes[i].plot([50, 50],
                             [min(self.q_mean[t][i + 6, :] * 180 / np.pi - self.q_std[t][i + 6, :] * 180 / np.pi),
                              max(self.q_mean[t][i + 6, :] * 180 / np.pi + self.q_std[t][i + 6, :] * 180 / np.pi)],
                             'k--')
                axes[i].set_xlim([0, 100])
                axes[i].set_xlabel('normalized time (%)')
            axes[0].set_ylabel('q (m/deg)')
        else:
            abscisse = np.linspace(0, 100, self.q_mean[0].shape[1])
            for (t, title) in enumerate(self.list_exp_files):
                fig, axes = plt.subplots(1, 3)
                axes = axes.flatten()
                fig.suptitle(self.name + "\nmean " + title)
                for i in range(len(axes)):
                    axes[i].set_title(self.label_q[i + 5])
                    axes[i].plot(abscisse, self.q_mean[t][i + 5, :] * 180 / np.pi, 'r')
                    axes[i].fill_between(abscisse,
                                         self.q_mean[t][i + 5, :] * 180 / np.pi - self.q_std[t][i + 5, :] * 180 / np.pi,
                                         self.q_mean[t][i + 5, :] * 180 / np.pi + self.q_std[t][i + 5, :] * 180 / np.pi,
                                         color='r', alpha=0.2)
                    axes[i].plot([50, 50],
                                 [min(self.q_mean[t][i + 5, :] * 180 / np.pi - self.q_std[t][i + 5, :] * 180 / np.pi),
                                  max(self.q_mean[t][i + 5, :] * 180 / np.pi + self.q_std[t][i + 5, :] * 180 / np.pi)],
                                 'k--')
                    axes[i].set_xlim([0, 100])
                    axes[i].set_xlabel('normalized time (%)')
                axes[0].set_ylabel('q (m/deg)')