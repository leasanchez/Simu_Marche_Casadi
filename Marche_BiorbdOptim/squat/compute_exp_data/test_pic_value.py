import matplotlib.pyplot as plt
import numpy as np
import spm1d
import biorbd_casadi as biorbd
from KINEMATIC import kinematic

def get_q_name(model):
    q_name = []
    for q in range(model.nbSegment()):
        for d in range(model.segment(q).nbQ()):
            q_name.append(f"{model.segment(q).name().to_string()}_{model.segment(q).nameDof(d).to_string()}")
    return q_name

def get_data_ratio(q_idx):
    list_condition = []  # 0 : control, 1 : 3cm, 2 : 4cm, 3 : 5cm
    subject = []
    y = np.ndarray((7 * 4, 200))

    for i in range(len(list_name)):
        subject_dir = "/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/squat/Data_test/" \
                      + list_name[i]
        idx = [q_idx, q_idx + 6] if higher_foot_list[i] == 'L' else [q_idx + 6, q_idx]
        for (j, c) in enumerate(condition):
            subject = np.hstack([subject, i])
            list_condition = np.hstack([list_condition, j])
            q = np.load(subject_dir + "/kalman/" + c + "_mean.npy")
            y[4 * i + j] = q[idx[1]] / q[idx[0]]
    return y, list_condition, subject

def get_data_ratio_repet(q_idx):
    n_repet=5
    a=0
    list_condition = []  # 0 : control, 1 : 3cm, 2 : 4cm, 3 : 5cm
    subject = []
    y = np.ndarray((7 * 4 * n_repet, 200))

    for i in range(len(list_name)):
        subject_dir = "/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/squat/Data_test/" \
                      + list_name[i]
        idx = [q_idx, q_idx + 6] if higher_foot_list[i] == 'L' else [q_idx + 6, q_idx]
        for (j, c) in enumerate(condition):
            q = np.load(subject_dir + "/kalman/" + c + "_repet.npy")
            idx_repet = int(q.shape[0] - n_repet) if q.shape[0] > n_repet else 0
            subject = np.hstack([subject, n_repet * [i]])
            list_condition = np.hstack([list_condition, n_repet * [j]])
            y[a:a + n_repet] = np.vstack(q[idx_repet:, idx[1], :]) / np.vstack(q[idx_repet:, idx[0], :])
            a+=n_repet
    return y, list_condition, subject

def get_data_mean(q_idx):
    list_condition = []  # 0 : control, 1 : 3cm, 2 : 4cm, 3 : 5cm
    leg = []  # 0 : control, # 1 : elevated
    subject = []
    y = np.ndarray((2 * 7 * 4, 200))

    for i in range(len(list_name)):
        subject_dir = "/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/squat/Data_test/" \
                      + list_name[i]
        idx = [q_idx, q_idx + 6] if higher_foot_list[i] == 'L' else [q_idx + 6, q_idx]
        for (j, c) in enumerate(condition):
            subject = np.hstack([subject, [i, i]])
            list_condition = np.hstack([list_condition, [j, j]])
            leg = np.hstack([leg, [0, 1]])
            q = np.load(subject_dir + "/kalman/" + c + "_mean.npy")
            y[(8 * i + 2 * j): (8 * i + 2 * j + 2)] = q[idx]
    return y, list_condition, leg, subject

def get_data_repet(q_idx):
    n_repet = 5
    a = 0
    list_condition = []  # 0 : control, 1 : 3cm, 2 : 4cm, 3 : 5cm
    leg = []  # 0 : control, # 1 : elevated
    subject = []
    y_repet = np.ndarray((2 * 7 * 4 * n_repet, 200))

    for i in range(len(list_name)):
        subject_dir = "/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/squat/Data_test/" \
                      + list_name[i]
        idx = [q_idx, q_idx + 6] if higher_foot_list[i] == 'L' else [q_idx + 6, q_idx]
        for (j, c) in enumerate(condition):
            q = np.load(subject_dir + "/kalman/" + c + "_repet.npy")
            idx_repet = int(q.shape[0] - n_repet) if q.shape[0] > n_repet else 0
            subject = np.hstack([subject, n_repet * [i, i]])
            list_condition = np.hstack([list_condition, n_repet * [j, j]])
            leg = np.hstack([leg, n_repet * [0, 1]])
            y_repet[a: a + 2 * n_repet] = np.vstack(q[idx_repet:, idx, :])
            a += 2 * n_repet
    return y_repet, list_condition, leg, subject

def anova_one_way(q_idx, alpha, repet=False):
    y, list_condition, subject = get_data_ratio_repet(q_idx) if repet else get_data_ratio(q_idx)
    Frm = spm1d.stats.anova1rm(y, list_condition, subject, equal_var=True)
    Firm = Frm.inference(alpha)

    # (2) Plot:
    x = np.linspace(0, 100, 200)
    plt.close('all')
    plt.figure(q_name[q_idx] + ' Ratio', figsize=(8, 3.5))
    ax0 = plt.axes((0.1, 0.15, 0.35, 0.8))
    ax1 = plt.axes((0.55, 0.15, 0.35, 0.8))
    ### plot mean subject trajectories:
    ax0.plot(x, y[list_condition == 0].T, 'b')  # controle
    ax0.plot(x, y[list_condition == 1].T, 'k')  # 3cm
    ax0.plot(x, y[list_condition == 2].T, 'r')  # 4cm
    ax0.plot(x, y[list_condition == 3].T, 'g')  # 5cm
    ax0.set_xlim([0, 100])
    ax0.set_xlabel('Time (%)')

    ### plot SPM results:
    Firm.plot(ax=ax1, color='r', facecolor=(0.8, 0.3, 0.3))
    plt.show()

def anova_2_way(q_idx, alpha, repet=False):
    y, list_condition, leg, subject = get_data_repet(q_idx) if repet else get_data_mean(q_idx)

    FFrm = spm1d.stats.anova2rm(y, list_condition, leg, subject)
    FFirm = FFrm.inference(alpha)

    # (2) Plot:
    x = np.linspace(0, 100, 200)
    plt.close('all')
    plt.figure(q_name[q_idx] + ' Mean condition', figsize=(8, 3.5))
    ax0 = plt.axes((0.1, 0.15, 0.35, 0.8))
    ax1 = plt.axes((0.55, 0.15, 0.35, 0.8))
    ### plot mean subject trajectories:
    ax0.plot(x, y[list_condition == 0].T, 'b')  # controle
    ax0.plot(x, y[list_condition == 1].T, 'k')  # 3cm
    ax0.plot(x, y[list_condition == 2].T, 'r')  # 4cm
    ax0.plot(x, y[list_condition == 3].T, 'g')  # 5cm
    ax0.set_xlim([0, 100])
    ax0.set_xlabel('Time (%)')
    ### plot SPM results:
    FFirm[0].plot(ax=ax1, color='r', facecolor=(0.8, 0.3, 0.3))

    plt.figure(q_name[q_idx] + ' Mean leg', figsize=(8, 3.5))
    ax0 = plt.axes((0.1, 0.15, 0.35, 0.8))
    ax1 = plt.axes((0.55, 0.15, 0.35, 0.8))
    ### plot mean subject trajectories:
    ax0.plot(x, y[leg == 0].T, 'b')  # elevated
    ax0.plot(x, y[leg == 1].T, 'k')  # control
    ax0.set_xlim([0, 100])
    ax0.set_xlabel('Time (%)')
    ### plot SPM results:
    FFirm[1].plot(ax=ax1, color='r', facecolor=(0.8, 0.3, 0.3))

    plt.figure(q_name[q_idx] + ' Mean interaction leg - condition', figsize=(8, 3.5))
    FFirm[2].plot(color='r', facecolor=(0.8, 0.3, 0.3))
    plt.show()




list_name = ['AmeCeg', 'AnaLau', 'BeaMoy',  'EtiGou', 'GauDes', 'JenDow', 'LudArs'] # 'ClaZia', 'GabFor', 'EriHou',
list_plot = ['Subject 1', 'Subject 2', 'Subject 3', 'Subject 4', 'Subject 5', 'Subject 6', 'Subject 7']
higher_foot_list = ['R', 'R', 'R', 'L', 'R', 'R', 'R'] #  'L', 'R', 'L',
condition = ['squat_controle', 'squat_3cm', 'squat_4cm', 'squat_5cm']
model = biorbd.Model("/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/squat/Data_test/EriHou/EriHou.bioMod")
q_name = get_q_name(model)

q_hip = 11
q_knee = 12
q_ankle = 14
alpha = 0.05

anova_one_way(q_knee, alpha, repet=True)
anova_2_way(q_knee, alpha, repet=True)