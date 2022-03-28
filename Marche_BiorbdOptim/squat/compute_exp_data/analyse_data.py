import bioviz
from casadi import MX
import matplotlib.pyplot as plt
import numpy as np
import spm1d
import scipy.stats as stats
import biorbd_casadi as biorbd
from EMG import emg
from KINEMATIC import kinematic
from MARKERS import markers
from PLOT import plot_function
from UTILS import utils
from FORCE_PLATFORM import force_platform


def get_data_repet_for_anova(kin_test, q_idx, ratio=False):
    condition = []
    leg = []
    subject = []
    y = []
    for (s, kin) in enumerate(kin_test): # subject
        idx = [q_idx, q_idx + 6] if kin.higher_foot == 'L' else [q_idx + 6, q_idx]  # put control leg first
        repet = [utils.interpolate_repetition(q_kal, kin.events[i], 100) for (i, q_kal) in enumerate(kin.q_kalman)]
        for (i, r) in enumerate(repet): # condition
            n_repet = len(r)
            if ratio:
                condition = np.hstack([condition, [i] * n_repet])
                subject = np.hstack([subject, [s] * n_repet])
                y.append(np.vstack([r[i][idx[1], :]/r[i][idx[0], :] for i in range(n_repet)]))
            else:
                leg = np.hstack([leg, [0, 1] * n_repet])
                condition = np.hstack([condition, [i, i] * n_repet])
                subject = np.hstack([subject, [s, s] * n_repet])
                y.append(np.vstack([r[i][idx, :] for i in range(n_repet)]))
    return subject, condition, leg, np.vstack(y)

def get_data_for_anova(kin_test, q_idx, ratio=False):
    n_condition = 4
    condition = []
    leg = []
    subjects = []
    y = []
    for (s,kin) in enumerate(kin_test): # sujet
        idx = [q_idx, q_idx + 6] if kin.higher_foot == 'L' else [q_idx + 6, q_idx]  # put control leg first
        if ratio:
            # ratio between elevated and control leg
            subjects = np.hstack([subjects, [s] * n_condition])
            condition = np.hstack([condition, [0, 1, 2, 3]])  # 0 : control, 1 : 3cm, 2 : 4cm, 3 : 5cm
            y.append(np.vstack([(kin.q_mean[i][idx[1], :]/kin.q_mean[i][idx[0], :]) for i in range(n_condition)]))
        else:
            # use data from the 2 legs
            subjects = np.hstack([subjects, [s, s] * n_condition])
            condition = np.hstack([condition, [0, 0, 1, 1, 2, 2, 3, 3]])
            leg = np.hstack([leg, [0, 1] * n_condition])
            y.append(np.vstack([np.vstack(kin.q_mean[i][idx, :]) for i in range(n_condition)]))
    return subjects, condition, leg, np.vstack(y)

def get_initial_max_final_angle_diff(kin_test, q_idx,condition_idx):
    diff_init = []
    diff_max = []
    diff_final = []
    for (i, kin) in enumerate(kin_test):
        leg_R = np.abs(kin.q_mean[condition_idx][q_idx, :])
        leg_L = np.abs(kin.q_mean[condition_idx][q_idx + 6, :])
        diff_init = np.hstack([diff_init, np.abs(leg_R[0] - leg_L[0]) * 180 / np.pi])
        diff_max = np.hstack([diff_max, np.abs(np.max(leg_R) - np.max(leg_L)) * 180 / np.pi])
        diff_final = np.hstack([diff_final, np.abs(leg_R[-1] - leg_L[-1]) * 180 / np.pi])
    return diff_init, diff_max, diff_final

pf = plot_function()

# list participant and info
list_name = ['AmeCeg', 'AnaLau', 'BeaMoy', 'EtiGou', 'JenDow', 'LudArs', 'GauDes', 'GabFor', 'EriHou', 'ClaZia']
higher_foot_list = ['R', 'R', 'R', 'L', 'R', 'R', 'R', 'R', 'L', 'L']
leg_length = [0.92, 0.9035, 0.8535, 0.934, 0.971, 0.925, 0.87, 0.939, 0.854, 0.8735]
list_condition = ['squat_controle', 'squat_3cm', 'squat_4cm', 'squat_5cm']

# init
kin_test = []
mark_test = []
emg_test = []
com_anato = []
fp_test = []

# get data
for (i, name) in enumerate(list_name):
    model = biorbd.Model("/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/squat/Data_test/"+ name + "/"+ name + ".bioMod")
    q_name = utils.get_q_name(model)
    path = "/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/squat/Data_test/" + name
    q_anato = utils.load_txt_file(path + "/kalman/anato_q_KalmanFilter.txt", model.nbQ())

    mark_test.append(markers(name))
    kin_test.append(kinematic(name, higher_foot_list[i]))
    # kin_test[i].plot_q_mean(idx_condition=0)
    # kin_test[i].plot_q_repet(idx_condition=0)
    emg_test.append(emg(name))
    # emg_test[i].plot_mean_activation(idx_condition=0)
    # emg_test[i].plot_repet_activation(idx_condition=0)
    fp_test.append(force_platform(name))
    # fp_test[i].plot_force_mean(idx_condition=0)
    # fp_test[i].plot_force_repet(idx_condition=0)
    compute_CoM = biorbd.to_casadi_func("CoM", model.CoM, MX.sym("q", model.nbQ(), 1))
    com_anato.append(compute_CoM(q_anato[:, 200]))

    # b = bioviz.Viz(model_path=path + "/" + name + ".bioMod", background_color=[1,1,1])
    # b.load_movement(q_anato) # position anatomique
    # b.load_movement(kin_test[i].q_kalman[0]) # essai complet
    # b.load_movement(kin_test[i].q_mean[0]) # moyenne des repetions
    # b.exec()

# pf.plot_initial_kinematic(kin_test, q_idx=12, condition_idx=0, title='knee flexion')
# pf.plot_initial_kinematic(kin_test, q_idx=11, condition_idx=0, title='hip flexion')
# pf.plot_initial_com_displacement(kin_test, condition_idx=0, title='com normalized by com position anatomic', norm=np.array(com_anato)[:, 2])
# plt.show()

q_idx = 14 # dof index
# anova 2 way
alpha = 0.05

subject_mean, condition_mean, leg_mean, y_mean = get_data_for_anova(kin_test, q_idx)
subject_repet, condition_repet, leg_repet, y_repet = get_data_repet_for_anova(kin_test, q_idx)

FFrm = spm1d.stats.anova2rm(y_repet, condition_repet, leg_repet, subject_repet)
FFirm = FFrm.inference(alpha)
pf.plot_anova(FFirm[0], y_repet, condition_repet, condition_name=list_condition, title='anova 2 way condition')
pf.plot_anova(FFirm[1], y_repet, leg_repet, condition_name=['control', 'elevated'], title='anova 2 way leg')

# anova 1 way with ratio
condition_ratio = np.hstack([0, 1, 2, 3] * len(list_name))
subjects_ratio = np.hstack([[i] * len(list_condition) for i in range(len(list_name))])
y_ratio = get_data_for_anova(kin_test, q_idx, ratio=True)
Frm = spm1d.stats.anova1rm(y_ratio, condition_ratio, subjects_ratio, equal_var=True)
Firm = Frm.inference(alpha)
pf.plot_anova(Firm, y_ratio, condition_ratio, list_condition, title='anova 1 way ratio')
plt.show()

# paired ttest on initial and final condition
diff_init_con, diff_max_con, diff_final_con = get_initial_max_final_angle_diff(kin_test, q_idx=q_idx, condition_idx=0)
diff_init_3cm, diff_max_3cm, diff_final_3cm = get_initial_max_final_angle_diff(kin_test, q_idx=q_idx, condition_idx=1)
diff_init_4cm, diff_max_4cm, diff_final_4cm = get_initial_max_final_angle_diff(kin_test, q_idx=q_idx, condition_idx=2)
diff_init_5cm, diff_max_5cm, diff_final_5cm = get_initial_max_final_angle_diff(kin_test, q_idx=q_idx, condition_idx=3)

t_init = stats.ttest_rel(diff_init_con, diff_init_5cm)
t_max = stats.ttest_rel(diff_max_con, diff_max_5cm)
t_final = stats.ttest_rel(diff_final_con, diff_final_5cm)
