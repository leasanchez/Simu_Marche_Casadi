import numpy as np
import matplotlib.pyplot as plt
import spm1d
from KINEMATIC import kinematic
from EMG import emg



def anova_ratio(muscle_idx):
    list_condition = []  # 0 : control, 1 : 3cm, 2 : 4cm, 3 : 5cm
    y = np.ndarray((40, 2000))  # 40 = 10 subjects x 4 conditions
    subject = []

    for i in range(len(list_name)):
        subject_dir = "/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/squat/Data_test/" + \
                      list_name[i]
        for (j, c) in enumerate(condition):
            subject.append(i)
            list_condition.append(j)  # condition -> 0 : controle, 1 : 3 cm, 2 : 4 cm, 3 : 5 cm
            e = np.load(subject_dir + "/EMG/" + c + "_mean_ratio.npy")
            if e.shape[1] == 4000:
                y[4 * i + j] = e[muscle_idx, 0::2]
            else:
                y[4 * i + j] = e[muscle_idx]

    # ANOVA:
    alpha = 0.05
    list_condition = np.array(list_condition)
    subject = np.array(subject)
    F = spm1d.stats.anova1(y, list_condition, equal_var=True)
    Fi = F.inference(alpha)
    Frm = spm1d.stats.anova1rm(y, list_condition, subject, equal_var=True)
    Firm = Frm.inference(alpha)

    # (2) Plot:
    x = np.linspace(0, 100, 2000)
    plt.close('all')
    plt.figure(muscle_label[muscle_idx] + ' Ratio', figsize=(8, 3.5))
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
    Firm.plot(ax=ax1, color='r', facecolor=(0.8, 0.3, 0.3), label='Within-subjects analysis')
    Fi.plot(ax=ax1, label='Between-subjects analysis')
    ax1.legend(fontsize=8)
    plt.show()



# Load data:
list_name = ['AmeCeg', 'AnaLau', 'BeaMoy', 'ClaZia', 'EriHou', 'EtiGou', 'GabFor', 'GauDes', 'JenDow', 'LudArs']
list_plot = ['Subject 1', 'Subject 2', 'Subject 3', 'Subject 4', 'Subject 5', 'Subject 6', 'Subject 7', 'Subject 8', 'Subject 9', 'Subject 10']
condition = ['squat_controle', 'squat_3cm', 'squat_4cm', 'squat_5cm']
higher_foot_list = ['R', 'R', 'R', 'L', 'L', 'L', 'R', 'R', 'R', 'R']
muscle_label = ['Gastrocnemien medial', 'Soleaire', 'Long fibulaire', 'Tibial anterieur', 'Vaste medial',
                'Droit anterieur', 'Semitendineux', 'Moyen fessier', 'Grand fessier', 'Long adducteur']

# mean ratio
for m in range(len(muscle_label)):
    anova_ratio(m)

# mean muscle value
muscle_idx = 0
list_condition = []  # 0 : control, 1 : 3cm, 2 : 4cm, 3 : 5cm
leg = [] # 0 : control, # 1 : elevated
subject = []
y = np.ndarray((80, 2000))  # 40 = 10 subjects x 4 conditions x 2 jambes

for i in range(len(list_name)):
    subject_dir = "/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/squat/Data_test/" \
                  + list_name[i]
    idx = [muscle_idx, muscle_idx + 1] if higher_foot_list[i] == 'R' else [muscle_idx + 1, muscle_idx]
    for (j, c) in enumerate(condition):
        list_condition.append(j)  # condition -> 0 : controle, 1 : 3 cm, 2 : 4 cm, 3 : 5 cm
        list_condition.append(j)
        leg.append(0)
        leg.append(1)
        e = np.load(subject_dir + "/EMG/" + c + "_mean.npy")
        if e.shape[1] == 4000:
            y[8 * i + 2*j: 8 * i + 2*j + 2] = e[idx, 0::2]
        else:
            y[8 * i + 2*j: 8 * i + 2*j + 2] = e[idx]

# ANOVA:
alpha = 0.5
list_condition = np.array(list_condition)
leg = np.array(leg)
F = spm1d.stats.anova2(y, list_condition, leg, equal_var=True)
Fi = F.inference(alpha)

# (2) Plot:
x = np.linspace(0, 100, 2000)
plt.close('all')
plt.figure(muscle_label[muscle_idx] + ' Ratio', figsize=(8, 3.5))
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
Fi.plot(ax=ax1)
plt.show()


