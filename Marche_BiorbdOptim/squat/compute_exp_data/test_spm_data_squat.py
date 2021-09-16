import matplotlib.pyplot as plt
import numpy as np
import spm1d
import biorbd_casadi as biorbd
import bioviz

from EMG import emg
from FORCE_PLATFORM import force_platform
from KINEMATIC import kinematic


list_name = ['AmeCeg', 'AnaLau', 'BeaMoy', 'ClaZia', 'EriHou', 'EtiGou', 'GabFor', 'GauDes', 'JenDow', 'LudArs']
data_control = np.zeros((len(list_name), 20, 2000))
data_5cm = np.zeros((len(list_name), 20, 2000))

for (i,name) in enumerate(list_name):
    squat_control_emg = np.load("/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/squat/Data_test/" + name + "/EMG/squat_controle_mean.npy")
    squat_5cm_emg = np.load("/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/squat/Data_test/" + name + "/EMG/squat_5cm_mean.npy")

    data_control[i, :, :] = squat_control_emg[:, 0::int(squat_control_emg.shape[1]/2000)]
    data_5cm[i, :, :] = squat_5cm_emg[:, 0::int(squat_5cm_emg.shape[1] / 2000)]

for e in range(20):
    alpha = 0.05
    t = spm1d.stats.ttest_paired(data_control[:, e, :], data_5cm[:, e, :])
    ti = t.inference(alpha, two_tailed=True, interp=True)

    plt.figure(figsize=(8, 3.5))
    ax = plt.axes((0.1, 0.15, 0.35, 0.8))
    spm1d.plot.plot_mean_sd(data_control[:, e, :])
    spm1d.plot.plot_mean_sd(data_5cm[:, e, :], linecolor='r', facecolor='r')
    ax.axhline(y=0, color='k', linestyle=':')
    ax.set_xlabel('Time')
    ax.set_ylabel('Muscle activation (%)')

    ax = plt.axes((0.55,0.15,0.35,0.8))
    ti.plot()
    ti.plot_threshold_label(fontsize=8)
    ti.plot_p_values(size=10, offsets=[(0,0.3)])
    ax.set_xlabel('Time')
    plt.show()


