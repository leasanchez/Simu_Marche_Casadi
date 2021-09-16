import matplotlib.pyplot as plt
import numpy as np
import spm1d
import biorbd_casadi as biorbd
import bioviz

from KINEMATIC import kinematic

list_name = ['AmeCeg', 'AnaLau', 'BeaMoy', 'EriHou', 'EtiGou', 'GauDes', 'JenDow', 'LudArs'] # 'ClaZia', 'GabFor',
list_plot = ['Subject 1', 'Subject 2', 'Subject 3', 'Subject 4', 'Subject 5', 'Subject 6', 'Subject 7', 'Subject 8']
higher_foot_list = ['R', 'R', 'R', 'L', 'L', 'R', 'R', 'R'] #  'L', 'R',

fig, axes = plt.subplots(3, 1, sharex=True)
fig.suptitle('Knee max angle flexion')
axes = axes.flatten()


fig2, axes2 = plt.subplots(3, 1, sharex=True)
fig2.suptitle('Knee amplitude flexion')
axes2 = axes2.flatten()

for i in range(len(list_name)):
    kin_test = kinematic(list_name[i], higher_foot_list[i])

    axes[0].plot(np.linspace(0, 3, 4), kin_test.pic_flexion_knee[0][:-1], linestyle='dashed', marker='o')
    axes[1].plot(np.linspace(0, 3, 4), kin_test.pic_flexion_knee[1][:-1], linestyle='dashed', marker='o')
    axes[2].plot(np.linspace(0, 3, 4), np.array(kin_test.pic_flexion_knee[0][:-1]) - np.array(kin_test.pic_flexion_knee[1][:-1]), linestyle='dashed', marker='o')

    axes2[0].plot(np.linspace(0, 3, 4), kin_test.amp_flexion_knee[0][:-1], linestyle='dashed', marker='o')
    axes2[1].plot(np.linspace(0, 3, 4), kin_test.amp_flexion_knee[1][:-1], linestyle='dashed', marker='o')
    axes2[2].plot(np.linspace(0, 3, 4), np.array(kin_test.amp_flexion_knee[0][:-1]) - np.array(kin_test.amp_flexion_knee[1][:-1]), linestyle='dashed', marker='o')

axes[0].set_title('Elevated foot')
axes[1].set_title('Control foot')
axes[2].set_title('Difference')
axes[2].set_xticks([0, 1, 2, 3])
axes[2].set_xticklabels(['control', '3cm', '4cm', '5cm'])

axes2[0].set_title('Elevated foot')
axes2[1].set_title('Control foot')
axes2[2].set_title('Difference')
axes2[2].set_xticks([0, 1, 2, 3])
axes2[2].set_xticklabels(['control', '3cm', '4cm', '5cm'])

plt.legend(list_plot)
plt.show()