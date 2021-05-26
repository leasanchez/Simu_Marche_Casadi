import matplotlib.pyplot as plt
import numpy as np
import os
from ezc3d import c3d
from EMG import emg
from FORCE_PLATFORM import force_platform
from MARKERS import markers


path = '../Data_test/AmeCeg'
# c=c3d(path + 'squat_controle_3.c3d')
# index_controle_0 = (7000,11200,15600,19200,24600,29200)
# index_controle_post = (7500,11200,14800,18300,22000,27400)
# index_3cm = (6800,10500,14400,18900,23000,27400)
# index_4cm = (6600,10400,13800,17700,21500,25900)
# index_5cm = (6300,10400,14600,18900,23800)

# contact = force_platform(path)
emg_test = emg(path)
markers_test = markers(path)

emg_test.plot_squat(emg_data=emg_test.emg_normalized_exp[-1], title=emg_test.list_exp_files[-1])


index_exp = [index_controle_0, index_3cm, index_4cm, index_5cm]
emg_test.plot_squat_comparison(files_exp, index_exp)

for (e, file) in enumerate(files_exp):
    emg_test.plot_squat_mean(file_path=file, index=index_exp[e])
    emg_test.plot_squat_repetition(file_path=file, index=index_exp[e])
plt.show()

contact = force_platform(path)
contact.plot_force_comparison(file_path=files_exp, index=index_exp)
contact.plot_cop(file_path=files_exp, index=index_exp)
for (e, file) in enumerate(files_exp):
    contact.plot_force_repetition(c3d_path=file, index=index_exp[e])
    contact.plot_force_mean(c3d_path=file, index=index_exp[e])
plt.show()
