import matplotlib.pyplot as plt
import numpy as np
from EMG import emg
import biorbd
import bioviz
from FORCE_PLATFORM import force_platform
from KINEMATIC import kinematic


name = 'AmeCeg'
model_path = "/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/squat/Data_test/" + name + "/" + name + ".bioMod"
model = biorbd.Model(model_path)
# b = bioviz.Viz(loaded_model=model)

# kinematic
kin_test = kinematic(name)
kin_test.plot_squat_mean()
kin_test.plot_squat_repetition()

# emg
emg_test = emg(name)
emg_test.plot_squat_mean_symetry()
emg_test.plot_assymetry_comparison_5cm()
emg_test.plot_squat_repetition()
emg_test.plot_squat_mean('squat_controle.c3d')
emg_test.plot_squat_comparison()
emg_test.plot_squat_comparison_5cm()
plt.show()


# force plateforme
# contact_test = force_platform(name)
# contact_test.plot_force_repetition()
# contact_test.plot_force_mean()
# contact_test.plot_force_comparison()
plt.show()
