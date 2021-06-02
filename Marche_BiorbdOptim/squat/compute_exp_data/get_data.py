import matplotlib.pyplot as plt
import numpy as np
from EMG import emg
from FORCE_PLATFORM import force_platform


path = '../Data_test/AmeCeg'

# emg
emg_test = emg(path)

emg_test.plot_squat_repetition(title='squat_controle.c3d')
emg_test.plot_squat_mean(title='squat_controle.c3d')
emg_test.plot_squat_comparison()

# contact
force_platform_test = force_platform(path)
