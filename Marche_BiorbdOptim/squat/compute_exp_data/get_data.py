import matplotlib.pyplot as plt
import numpy as np
from EMG import emg
from FORCE_PLATFORM import force_platform


name = 'AmeCeg'

# emg
emg_test = emg(name)

emg_test.plot_squat_repetition()
emg_test.plot_squat_mean()
emg_test.plot_squat_comparison()
emg_test.plot_squat_comparison_5cm()
plt.show()


# contact
# force_platform_test = force_platform(name)
