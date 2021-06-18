import matplotlib.pyplot as plt
import numpy as np
from EMG import emg
from FORCE_PLATFORM import force_platform


name = 'AmeCeg'

# emg
# emg_test = emg(name)
#
# emg_test.plot_squat_repetition()
# emg_test.plot_squat_mean()
# emg_test.plot_squat_comparison()
# emg_test.plot_squat_comparison_5cm()
# plt.show()


# force plateforme
contact_test = force_platform(name)
contact_test.plot_force_repetition()
contact_test.plot_force_mean()
contact_test.plot_force_comparison()
plt.show()
