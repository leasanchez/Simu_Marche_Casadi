import matplotlib.pyplot as plt
import numpy as np
from ezc3d import c3d

def get_cop(loaded_c3d):
    """
    get the trajectory of the center of pressure (cop)
    from force platform
    """
    cop = []
    platform = loaded_c3d["data"]["platform"]
    for p in platform:
        cop.append(p["center_of_pressure"] * 1e-3)
    return cop

def plot_cop(cop):
    plt.figure()
    plt.plot(cop[0][1, :], cop[0][0, :])
    plt.plot(cop[1][1, :], cop[1][0, :])

def plot_cop_x(cop):
    legend = ['front', 'side', 'head']
    plt.figure('plateforme 1 - left')
    for c in cop:
     plt.plot(c[0][0, :])
    plt.legend(legend)

    plt.figure('plateforme 2 - right')
    for c in cop:
     plt.plot(c[1][0, :])
    plt.legend(legend)

path_arm = "Data_test/position_bras/"
front = c3d(path_arm + "squat_control01.c3d", extract_forceplat_data=True)
cop_front = get_cop(front)
plot_cop(cop_front)

side = c3d(path_arm + "squat_control02.c3d", extract_forceplat_data=True)
cop_side = get_cop(side)
plot_cop(cop_side)

head = c3d(path_arm + "squat_control03.c3d", extract_forceplat_data=True)
cop_head = get_cop(head)
plot_cop(cop_head)

plot_cop_x((cop_front, cop_side, cop_head))
plt.show()


