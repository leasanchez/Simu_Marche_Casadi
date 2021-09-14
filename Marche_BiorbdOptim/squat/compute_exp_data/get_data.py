import matplotlib.pyplot as plt
import numpy as np
import spm1d
import biorbd_casadi as biorbd
import bioviz
from ezc3d import c3d
import xlsxwriter

from EMG import emg, interpolate_squat_repetition
from FORCE_PLATFORM import force_platform
from KINEMATIC import kinematic

def print_markers_excel(name):
    c3d_path = "/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/squat/Data_test/" + name + "/SCoRE/anato.c3d"
    position_anato = c3d(c3d_path)
    markers_anato = position_anato["data"]["points"]
    label_markers = np.array(position_anato['parameters']['POINT']['LABELS']['value'])

    workbook = xlsxwriter.Workbook("/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/squat/Data_test/" + name + "/" +name + "_marker_anato.xlsx")
    worksheet = workbook.add_worksheet()
    worksheet.write(3, 0, "Frame")
    worksheet.write_column(6, 0, range(1, markers_anato.shape[2]+1))
    worksheet.write(3, 1, "Time")
    worksheet.write_column(6, 1, np.linspace(0, (markers_anato.shape[2]-1)*0.01, markers_anato.shape[2]))
    for i, label in enumerate(label_markers):
        worksheet.write(3, 2 +3*i, label)
        worksheet.write_row(4, 2 +3*i, ["X"+str(i+1), "Y"+str(i+1), "Z"+str(i+1)])
    for m in range(markers_anato.shape[1]):
        worksheet.write_column(6, 2 + 3 * m, markers_anato[0, m, :])
        worksheet.write_column(6, 3 + 3 * m, markers_anato[1, m, :])
        worksheet.write_column(6, 4 + 3 * m, markers_anato[2, m, :])
    workbook.close()

def save_mean_data(mean_data, save_path, list_exp_files):
    for (i, d) in enumerate(mean_data):
       np.save(save_path + list_exp_files[i] + "_mean.npy", d)


name = 'GabFor'
model_path = "/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/squat/Data_test/" + name + "/" + name + ".bioMod"
model = biorbd.Model(model_path)
higher_foot = 'R'

# kinematic
kin_test = kinematic(name, higher_foot)
# save_mean_data(kin_test.q_mean, "/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/squat/Data_test/" + name + "/kalman/", kin_test.list_exp_files)
kin_test.plot_squat_mean_leg('squat_controle')
kin_test.plot_squat_mean_leg('squat_5cm')
kin_test.plot_squat_repetition('squat_3cm')
plt.show()

# emg
emg_test = emg(name)
# save_mean_data(emg_test.mean, "/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/squat/Data_test/" + name + "/EMG/", emg_test.list_exp_files)
# for (i, emg) in enumerate(emg_test.mean):
#     a = interpolate_squat_repetition(emg_test.emg_normalized_exp[i], emg_test.events[i], emg_test.freq)
#     np.save("/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/squat/Data_test/" + name + "/EMG/" +emg_test.list_exp_files[i] + "_repet.npy", a)
emg_test.plot_squat_comparison_5cm()
emg_test.plot_squat_mean_symetry("squat_controle")
emg_test.plot_squat_mean('squat_controle')
emg_test.plot_squat_mean('squat_5cm')





plt.show()

emg_test.plot_sort_activation()
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
