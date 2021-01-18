from ezc3d import c3d
import numpy as np
from scipy.signal import butter, filtfilt
from matplotlib import pyplot as plt


# LOAD C3D FILE
Fs = 2000 # sample rate emg

[b, a] = butter(2, [10 / (0.5*Fs), 400 / (0.5*Fs)], btype='bandpass') # bandpass filter
[f, e] = butter(2, [9 / (0.5*Fs)], btype='lowpass') # lowpass filter

# --- JAMBE ---
nmus_jambe = 4

file_jambe = '/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/marche_saine/GauthierData/Walking/Walking_1.c3d'
measurements = c3d(file_jambe)
analogs = measurements["data"]["analogs"]
labels_analogs = measurements["parameters"]["ANALOG"]["LABELS"]["value"]
labels_muscle = ["R_TA.EMG1", "R_GM.EMG2", "R_Sol.EMG3", "R_PL.EMG4", "L_TA.EMG5", "L_GM.IM EMG6", "L_Sol.IM EMG7", "L_PL.IM EMG8"]
EMG_jambe = np.zeros((2, nmus_jambe, 2681))
EMG_jambe_filt = np.zeros((2, nmus_jambe, 2681))
Fz1 = analogs[0, labels_analogs.index("Force.Fz1"), :].squeeze()
Fz2 = analogs[0, labels_analogs.index("Force.Fz2"), :].squeeze()

figure, axes = plt.subplots(2, 4)
axes = axes.flatten()
for m in range(nmus_jambe):
    # Rigth leg
    EMG_jambe[0, m, :] = analogs[0, labels_analogs.index(labels_muscle[m]), 1830:4511].squeeze()
    EMG_jambe_filt[0, m, :] = EMG_jambe[0, m, :] - np.mean(EMG_jambe[0, m, :])  # remove baseline
    EMG_jambe_filt[0, m, :] = filtfilt(b, a, EMG_jambe_filt[0, m, :]) # bandpass filter
    EMG_jambe_filt[0, m, :] = np.abs(EMG_jambe_filt[0, m, :])  # absolute value
    EMG_jambe_filt[0, m, :] = filtfilt(f, e, EMG_jambe_filt[0, m, :]) # lowpass filter
    EMG_jambe_filt[0, m, :] = EMG_jambe_filt[0, m, :]/np.max(EMG_jambe[0, m, :]) # normalization
    axes[m].plot(EMG_jambe_filt[0, m, :])
    axes[m].set_title(labels_muscle[m])
    axes[m].set_ylim([0.0, 1.01])

    # Left leg
    EMG_jambe[1, m, :] = analogs[0, labels_analogs.index(labels_muscle[m + 4]), 1830:4511].squeeze()
    EMG_jambe_filt[1, m, :] = EMG_jambe[1, m, :] - np.mean(EMG_jambe[1, m, :])  # remove baseline
    EMG_jambe_filt[1, m, :] = filtfilt(b, a, EMG_jambe_filt[1, m, :]) # bandpass filter
    EMG_jambe_filt[1, m, :] = np.abs(EMG_jambe_filt[1, m, :])  # absolute value
    EMG_jambe_filt[1, m, :] = filtfilt(f, e, EMG_jambe_filt[1, m, :]) # lowpass filter
    EMG_jambe_filt[1, m, :] = EMG_jambe_filt[1, m, :] / np.max(EMG_jambe[1, m, :])  # normalization
    axes[m + nmus_jambe].plot(EMG_jambe_filt[1, m, :])
    axes[m + nmus_jambe].set_title(labels_muscle[m + nmus_jambe])
    axes[m + nmus_jambe].set_ylim([0.0, 1.01])


# --- CUISSE ---
nmus_cuisse = 3 #RF, VM, ST

file_cuisse = '/home/leasanchez/programmation/Simu_Marche_Casadi/Marche_BiorbdOptim/marche_saine/GauthierData/Walking_medicus/CeKo_SansOrthese.c3d'
measurements = c3d(file_cuisse)
analogs = measurements["data"]["analogs"]
labels_analogs = measurements["parameters"]["ANALOG"]["LABELS"]["value"]
labels_muscle = ["R_RF.EMG1", "R_VM.EMG2", "R_ST.EMG3", "L_RF.EMG4", "L_VM.EMG5", "L_ST.EMG6"]
Fz1 = analogs[0, labels_analogs.index("Force.Fz1"), :].squeeze()
Fz2 = analogs[0, labels_analogs.index("Force.Fz2"), :].squeeze()


EMG_cuisse = np.zeros((2, nmus_cuisse, 2087))
EMG_cuisse_filt = np.zeros((2, nmus_cuisse, 2087))

figure, axes = plt.subplots(2, 3)
axes = axes.flatten()
for m in range(nmus_cuisse):
    EMG_cuisse[0, m, :] = analogs[0, labels_analogs.index(labels_muscle[m]), 1770:3857].squeeze()
    EMG_cuisse_filt[0, m, :] = EMG_cuisse[0, m, :] - np.mean(EMG_cuisse[0, m, :])  # remove baseline
    EMG_cuisse_filt[0, m, :] = filtfilt(b, a, EMG_cuisse_filt[0, m, :]) # bandpass filter
    EMG_cuisse_filt[0, m, :] = np.abs(EMG_cuisse_filt[0, m, :])  # absolute value
    EMG_cuisse_filt[0, m, :] = filtfilt(f, e, EMG_cuisse_filt[0, m, :]) # lowpass filter
    EMG_cuisse_filt[0, m, :] = EMG_cuisse_filt[0, m, :]/np.max(EMG_cuisse[0, m, :])
    axes[m].plot(EMG_cuisse_filt[0, m, :])
    axes[m].set_title(labels_muscle[m])
    axes[m].set_ylim([0.0, 1.01])

    EMG_cuisse[1, m, :] = analogs[0, labels_analogs.index(labels_muscle[m + nmus_cuisse]), 1770:3857].squeeze()
    EMG_cuisse_filt[1, m, :] = EMG_cuisse[1, m, :] - np.mean(EMG_cuisse[1, m, :])  # remove baseline
    EMG_cuisse_filt[1, m, :] = filtfilt(b, a, EMG_cuisse_filt[1, m, :]) # bandpass filter
    EMG_cuisse_filt[1, m, :] = np.abs(EMG_cuisse_filt[1, m, :])  # absolute value
    EMG_cuisse_filt[1, m, :] = filtfilt(f, e, EMG_cuisse_filt[1, m, :]) # lowpass filter
    EMG_cuisse_filt[1, m, :] = EMG_cuisse_filt[1, m, :]/np.max(EMG_cuisse[1, m, :])
    axes[m + nmus_cuisse].plot(EMG_cuisse_filt[1, m, :])
    axes[m + nmus_cuisse].set_title(labels_muscle[m + nmus_cuisse])
    axes[m + nmus_cuisse].set_ylim([0.0, 1.01])

plt.show()