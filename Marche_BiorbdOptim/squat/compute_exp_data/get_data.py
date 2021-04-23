import matplotlib.pyplot as plt
import numpy as np
import os
from ezc3d import c3d
from pyomeca import Analogs
from EMG import emg

def emg_processed(file_path, name):
    emg = Analogs.from_c3d(file_path, usecols=name)
    emg_process = (
        emg.meca.band_pass(order=2, cutoff=[10, 425])
            .meca.center()
            .meca.abs()
            .meca.low_pass(order=4, cutoff=5, freq=emg.rate)
    )
    return emg_process

def emg_processed_exp(file_path, name, mvc):
    emg = Analogs.from_c3d(file_path, usecols=name)
    emg_process = []
    for (i, e) in enumerate(emg):
        e_process = (
            e.meca.band_pass(order=2, cutoff=[10, 425])
                .meca.center()
                .meca.abs()
                .meca.low_pass(order=4, cutoff=5, freq=e.rate)
                .meca.normalize(mvc[i])
        )
        emg_process.append(e_process)
    return emg_process

def compute_mvc_value(data):
    a = np.sort(data)[-1000:]
    mean = np.mean(a)
    median = np.median(a)
    return mean, median

def get_mvc_value(emg_mvc):
    MVC = {}
    MVC['muscle_name'] = []
    MVC['mvc_value_mean'] = []
    MVC['mvc_value_median'] = []
    mvc_value_mean = np.zeros((20, 20))
    mvc_value_median = np.zeros((20, 20))
    for (i, emg) in enumerate(emg_mvc):
        MVC['muscle_name'].append(emg['muscle'])
        mean = []
        median = []
        for m in range(emg['emg'][0].data.shape[0]):
            m1, md1 = compute_mvc_value(emg['emg'][0].data[m, :])
            m2, md2 = compute_mvc_value(emg['emg'][0].data[m, :])
            mvc_value_mean[i, m] = np.mean([m1, m2])
            mvc_value_median[i, m] = np.mean([md1, md2])
            mean.append(np.mean([m1, m2]))
            median.append(np.mean([md1, md2]))
        MVC['mvc_value_mean'].append(mean)
        MVC['mvc_value_median'].append(median)
    MVC['mvc_value_mean_tot'] = mvc_value_mean
    MVC['mvc_value_median_tot'] = mvc_value_median
    return MVC


path = '../Data_test/Eve/20042021/'
c=c3d(path + 'squat_controle_3.c3d')
index_controle_0 = (7000,11200,15600,19200,24600,29200)
index_controle_post = (7500,11200,14800,18300,22000,27400)
index_3cm = (6800,10500,14400,18900,23000,27400)
index_4cm = (6600,10400,13800,17700,21500,25900)
index_5cm = (6300,10400,14600,18900,23800)


emg_test = emg(path)
emg_test.plot_squat_mean('squat_controle.c3d', index=index_controle_0)
emg_test.plot_squat_repetition('squat_controle.c3d', index=index_controle_0)

emg_test.plot_squat_mean('squat_controle_post.c3d', index=index_controle_post)
emg_test.plot_squat_repetition('squat_controle_post.c3d', index=index_controle_post)

emg_test.plot_squat_mean('squat_3cm.c3d', index=index_3cm)
emg_test.plot_squat_repetition('squat_3cm.c3d', index=index_3cm)

emg_test.plot_squat_mean('squat_4cm.c3d', index=index_4cm)
emg_test.plot_squat_repetition('squat_4cm.c3d', index=index_4cm)

emg_test.plot_squat_mean('squat_5cm.c3d', index=index_5cm)
emg_test.plot_squat_repetition('squat_5cm.c3d', index=index_5cm)

files_exp = ['squat_controle.c3d', 'squat_3cm.c3d', 'squat_4cm.c3d', 'squat_5cm.c3d']
index_exp = [index_controle_0, index_3cm, index_4cm, index_5cm]
emg_test.plot_squat_comparison(files_exp, index_exp)

plt.show()