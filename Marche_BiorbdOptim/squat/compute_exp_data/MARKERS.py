import matplotlib.pyplot as plt
import numpy as np
import os
from ezc3d import c3d
from scipy import signal
from UTILS import utils

def plot_events(markers_position, marker_anato, events):
    plt.figure()
    plt.plot(markers_position)
    plt.plot([0, 1900], [find_initial_height(marker_anato) - 5, find_initial_height(marker_anato) - 5], 'k--')
    for e in events:
        plt.plot([e, e], [700, 1050], 'r')
    for m in min:
        plt.plot([m, m], [700, 1050], 'g')
    plt.show()

def get_exp_files(path):
    return os.listdir(path + '/Squats/')

def find_initial_height(marker_anato):
    m = marker_anato[~np.isnan(marker_anato)]
    return round(np.mean(m))

def find_indices(markers_position, marker_anato):
    indices = []
    # index_anato = np.where(markers_position > (find_initial_height(markers_position[:50]) - 10))[0]
    index_anato = np.where(markers_position > (find_initial_height(marker_anato) - 5))[0]
    discontinuities_idx = np.where(np.gradient(index_anato) > 1)[0]
    for idx in index_anato[discontinuities_idx]:
        a = np.where((markers_position[idx:idx + 51] - markers_position[idx]) < 0)[0]
        b = np.where((markers_position[idx - 50:idx + 1] - markers_position[idx]) < 0)[0]
        if (a.size > 40) or (b.size > 40):
            indices.append(idx)
    return indices

def find_min(markers_position, marker_anato, events):
    n_repet = int(len(events)/2)
    min = [np.argmin(markers_position[events[2*i]:events[2*i + 1]]) + events[2*i] for i in range(n_repet)]
    # plot_events(markers_position, marker_anato, events)
    return min

def apply_filter(data, b, a):
    '''
    interpolate nan value and use butterworth lowpass filter
    input : data : raw markers coordinates
            b, a : filter parameters
    output : data_filt : filtered markers coordinates
    '''

    data_filt = np.ndarray(data.shape)
    data_nan = np.ndarray(data.shape)
    for m in range(data.shape[1]):
        nan_value = [utils.fill_nan(data[i, m, :]) for i in range(3) if utils.has_nan(data[i, m, :])] # extrapolate data if nan detected
        data_nan[:, m, :] = np.array(nan_value) if nan_value else data[:, m, :]  # fill nan or use raw data
        data_filt[:, m, :] = np.array([signal.filtfilt(b, a, data_nan[i, m, :]) for i in range(3)]) # apply filter
    return data_filt


class markers:
    def __init__(self, path):
        self.path = path
        self.list_exp_files = ['squat_controle.c3d', 'squat_3cm.c3d', 'squat_4cm.c3d', 'squat_5cm.c3d', 'squat_controle_post.c3d']
        self.loaded_c3d = self.load_c3d()
        self.n_marker = 52
        self.labels_markers = self.loaded_c3d[-1]["parameters"]["POINT"]["LABELS"]["value"][:52]
        self.markers_position = self.get_markers_position()
        self.filtered_markers_position = self.filter_markers_position()
        self.events, self.mid_events = self.get_events()
        self.time = self.get_time()


    def load_c3d(self):
        loaded_c3d = [c3d(self.path + '/Squats/' + file) for file in self.list_exp_files]
        return loaded_c3d

    def get_markers_position(self):
        markers_position = [c["data"]["points"][:3, :, :] for c in self.loaded_c3d]
        return markers_position

    def filter_markers_position(self):
        b, a = utils.define_butterworth_filter(fs=100, fc=6)
        filtered_markers_position = [apply_filter(mark_pos, b, a) for mark_pos in self.markers_position]
        return filtered_markers_position

    def get_events(self):
        position_anato = c3d(self.path + '/SCoRE/anato.c3d')
        marker_anato = position_anato["data"]["points"]

        events = [find_indices(mark[2, 1, :], marker_anato[2, 1, :]) for mark in self.filtered_markers_position]
        mid_events = [find_min(mark[2, 1, :], marker_anato[2, 1, :], find_indices(mark[2, 1, :], marker_anato[2, 1, :]))
                      for mark in self.filtered_markers_position]
        return events, mid_events

    def get_time(self):
        time = []
        for (n,e) in enumerate(self.events):
            t = np.mean((np.array(e[1::2]) - np.array(e[0::2]))/100)
            tfall = np.mean((self.mid_events[n] - np.array(e[0::2]))/100)
            tclimb = np.mean((np.array(e[1::2]) - self.mid_events[n])/100)
            time.append([t, tfall, tclimb])
        return time