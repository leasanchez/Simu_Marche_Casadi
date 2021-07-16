import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import os
from ezc3d import c3d

def get_exp_files(path):
    return os.listdir(path + '/Squats/')

def find_initial_height(marker_anato):
    return round(np.mean(marker_anato))

def find_indices(markers_position, marker_anato):
    indices = []
    index_anato = np.where(markers_position > (find_initial_height(marker_anato) - 5))[0]
    discontinuities_idx = np.where(np.gradient(index_anato) > 1)[0]
    for idx in index_anato[discontinuities_idx]:
        a = np.where((markers_position[idx:idx + 51] - markers_position[idx]) < 0)[0]
        b = np.where((markers_position[idx - 50:idx + 1] - markers_position[idx]) < 0)[0]
        if (a.size > 45) or (b.size > 45):
            indices.append(idx)
    return indices

class markers:
    def __init__(self, path):
        self.path = path
        self.list_exp_files = ['squat_controle.c3d', 'squat_3cm.c3d', 'squat_4cm.c3d', 'squat_5cm.c3d', 'squat_controle_post.c3d']
        self.loaded_c3d = self.load_c3d()
        self.labels_markers = self.loaded_c3d[-1]["parameters"]["POINT"]["LABELS"]["value"][:52]
        self.markers_position = self.get_markers_position()
        self.events = self.get_events()
        self.mid_events = self.get_mid_events()
        self.time = self.get_time()


    def load_c3d(self):
        loaded_c3d = []
        for file in self.list_exp_files:
            loaded_c3d.append(c3d(self.path + '/Squats/' + file))
        return loaded_c3d

    def get_markers_position(self):
        markers_position = []
        for c in self.loaded_c3d:
            markers_position.append(c["data"]["points"])
        return markers_position


    def get_events(self):
        events=[]
        markers_position = self.get_markers_position()
        position_anato = c3d(self.path + '/SCoRE/anato.c3d')
        marker_anato = position_anato["data"]["points"]

        for mark in markers_position:
            indices = find_indices(mark[2, 0, :], marker_anato[2, 0, :])

            init_anato = find_initial_height(marker_anato[2, 0, :])
            events.append(indices)

            # plt.figure()
            # plt.plot(mark[2, 0, :])
            # plt.plot(marker_anato[2, 0, :], 'r')
            # plt.plot([0, 1900], [init_anato, init_anato], 'k--')
            # # plt.plot([0, 1900], [init_anato - 5, init_anato - 5], 'g--')
            # for idx in indices:
            #     plt.plot([idx, idx], [700, 1050], 'g--')
            # plt.show()
        return events

    def get_time(self):
        time = []
        for (n,e) in enumerate(self.events):
            t = 0
            t_fall = 0
            t_climb = 0
            for i in range(int(len(e)/2)):
                t += (e[2*i + 1] - e[2*i])/100
                t_fall += (self.mid_events[n][i] - e[2*i])/100
                t_climb += (e[2 * i + 1] - self.mid_events[n][i]) / 100
            time.append([t_fall/6, t_climb/6, t/6])
        return time

    def get_mid_events(self):
        mid_events=[]
        for event in self.events:
            md = []
            for i in range(int(len(event)/2)):
                md.append(int(np.mean([event[2*i], event[2*i + 1]])))
            mid_events.append(np.array(md))

        markers_position = self.get_markers_position()
        position_anato = c3d(self.path + '/SCoRE/anato.c3d')
        marker_anato = position_anato["data"]["points"]

        for (i, mark) in enumerate(markers_position):
            indices = find_indices(mark[2, 0, :], marker_anato[2, 0, :])

            init_anato = find_initial_height(marker_anato[2, 0, :])

            # plt.figure()
            # plt.plot(mark[2, 0, :])
            # plt.plot(marker_anato[2, 0, :], 'r')
            # plt.plot([0, 1900], [init_anato, init_anato], 'k--')
            # for idx in indices:
            #     plt.plot([idx, idx], [700, 1050], 'g--')
            # for m in range(mid_events[i].shape[0]):
            #     plt.plot([mid_events[i][m], mid_events[i][m]], [612, 1050], 'm--')
            # plt.show()
        return mid_events