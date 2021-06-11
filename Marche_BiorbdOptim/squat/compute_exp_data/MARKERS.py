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
    index_anato = np.where(markers_position[2, 0, :] > (find_initial_height(marker_anato) - 5))[0]
    discontinuities_idx = np.where(np.gradient(index_anato) > 1)[0]
    return index_anato[discontinuities_idx]

class markers:
    def __init__(self, path):
        self.path = path
        self.list_exp_files = ['squat_controle.c3d', 'squat_3cm.c3d', 'squat_4cm.c3d', 'squat_5cm.c3d', 'squat_controle_post.c3d']
        self.loaded_c3d = self.load_c3d()
        self.labels_markers = self.loaded_c3d[-1]["parameters"]["POINT"]["LABELS"]["value"][:52]
        self.markers_position = self.get_markers_position()
        self.events = self.get_events()


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
            indices = find_indices(mark, marker_anato[2, 0, :])

            init_anato = find_initial_height(marker_anato[2, 0, :]) - 5
            events.append(indices)

            plt.figure()
            plt.plot(mark[2, 0, :])
            plt.plot(marker_anato[2, 0, :], 'r')
            plt.plot([0, 1900], [init_anato, init_anato], 'k--')
            for idx in indices:
                plt.plot([idx, idx], [700, 1050], 'g--')
            plt.show()

        return events