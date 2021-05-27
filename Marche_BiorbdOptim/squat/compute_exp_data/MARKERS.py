import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import os
from ezc3d import c3d

def get_exp_files(path):
    return os.listdir(path + '/Squats/')

def find_initial_height(marker_position):
    return int(np.mean(marker_position[:101]))

def find_indices(markers_position):
    pelvis_center = np.mean(markers_position[2, :6, :], axis=0)
    index_CENTER = np.where(pelvis_center > (find_initial_height(pelvis_center) - 5))[0]
    discontinuities_idx = np.where(np.gradient(index_CENTER) > 1)[0]
    return index_CENTER[discontinuities_idx]

class markers:
    def __init__(self, path):
        self.path = path
        self.list_exp_files = get_exp_files(path)
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

        for mark in markers_position:
            indices = find_indices(mark)

            pelvis_center = np.mean(mark[2, :6, :], axis=0)
            init_CENTER = find_initial_height(pelvis_center)
            events.append(indices)

            # plt.figure()
            # plt.plot(pelvis_center)
            # plt.plot([0, 1900], [init_CENTER - 5, init_CENTER - 5], 'k--')
            # for idx in indices:
            #     plt.plot([idx, idx], [700, 1050], 'g--')
            # plt.show()

        return events