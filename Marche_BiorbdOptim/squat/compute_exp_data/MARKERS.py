import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import os
from ezc3d import c3d

def get_exp_files(path):
    return os.listdir(path + '/Squats/')

def find_initial_height(marker_position):
    return int(np.mean(marker_position[:101]))

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

        init_L_ASIS = int(np.mean(markers_position[-1][2, 0, :101]))
        init_R_ASIS = int(np.mean(markers_position[-1][2, 5, :101]))
        init_CENTER = int(np.mean(np.mean(markers_position[-1][2, :6, :101], axis=0)))

        index_L_ASIS = np.where(markers_position[-1][2, 0, :] > init_L_ASIS)
        index_R_ASIS = np.where(markers_position[-1][2, 5, :] > init_R_ASIS)
        index_CENTER = np.where(np.mean(markers_position[-1][2, :6, :], axis=0) > init_CENTER)

        plt.figure()
        plt.plot(markers_position[-1][2, 0, :])
        plt.plot(markers_position[-1][2, 5, :])
        plt.plot(np.mean(markers_position[-1][2, :6, :], axis=0))
        plt.plot([0, 1600], [init_L_ASIS - 5, init_L_ASIS - 5], 'k--')
        plt.plot([0, 1600], [init_R_ASIS - 5, init_R_ASIS - 5], 'k--')
        plt.plot([0, 1600], [init_CENTER - 5, init_CENTER - 5], 'k--')
        plt.legend(['L_ASIS', 'R_ASIS', 'CENTER'])

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        for i in range(6):
            ax.scatter3D(markers_position[-1][0, i, :], markers_position[-1][1, i, :], markers_position[-1][2, i, :])
        ax.scatter3D(np.mean(markers_position[-1][0, :6, :], axis=0),
                     np.mean(markers_position[-1][1, :6, :], axis=0),
                     np.mean(markers_position[-1][2, :6, :], axis=0),)

        return events