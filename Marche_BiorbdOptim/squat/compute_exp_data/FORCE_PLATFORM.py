import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import interpolate
from ezc3d import c3d

class force_platform:
    def __init__(self, path):
        self.path = path

    def get_cop(self, loaded_c3d):
        """
        get the trajectory of the center of pressure (cop)
        from force platform
        """
        cop = []
        platform = loaded_c3d["data"]["platform"]
        for p in platform:
            cop.append(p["center_of_pressure"] * 1e-3)
        return cop

    def get_corners_position(self, loaded_c3d):
        """
        get platform corners position
        """
        corners = []
        platform = loaded_c3d["data"]["platform"]
        for p in platform:
            corners.append(p["corners"] * 1e-3)
        return corners

    def get_forces(self, loaded_c3d):
        """
        get the ground reaction forces
        from force platform
        """
        force = []
        platform = loaded_c3d["data"]["platform"]
        for p in platform:
            force.append(p["force"])
        return force

    def get_moments(self, loaded_c3d):
        """
        get the ground reaction moments
        from force platform
        """
        moment = []
        platform = loaded_c3d["data"]["platform"]
        for p in platform:
            moment.append(p["moment"] * 1e-3)
        return moment

    def get_moments_at_cop(self, loaded_c3d):
        """
        get the ground reaction moments at cop
        from force platform
        """
        Tz = []
        platform = loaded_c3d["data"]["platform"]
        for p in platform:
            Tz.append(p["Tz"] * 1e-3)
        return Tz

    def load_c3d(self, c3d_path):
        return c3d(self.path + c3d_path, extract_forceplat_data=True)

    def interpolate_data(self, data, index):
        data_interp = []
        for d in data:
            interp = np.zeros((len(index)-1, 3, 5000))
            for i in range(len(index)-1):
                x_start = np.arange(0, d[0, index[i]:index[i+1]].shape[0])
                x_interp = np.linspace(0, x_start[-1], 5000)
                for m in range(3):
                    f = interpolate.interp1d(x_start, d[m, index[i]:index[i+1]])
                    interp[i, m, :] = f(x_interp)
            data_interp.append(interp)
        return data_interp

    def mean_data(self, data):
        mean = []
        std = []
        for d in data:
            mean_d = np.zeros((3, 5000))
            std_d = np.zeros((3, 5000))
            for m in range(3):
                mean_d[m, :] = np.mean(d[:, m, :], axis=0)
                std_d[m, :] = np.std(d[:, m, :], axis=0)
            mean.append(mean_d)
            std.append(std_d)
        return mean, std

    def plot_cop(self, file_path, index):
        corners = self.get_corners_position(self.load_c3d(file_path[0]))
        color = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
        mean = []
        for (i, file) in enumerate(file_path):
            cop = self.interpolate_data(data=self.get_cop(self.load_c3d(file)), index=index[i])
            m, std = self.mean_data(cop)
            mean.append(m)

        plt.figure()
        plt.suptitle('cop trajectories comparison')
        for (i, c) in enumerate(mean):
            plt.scatter(-c[0][1, :], c[0][0, :], c=color[i], alpha=0.5, label=file_path[i])
            plt.scatter(-c[1][1, :], c[1][0, :], c=color[i], alpha=0.5, label=file_path[i])
        plt.scatter(-corners[0][1, :], corners[0][0, :], c='k', marker='+') # platform 1 --> left
        plt.scatter(-corners[1][1, :], corners[1][0, :], c='y', marker='+')  # platform 2 --> right
        plt.xlabel('y (m)')
        plt.ylabel('x (m)')
        plt.legend()

    def plot_force_repetition(self, c3d_path, index):
        force = self.interpolate_data(data=self.get_forces(self.load_c3d(c3d_path)), index=index)
        label = ['x', 'y', 'z']

        fig, axes = plt.subplots(2, 3)
        axes = axes.flatten()
        fig.suptitle(f"ground reaction forces : {c3d_path}")
        for i in range(3):
            axes[i].set_title(f"platform 1 : {label[i]}")
            axes[i].plot(np.linspace(0, 100, 5000), force[0][:, i, :].T)
            axes[i].set_xlim([0, 100])
        for i in range(3):
            axes[i + 3].set_title(f"platform 2 : {label[i]}")
            axes[i + 3].plot(np.linspace(0, 100, 5000), force[1][:, i, :].T)
            axes[i + 3].set_xlim([0, 100])

    def plot_force_mean(self, c3d_path, index):
        force = self.interpolate_data(data=self.get_forces(self.load_c3d(c3d_path)), index=index)
        mean_force, std_force = self.mean_data(force)
        label = ['x', 'y', 'z']

        fig, axes = plt.subplots(1, 3)
        axes = axes.flatten()
        fig.suptitle(f"ground reaction forces : {c3d_path}")
        for i in range(3):
            axes[i].set_title(f"{label[i]}")
            axes[i].plot(np.linspace(0, 100, 5000), mean_force[0][i, :], 'b') # left
            axes[i].plot(np.linspace(0, 100, 5000), mean_force[1][i, :], 'r') # right

            axes[i].fill_between(np.linspace(0, 100, 5000),
                                 mean_force[0][i, :] - std_force[0][i, :],
                                 mean_force[0][i, :] + std_force[0][i, :],
                                 color='b', alpha=0.2)
            axes[i].fill_between(np.linspace(0, 100, 5000),
                                 mean_force[1][i, :] - std_force[1][i, :],
                                 mean_force[1][i, :] + std_force[1][i, :],
                                 color='r', alpha=0.2)
            axes[i].set_xlim([0, 100])
        plt.legend(['left', 'right'])

    def plot_force_comparison(self, file_path, index):
        mean = []
        label = ['x', 'y', 'z']
        for (i, file) in enumerate(file_path):
            force = self.interpolate_data(data=self.get_forces(self.load_c3d(file)), index=index[i])
            m, std = self.mean_data(force)
            mean.append(m)

        fig, axes = plt.subplots(2, 3)
        axes = axes.flatten()
        fig.suptitle(f"comparison ground reaction forces")
        for i in range(3):
            axes[i].set_title(f"platform 1 : {label[i]} (L)")
            for m in mean:
                axes[i].plot(np.linspace(0, 100, 5000), m[0][i, :])
            axes[i].set_xlim([0, 100])
        for i in range(3):
            axes[i + 3].set_title(f"platform 2 : {label[i]} (R)")
            for m in mean:
                axes[i + 3].plot(np.linspace(0, 100, 5000), m[1][i, :])
            axes[i + 3].set_xlim([0, 100])
        plt.legend(file_path)