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
        from force platform
        """
        corners = []
        platform = loaded_c3d["data"]["platform"]
        for p in platform:
            corners.append(p["corners"] * 1e-3)
        return corners

    def get_origin_position(self, loaded_c3d):
        """
        get the trajectory of the center of pressure (cop)
        from force platform
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

    def plot_cop(self, c3d_path):
        cop = self.get_cop(self.load_c3d(c3d_path))
        plt.figure()
        plt.plot(cop[0][1, :], cop[0][0, :])
        plt.plot(cop[1][1, :], cop[1][0, :])