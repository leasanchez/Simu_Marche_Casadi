import numpy as np
from scipy import interpolate
from casadi import vertcat

class Constraints:
    def __init__(self):
        self.x = np.array(
        [0, 0.174533, 0.349066, 0.523599, 0.698132, 0.872665, 1.0472, 1.22173, 1.39626, 1.5708, 1.74533, 1.91986,
         2.0944])

    # -- tibia_r --
    def tibia_r_Ty(self, ocp, nlp, t, x, u, p, q_ref_idx, q_target_idx):
        y=np.array([0, 0.000479, 0.000835, 0.001086, 0.001251, 0.001346, 0.001391, 0.001403, 0.0014, 0.0014, 0.001421,
                      0.001481, 0.001599])
        f = interpolate.interp1d(self.x, y, kind="cubic")
        nq = nlp["nbQ"]
        val = []
        for v in x:
            q = v[:nq]
            val = vertcat(val, q[q_target_idx] - 0.95799999999999996 * f(q[q_ref_idx]))
        return val

    def tibia_r_Tz(self, ocp, nlp, t, x, u, p, q_ref, q_target):
        y=np.array(
            [0, 0.000988, 0.001899, 0.002734, 0.003492, 0.004173, 0.004777, 0.005305, 0.005756, 0.00613, 0.006427,
             0.006648, 0.006792])
        f = interpolate.interp1d(self.x, y, kind="cubic")
        val = q_target - 0.95799999999999996 * f(q_ref)
        return val

    def tibia_r_Rz(self, q_ref, q_target):
        y=np.array(
            [0, 0.0126809, 0.0226969, 0.0296054, 0.0332049, 0.0335354, 0.0308779, 0.0257548, 0.0189295, 0.011407,
             0.00443314, -0.00050475, -0.0016782])
        f = interpolate.interp1d(self.x, y, kind="cubic")
        val = q_target - f(q_ref)
        return val

    def tibia_r_Ry(self, q_ref, q_target):
        y=np.array([0, 0.059461, 0.109399, 0.150618, 0.18392, 0.210107, 0.229983, 0.24435, 0.254012, 0.25977, 0.262428, 0.262788, 0.261654])
        f = interpolate.interp1d(self.x, y, kind="cubic")
        val = q_target - f(q_ref)
        return val

    # -- patella_r --
    def patella_r_Tx(self, q_ref, q_target):
        y = np.array(
            [0.0524, 0.0488, 0.0437, 0.0371, 0.0296, 0.0216, 0.0136, 0.0057, -0.0019, -0.0088, -0.0148, -0.0196,
             -0.0227])
        f = interpolate.interp1d(self.x, y, kind="cubic")
        val = q_target - 0.95799999999999996 * f(q_ref)
        return val

    def patella_r_Ty(self, q_ref, q_target):
        y=np.array([-0.0108, -0.019, -0.0263, -0.0322, -0.0367, -0.0395, -0.0408, -0.0404, -0.0384, -0.0349, -0.0301, -0.0245, -0.0187])
        f = interpolate.interp1d(self.x, y, kind="cubic")
        val = q_target - 0.95799999999999996 * f(q_ref)
        return val

    def patella_r_Rz(self, q_ref, q_target):
        y=np.array([0.00113686, -0.00629212, -0.105582, -0.253683, -0.414245, -0.579047, -0.747244, -0.91799, -1.09044, -1.26379, -1.43763, -1.61186, -1.78634])
        f = interpolate.interp1d(self.x, y, kind="cubic")
        val = q_target - f(q_ref)
        return val

    # -- tibia_l --
    def tibia_l_Ty(self, q_ref, q_target):
        y = np.array([0, 0.000479, 0.000835, 0.001086, 0.001251, 0.001346, 0.001391, 0.001403, 0.0014, 0.0014, 0.001421,
                      0.001481, 0.001599])
        f = interpolate.interp1d(self.x, y, kind="cubic")
        val = q_target - 0.95799999999999996 * f(-q_ref)
        return val

    def tibia_l_Tz(self, q_ref, q_target):
        y = np.array(
            [0, -0.000988, -0.001899, -0.002734, -0.003492, -0.004173, -0.004777, -0.005305, -0.005756, -0.00613,
             -0.006427, -0.006648, -0.006792])
        f = interpolate.interp1d(self.x, y, kind="cubic")
        val = q_target - 0.95799999999999996 * f(-q_ref)
        return val

    def tibia_l_Rz(self, q_ref, q_target):
        y=np.array([0, 0.0126809, 0.0226969, 0.0296054, 0.0332049, 0.0335354, 0.0308779, 0.0257548, 0.0189295, 0.011407, 0.00443314, -0.00050475, -0.0016782])
        f = interpolate.interp1d(self.x, y, kind="cubic")
        val = q_target - f(-q_ref)
        return val

    def tibia_l_Ry(self, q_ref, q_target):
        y=np.array([0, -0.059461, -0.109399, -0.150618, -0.18392, -0.210107, -0.229983, -0.24435, -0.254012, -0.25977, -0.262428, -0.262788, -0.261654])
        f = interpolate.interp1d(self.x, y, kind="cubic")
        val = q_target - f(-q_ref)
        return val

    # -- patella_l --
    def patella_l_Tx(self, q_ref, q_target):
        y=np.array([0.0524, 0.0488, 0.0437, 0.0371, 0.0296, 0.0216, 0.0136, 0.0057, -0.0019, -0.0088, -0.0148, -0.0196, -0.0227])
        f = interpolate.interp1d(self.x, y, kind="cubic")
        val = q_target - 0.95799999999999996 * f(q_ref)
        return val

    def patella_l_Ty(self, q_ref, q_target):
        y=np.array([-0.0108, -0.019, -0.0263, -0.0322, -0.0367, -0.0395, -0.0408, -0.0404, -0.0384, -0.0349, -0.0301, -0.0245, -0.0187])
        f = interpolate.interp1d(self.x, y, kind="cubic")
        val = q_target - 0.95799999999999996 * f(q_ref)
        return val

    def patella_l_Rz(self, q_ref, q_target):
        y=np.array([0.00113686, -0.00629212, -0.105582, -0.253683, -0.414245, -0.579047, -0.747244, -0.91799, -1.09044, -1.26379, -1.43763, -1.61186, -1.78634])
        f = interpolate.interp1d(self.x, y, kind="cubic")
        val = q_target - f(q_ref)
        return val