import numpy as np
from .Utils_start import utils
from casadi import MX, Function

def markers_func_casadi(model):
    symbolic_q = MX.sym("q", model.nbQ(), 1)
    markers_func = []
    for m in range(model.nbMarkers()):
        markers_func.append(Function(
            "ForwardKin",
            [symbolic_q], [model.marker(symbolic_q, m).to_mx()],
            ["q"],
            ["markers"],
        ).expand())
    return markers_func

class contact:
    def __init__(self, ocp, sol, muscles=False):
        self.muscles = muscles
        self.ocp = ocp
        self.sol = sol
        self.model = ocp.nlp[0].model

        # results data
        self.q = sol.states["q"]
        self.q_dot = sol.states["qdot"]
        self.tau = sol.controls["tau"]
        if self.muscles:
            self.activations = sol.controls["muscles"]

        # contact
        self.position={}
        self.set_contact_position()
        self.individual_forces = {}
        self.forces = {}
        self.labels_forces = ['Heel_r_X',
                              'Heel_r_Y',
                              'Heel_r_Z',
                              'Meta_1_r_X',
                              'Meta_1_r_Y',
                              'Meta_1_r_Z',
                              'Meta_5_r_X',
                              'Meta_5_r_Y',
                              'Meta_5_r_Z',
                              'Heel_l_X',
                              'Heel_l_Y',
                              'Heel_l_Z',
                              'Meta_1_l_X',
                              'Meta_1_l_Y',
                              'Meta_1_l_Z',
                              'Meta_5_l_X',
                              'Meta_5_l_Y',
                              'Meta_5_l_Z']
        self.set_individual_forces()
        self.set_sum_forces()
        self.individual_moments={}
        self.moments = {}
        self.set_individual_moments()
        self.set_sum_moments()
        self.cop = {}
        self.set_cop()



    def compute_contact_position(self):
        markers_func = markers_func_casadi(self.model)
        marker_pos = np.empty((3, self.model.nbMarkers(), self.q.shape[1]))
        for m in range(self.model.nbMarkers()):
            for n in range(self.q.shape[1]):
                marker_pos[:, m, n:n+1] = markers_func[m](self.q[:, n])
        return marker_pos

    def set_contact_position(self):
        markers_pos=self.compute_contact_position()
        self.position["Heel_r"] = markers_pos[:, 31, :]
        self.position["Meta_1_r"] = markers_pos[:, 32, :]
        self.position["Meta_5_r"] = markers_pos[:, 33, :]
        self.position["Heel_l"] = markers_pos[:, 55, :]
        self.position["Meta_1_l"] = markers_pos[:, 56, :]
        self.position["Meta_5_l"] = markers_pos[:, 57, :]

    def compute_individual_forces(self):
        forces = np.empty((self.model.nbContacts(), self.q.shape[1]))
        for n in range(self.ocp.nlp[0].ns + 1):
            x = np.concatenate([self.q[:, n], self.q_dot[:, n]])
            if self.muscles:
                u = np.concatenate([self.tau[:, n], self.activations[:, n]])
            else:
                u = self.tau[:, n]
            forces[:, n:n+1] = self.ocp.nlp[0].contact_forces_func(x, u, 0)
        return forces

    def set_individual_forces(self):
        c_name = utils.get_contact_name(self.model)
        forces_sim = self.compute_individual_forces()
        for name in self.labels_forces:
            if name in c_name:
                self.individual_forces[name] = forces_sim[c_name.index(name), :]
            else:
                self.individual_forces[name] = np.zeros(self.q.shape[1])

    def set_sum_forces(self):
        self.forces["forces_r_X"] = self.individual_forces["Heel_r_X"] + self.individual_forces["Meta_1_r_X"] + self.individual_forces["Meta_5_r_X"]
        self.forces["forces_r_Y"] = self.individual_forces["Heel_r_Y"] + self.individual_forces["Meta_1_r_Y"] + self.individual_forces["Meta_5_r_Y"]
        self.forces["forces_r_Z"] = self.individual_forces["Heel_r_Z"] + self.individual_forces["Meta_1_r_Z"] + self.individual_forces["Meta_5_r_Z"]
        self.forces["forces_l_X"] = self.individual_forces["Heel_l_X"] + self.individual_forces["Meta_1_l_X"] + self.individual_forces["Meta_5_l_X"]
        self.forces["forces_l_Y"] = self.individual_forces["Heel_l_Y"] + self.individual_forces["Meta_1_l_Y"] + self.individual_forces["Meta_5_l_Y"]
        self.forces["forces_l_Z"] = self.individual_forces["Heel_l_Z"] + self.individual_forces["Meta_1_l_Z"] + self.individual_forces["Meta_5_l_Z"]

    def set_individual_moments(self):
        for name in self.position.keys():
            self.individual_moments[f"{name}_X"] = self.position[name][1, :] * self.individual_forces[f"{name}_Z"]
            self.individual_moments[f"{name}_Y"] = -self.position[name][0, :] * self.individual_forces[f"{name}_Z"]
            self.individual_moments[f"{name}_Z"] = self.position[name][0, :] * self.individual_forces[f"{name}_Y"] \
                                                   - self.position[name][1, :] * self.individual_forces[f"{name}_X"]
    def set_sum_moments(self):
        self.moments["moments_r_X"] = self.individual_moments["Heel_r_X"] + self.individual_moments["Meta_1_r_X"] + self.individual_moments["Meta_5_r_X"]
        self.moments["moments_r_Y"] = self.individual_moments["Heel_r_Y"] + self.individual_moments["Meta_1_r_Y"] + self.individual_moments["Meta_5_r_Y"]
        self.moments["moments_r_Z"] = self.individual_moments["Heel_r_Z"] + self.individual_moments["Meta_1_r_Z"] + self.individual_moments["Meta_5_r_Z"]
        self.moments["moments_l_X"] = self.individual_moments["Heel_l_X"] + self.individual_moments["Meta_1_l_X"] + self.individual_moments["Meta_5_l_X"]
        self.moments["moments_l_Y"] = self.individual_moments["Heel_l_Y"] + self.individual_moments["Meta_1_l_Y"] + self.individual_moments["Meta_5_l_Y"]
        self.moments["moments_l_Z"] = self.individual_moments["Heel_l_Z"] + self.individual_moments["Meta_1_l_Z"] + self.individual_moments["Meta_5_l_Z"]

    def set_cop(self):
        Fz_r = self.forces["forces_r_Z"]
        Fz_r[Fz_r == 0] = np.nan
        self.cop["cop_r_X"] = -self.moments["moments_r_Y"] / Fz_r
        self.cop["cop_r_Y"] = self.moments["moments_r_X"] / Fz_r
        self.cop["cop_r_Z"] = np.zeros(self.cop["cop_r_Y"].shape[0])

        Fz_l = self.forces["forces_l_Z"]
        Fz_l[Fz_l == 0] = np.nan
        self.cop["cop_l_X"] = -self.moments["moments_l_Y"] / Fz_r
        self.cop["cop_l_Y"] = self.moments["moments_l_X"] / Fz_r
        self.cop["cop_l_Z"] = np.zeros(self.cop["cop_l_Y"].shape[0])