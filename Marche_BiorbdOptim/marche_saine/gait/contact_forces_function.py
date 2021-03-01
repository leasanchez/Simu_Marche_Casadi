import numpy as np
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
        self.n_phases = len(ocp.nlp)

        # results data
        self.model = []
        self.number_shooting_points = []
        self.q = []
        self.q_dot = []
        self.tau = []
        if self.muscles:
            self.activations = []

        for p in range(self.n_phases):
            self.model.append(self.ocp.nlp[p].model)
            self.number_shooting_points.append(self.ocp.nlp[p].ns)
            self.q.append(sol.states[p]["q"])
            self.q_dot.append(sol.states[p]["qdot"])
            self.tau.append(sol.controls[p]["tau"])
            if self.muscles:
                self.activations.append(sol.controls[p]["muscles"])

        # contact
        self.contact_name = []
        self.get_contact_name()
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
                              'Toe_r_X',
                              'Toe_r_Y',
                              'Toe_r_Z']
        self.set_individual_forces()
        self.set_sum_forces()
        self.individual_moments={}
        self.moments = {}
        self.set_individual_moments()
        self.cop = {}
        self.set_cop()


    def get_contact_name(self):
        cn = []
        for p in range(self.n_phases):
            for name in self.model[p].contactNames():
                cn.append(name.to_string())
            self.contact_name.append(cn)

    def compute_contact_position(self, phase):
        markers_func = markers_func_casadi(self.model[phase])
        marker_pos = np.empty((3, self.model[phase].nbMarkers(), self.q[phase].shape[1]))
        for m in range(self.model[phase].nbMarkers()):
            for n in range(self.q[phase].shape[1]):
                marker_pos[:, m, n:n+1] = markers_func[m](self.q[phase][:, n])
        return marker_pos

    def set_contact_position(self):
        self.position["Heel_r"] = []
        self.position["Meta_1_r"] = []
        self.position["Meta_5_r"] = []
        self.position["Toe_r"] = []
        for phase in range(self.n_phases):
            markers_pos=self.compute_contact_position(p)
            self.position["Heel_r"].append(markers_pos[:, 26, :])
            self.position["Meta_1_r"].append(markers_pos[:, 27, :])
            self.position["Meta_5_r"].append(markers_pos[:, 28, :])
            self.position["Toe_r"].append(markers_pos[:, 29, :])

    def compute_individual_forces(self, phase):
        forces = np.empty((self.model[phase].nbContacts(), self.q[phase].shape[1]))
        for n in range(self.ocp.nlp[phase].ns + 1):
            x = np.concatenate([self.q[phase][:, n], self.q_dot[phase][:, n]])
            if self.muscles:
                u = np.concatenate([self.tau[phase][:, n], self.activations[phase][:, n]])
            else:
                u = self.tau[phase][:, n]
            forces[:, n:n+1] = self.ocp.nlp[phase].contact_forces_func(x, u, 0)
        return forces

    def set_individual_forces(self):
        for name in self.labels_forces:
            self.individual_forces[name] = []
        for phase in range(self.n_phases):
            forces_sim = self.compute_individual_forces(phase)
            for name in self.labels_forces:
                if name in self.contact_name[phase]:
                    self.individual_forces[name].append(forces_sim[self.contact_name[phase].index(name), :])
                else:
                    self.individual_forces[name].append(np.zeros(self.q[phase].shape[1]))

    def set_sum_forces(self):
        self.forces["forces_r_X"] = []
        self.forces["forces_r_Y"] = []
        self.forces["forces_r_Z"] = []
        for phase in range(self.n_phases):
            self.forces["forces_r_X"].append(self.individual_forces["Heel_r_X"][phase] \
                                        + self.individual_forces["Meta_1_r_X"][phase] \
                                        + self.individual_forces["Meta_5_r_X"][phase] \
                                        + self.individual_forces["Toe_r_X"][phase])
            self.forces["forces_r_Y"].append(self.individual_forces["Heel_r_Y"][phase] \
                                        + self.individual_forces["Meta_1_r_Y"][phase] \
                                        + self.individual_forces["Meta_5_r_Y"][phase] \
                                        + self.individual_forces["Toe_r_Y"][phase])
            self.forces["forces_r_Z"].append(self.individual_forces["Heel_r_Z"][phase]\
                                        + self.individual_forces["Meta_1_r_Z"][phase]\
                                        + self.individual_forces["Meta_5_r_Z"][phase]\
                                        + self.individual_forces["Toe_r_Z"][phase])

    def set_individual_moments(self):
        for name in self.position.keys():
            self.individual_moments[f"{name}_X"] = []
            self.individual_moments[f"{name}_Y"] = []
            self.individual_moments[f"{name}_Z"] = []
        for phase in range(self.n_phases):
            for name in self.position.keys():
                self.individual_moments[f"{name}_X"].append(self.position[name][phase][1, :] * self.individual_forces[f"{name}_Z"][phase])
                self.individual_moments[f"{name}_Y"].append(-self.position[name][phase][0, :] * self.individual_forces[f"{name}_Z"][phase])
                self.individual_moments[f"{name}_Z"].append(self.position[name][phase][0, :] * self.individual_forces[f"{name}_Y"][phase] \
                                                       - self.position[name][phase][1, :] * self.individual_forces[f"{name}_X"][phase])
    def set_sum_moments(self):
        self.moments["moments_r_X"] = []
        self.moments["moments_r_Y"] = []
        self.moments["moments_r_Z"] = []
        for phase in range(self.n_phases):
            self.moments["moments_r_X"].append(self.individual_moments["Heel_r_X"][phase]
                                               + self.individual_moments["Meta_1_r_X"][phase]
                                               + self.individual_moments["Meta_5_r_X"][phase]
                                               + self.individual_moments["Toe_r_X"][phase])
            self.moments["moments_r_Y"].append(self.individual_moments["Heel_r_Y"][phase]
                                               + self.individual_moments["Meta_1_r_Y"][phase]
                                               + self.individual_moments["Meta_5_r_Y"][phase]
                                               + self.individual_moments["Toe_r_Y"][phase])
            self.moments["moments_r_Z"].append(self.individual_moments["Heel_r_Z"][phase]
                                               + self.individual_moments["Meta_1_r_Z"][phase]
                                               + self.individual_moments["Meta_5_r_Z"][phase]
                                               + self.individual_moments["Toe_r_Z"][phase])

    def set_cop(self):
        self.cop["cop_r_X"] = []
        self.cop["cop_r_Y"] = []
        self.cop["cop_r_Z"] = []
        for phase in range(self.n_phases):
            Fz_r = self.forces["forces_r_Z"][phase]
            Fz_r[Fz_r == 0] = np.nan
            self.cop["cop_r_X"].append(-self.moments["moments_r_Y"][phase] / Fz_r)
            self.cop["cop_r_Y"].append(self.moments["moments_r_X"][phase] / Fz_r)
            self.cop["cop_r_Z"].append(np.zeros(self.cop["cop_r_Y"][phase].shape[0]))

    def merged_result(self, x):
        x_merged = {}
        for key in x.keys():
            x_merged[key] = np.empty(sum(self.number_shooting_points) + 1)
            n_shoot=0
            for phase in range(self.n_phases):
                x_merged[key][n_shoot:n_shoot + self.number_shooting_points[phase] + 1] = x[key][phase]
                n_shoot += self.number_shooting_points[phase]