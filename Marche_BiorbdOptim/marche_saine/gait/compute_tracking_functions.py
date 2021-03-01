import numpy as np
from .contact_forces_function import contact


class tracking:
    def __init__(self, ocp, sol, data, muscles=False):
        self.muscles = muscles
        self.ocp = ocp
        self.sol = sol
        self.n_phases = len(ocp.nlp)
        self.n_q = ocp.nlp[0].model.nbQ()

        # reference tracked
        self.q_ref = data.q_ref
        self.qdot_ref = data.qdot_ref
        self.markers_ref = data.markers_ref
        self.grf_ref = data.grf_ref
        self.moments_ref = data.moments_ref
        self.cop_ref = data.cop_ref

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

        self.contact = contact(self.ocp, self.sol, self.muscles)

    def merged_reference(self, x):
        x_merged = {}
        for i in x.shape[0]:
            x_merged[i] = np.empty(sum(self.number_shooting_points) + 1)
            n_shoot = 0
            for phase in range(self.n_phases):
                x_merged[i, n_shoot:n_shoot + self.number_shooting_points[phase] + 1] = x[i, :][phase]
                n_shoot += self.number_shooting_points[phase]

    def compute_error_force_tracking_per_phase(self):
        diff_grf = []
        for phase in range(self.n_phases):
            diff = np.zeros(3)
            diff[0] = np.sqrt(np.mean((self.contact.forces["forces_r_X"][phase] - self.grf_ref[phase][0, :])**2))
            diff[1] = np.sqrt(np.mean((self.contact.forces["forces_r_Y"][phase] - self.grf_ref[phase][1, :]) ** 2))
            diff[2] = np.sqrt(np.mean((self.contact.forces["forces_r_Z"][phase] - self.grf_ref[phase][2, :]) ** 2))
            diff_grf.append(diff)
        return diff_grf

    def compute_error_q_tracking_per_phase(self):
        diff_q_tot = []
        for phase in range(self.n_phases):
            diff_q=[]
            for i in range(self.n_q):
                diff_q.append(np.sqrt(np.mean((self.q[phase][i, :] - self.q_ref[phase][i, :]) ** 2)))
            diff_q_tot.append(diff_q)
        return diff_q_tot