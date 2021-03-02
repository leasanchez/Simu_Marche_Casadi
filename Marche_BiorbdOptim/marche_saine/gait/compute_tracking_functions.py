import numpy as np
from casadi import MX, Function
from .contact_forces_function import contact

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

class tracking:
    def __init__(self, ocp, sol, data, muscles=False):
        self.muscles = muscles
        self.ocp = ocp
        self.sol = sol
        self.sol_merged = sol.merge_phases()
        self.n_phases = len(ocp.nlp)
        self.n_q = ocp.nlp[0].model.nbQ()
        self.n_markers = ocp.nlp[0].model.nbMarkers()

        # reference tracked
        self.q_ref = data.q_ref
        self.qdot_ref = data.qdot_ref
        self.markers_ref = data.markers_ref
        self.grf_ref = data.grf_ref
        self.moments_ref = data.moments_ref
        self.cop_ref = data.cop_ref

        # markers indices
        self.markers_pelvis = [0, 1, 2, 3]
        self.markers_anat = [4, 9, 10, 11, 12, 17, 18]
        self.markers_tissus = [5, 6, 7, 8, 13, 14, 15, 16]
        self.markers_pied = [19, 20, 21, 22, 23, 24, 25]

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

    def get_results(self):
        q = self.sol_merged.states["q"]
        qdot = self.sol_merged.states["qdot"]
        tau = self.sol_merged.controls["tau"]
        muscle = self.sol_merged.controls["muscles"]
        return q, qdot, tau, muscle

    def merged_reference(self, x):
        x_merged = np.empty((x[0].shape[0], sum(self.number_shooting_points) + 1))
        for i in range(x[0].shape[0]):
            n_shoot = 0
            for phase in range(self.n_phases):
                x_merged[i, n_shoot:n_shoot + self.number_shooting_points[phase] + 1] = x[phase][i, :]
                n_shoot += self.number_shooting_points[phase]
        return x_merged

    def compute_error_force_tracking_per_phase(self):
        diff_grf = []
        for phase in range(self.n_phases-1):
            diff = np.zeros(3)
            diff[0] = np.sqrt(np.mean((self.contact.forces["forces_r_X"][phase] - self.grf_ref[phase][0, :])**2))
            diff[1] = np.sqrt(np.mean((self.contact.forces["forces_r_Y"][phase] - self.grf_ref[phase][1, :]) ** 2))
            diff[2] = np.sqrt(np.mean((self.contact.forces["forces_r_Z"][phase] - self.grf_ref[phase][2, :]) ** 2))
            diff_grf.append(diff)
        return diff_grf

    def compute_error_force_tracking(self):
        diff_grf = []
        grf = self.merged_reference(self.grf_ref)
        forces_sim = self.contact.merged_result(self.contact.forces)
        diff_grf.append(np.sqrt(np.mean((forces_sim["forces_r_X"][:sum(self.number_shooting_points[:-1])] - grf[0, :sum(self.number_shooting_points[:-1])])**2)))
        diff_grf.append(np.sqrt(np.mean((forces_sim["forces_r_Y"][:sum(self.number_shooting_points[:-1])] - grf[1, :sum(self.number_shooting_points[:-1])]) ** 2)))
        diff_grf.append(np.sqrt(np.mean((forces_sim["forces_r_Z"][:sum(self.number_shooting_points[:-1])] - grf[2, :sum(self.number_shooting_points[:-1])]) ** 2)))
        return diff_grf

    def compute_error_q_tracking_per_phase(self):
        diff_q_tot = []
        for phase in range(self.n_phases):
            diff_q=[]
            for i in range(self.n_q):
                diff_q.append(np.sqrt(np.mean((self.q[phase][i, :] - self.q_ref[phase][i, :]) ** 2)))
            diff_q_tot.append(diff_q)
        return diff_q_tot

    def compute_error_q_tracking(self):
        q_ref = self.merged_reference(self.q_ref)
        q = self.sol_merged.states["q"]
        diff_q_tot = []
        for i in range(self.n_q):
            diff_q_tot.append(np.sqrt(np.mean((q[i, :] - q_ref[i, :]) ** 2)))
        return diff_q_tot

    def compute_marker_position(self, phase):
        markers_func = markers_func_casadi(self.model[phase])
        marker_pos = np.empty((3, self.model[phase].nbMarkers(), self.q[phase].shape[1]))
        for m in range(self.model[phase].nbMarkers()):
            for n in range(self.q[phase].shape[1]):
                marker_pos[:, m, n:n+1] = markers_func[m](self.q[phase][:, n])
        return marker_pos

    def compute_error_markers_tracking_per_phase(self):
        diff_marker_tot = []
        for phase in range(self.n_phases):
            markers_pos = self.compute_marker_position(phase)
            diff_marker=[]
            for i in range(self.n_markers - 4):
                diff_marker.append(np.sqrt(np.mean((markers_pos[:, i, :] - self.markers_ref[phase][:, i, :]) ** 2)))
            diff_marker_tot.append(diff_marker)
        return diff_marker_tot

    def compute_error_markers_tracking(self):
        diff_marker_tot = []
        err_per_phase = self.compute_error_markers_tracking_per_phase()
        for i in range(self.n_markers - 4):
            err = 0
            for phase in range(self.n_phases):
                err += err_per_phase[phase][i]
            diff_marker_tot.append(err/self.n_phases)
        return diff_marker_tot

    def compute_error_markers_tracking_per_objectif(self):
        diff_marker_tot = {}
        err_markers = self.compute_error_markers_tracking()
        diff_marker_tot["markers_pelvis"] = np.mean(err_markers[self.markers_pelvis])
        diff_marker_tot["markers_pied"] = np.mean(err_markers[self.markers_pied])
        diff_marker_tot["markers_anat"] = np.mean(err_markers[self.markers_anat])
        diff_marker_tot["markers_tissus"] = np.mean(err_markers[self.markers_tissus])
        return diff_marker_tot