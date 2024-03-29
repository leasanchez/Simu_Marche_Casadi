import numpy as np
import biorbd
from casadi import MX, Function
from .contact_forces_function import contact
from matplotlib import pyplot as plt

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

def get_q_name(model):
    q_name = []
    for s in range(model.nbSegment()):
        seg_name = model.segment(s).name().to_string()
        for d in range(model.segment(s).nbDof()):
            dof_name = model.segment(s).nameDof(d).to_string()
            q_name.append(seg_name + "_" + dof_name)
    return q_name

def get_q_range(model):
    q_max = []
    q_min = []
    for s in range(model.nbSegment()):
        q_range = model.segment(s).QRanges()
        for r in q_range:
            q_max.append(r.max())
            q_min.append(r.min())
    return q_min, q_max

def get_qdot_range(model):
    qdot_max = []
    qdot_min = []
    for s in range(model.nbSegment()):
        qdot_range = model.segment(s).QDotRanges()
        for r in qdot_range:
            qdot_max.append(r.max())
            qdot_min.append(r.min())
    return qdot_min, qdot_max

class tracking:
    def __init__(self, ocp, sol, data, muscles=False):
        self.muscles = muscles
        self.ocp = ocp
        self.sol = sol
        self.sol_merged = sol.merge_phases()
        self.n_phases = len(ocp.nlp)

        # reference tracked
        self.data = data
        self.q_ref = data.q_ref
        self.qdot_ref = data.qdot_ref
        self.markers_ref = data.markers_ref
        self.grf_ref = data.grf_ref
        self.moments_ref = data.moments_ref
        self.cop_ref = data.cop_ref
        self.nb_phases = len(self.q_ref)

        # markers indices
        self.markers_pelvis = [0, 1, 2, 3]
        self.markers_anat = [4, 9, 10, 11, 12, 17, 18]
        self.markers_tissus = [5, 6, 7, 8, 13, 14, 15, 16]
        self.markers_pied = [19, 20, 21, 22, 23, 24, 25]

        # results data
        self.time = self.get_time_vector()
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

        # model data
        self.q_name = get_q_name(self.model[0])
        self.q_min, self.q_max = get_q_range(self.model[0])
        self.qdot_min, self.qdot_max = get_qdot_range(self.model[0])
        self.n_q = ocp.nlp[0].model.nbQ()
        self.n_markers = ocp.nlp[0].model.nbMarkers()
        self.n_muscles = ocp.nlp[0].model.nbMuscleTotal()
        self.contact = contact(self.ocp, self.sol, self.muscles)

    def get_results(self):
        q = self.sol_merged.states["q"]
        qdot = self.sol_merged.states["qdot"]
        tau = self.sol_merged.controls["tau"]
        muscle = self.sol_merged.controls["muscles"]
        return q, qdot, tau, muscle

    def get_time_vector(self):
        t = np.linspace(0, self.ocp.nlp[0].tf, self.ocp.nlp[0].ns + 1)
        for p in range(1, self.nb_phases):
            t = np.concatenate((t[:-1], t[-1] + np.linspace(0, self.ocp.nlp[p].tf, self.ocp.nlp[p].ns + 1)))
        return t

    def merged_reference(self, x):
        if (len(x[0].shape) == 3):
            x_merged = np.empty((x[0].shape[0], x[0].shape[1], sum(self.number_shooting_points) + 1))
            for i in range(x[0].shape[1]):
                n_shoot = 0
                for phase in range(self.n_phases):
                    x_merged[:, i, n_shoot:n_shoot + self.number_shooting_points[phase] + 1] = x[phase][:, i, :]
                    n_shoot += self.number_shooting_points[phase]
        else:
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
        diff_grf.append(np.sqrt((forces_sim["forces_r_X"][:sum(self.number_shooting_points[:-1])] - grf[0, :sum(self.number_shooting_points[:-1])])**2))
        diff_grf.append(np.sqrt((forces_sim["forces_r_Y"][:sum(self.number_shooting_points[:-1])] - grf[1, :sum(self.number_shooting_points[:-1])]) ** 2))
        diff_grf.append(np.sqrt((forces_sim["forces_r_Z"][:sum(self.number_shooting_points[:-1])] - grf[2, :sum(self.number_shooting_points[:-1])]) ** 2))
        return diff_grf

    def compute_mean_error_force_tracking(self):
        mean_diff_grf = []
        diff_grf = self.compute_error_force_tracking()
        mean_diff_grf.append(np.mean(diff_grf[0]))
        mean_diff_grf.append(np.mean(diff_grf[1]))
        mean_diff_grf.append(np.mean(diff_grf[2]))
        return mean_diff_grf

    def plot_grf(self):
        forces_sim = self.contact.merged_result(self.contact.forces)
        plt.figure()
        plt.plot(self.time, forces_sim["forces_r_X"], 'g')
        plt.plot(self.time, forces_sim["forces_r_Y"], 'b')
        plt.plot(self.time, forces_sim["forces_r_Z"], 'r')
        plt.legend(['X', 'Y', 'Z'])
        pt = 0
        for p in range(self.nb_phases):
            pt += self.ocp.nlp[p].tf
            plt.plot([pt, pt], [-200, 850], 'k--')
        plt.xlim([self.time[0], self.time[-1]])

    def plot_diff_grf(self):
        diff_grf = self.compute_error_force_tracking()
        plt.figure()
        plt.plot(self.time, np.concatenate([diff_grf[0], np.zeros(self.number_shooting_points[-1] + 1)]), 'g')
        plt.plot(self.time, np.concatenate([diff_grf[1], np.zeros(self.number_shooting_points[-1] + 1)]), 'b')
        plt.plot(self.time, np.concatenate([diff_grf[2], np.zeros(self.number_shooting_points[-1] + 1)]), 'r')
        plt.legend(['X', 'Y', 'Z'])
        pt = 0
        for p in range(self.nb_phases):
            pt += self.ocp.nlp[p].tf
            plt.plot([pt, pt], [0, 40], 'k--')
        plt.xlim([self.time[0], self.time[-1]])

    def plot_cop(self, reference=True):
        pos_heel = self.contact.compute_contact_position(0)[:, 26, 0]
        pos_m1 = self.contact.compute_contact_position(1)[:, 27, 0]
        pos_m5 = self.contact.compute_contact_position(1)[:, 28, 0]
        pos_toe = self.contact.compute_contact_position(2)[:, 29, 0]

        plt.figure()
        plt.scatter([pos_heel[0], pos_m1[0], pos_m5[0], pos_toe[0]], [pos_heel[1], pos_m1[1], pos_m5[1], pos_toe[1]], color='k')
        if reference:
            cop = self.merged_reference(self.data.cop_ref)
            plt.scatter(cop[0, :self.number_shooting_points[0]], cop[1, :self.number_shooting_points[0]], color='g')
            plt.scatter(cop[0, self.number_shooting_points[0]:sum(self.number_shooting_points[:2])],
                        cop[1, self.number_shooting_points[0]:sum(self.number_shooting_points[:2])], color='b')
            plt.scatter(cop[0, sum(self.number_shooting_points[:2]):sum(self.number_shooting_points[:3])],
                        cop[1, sum(self.number_shooting_points[:2]):sum(self.number_shooting_points[:3])], color='r')
        else:
            cop = self.contact.cop
            plt.scatter(cop['cop_r_X'][0], cop['cop_r_Y'][0], color='g')
            plt.scatter(cop['cop_r_X'][1], cop['cop_r_Y'][1], color='b')
            plt.scatter(cop['cop_r_X'][2][:-2], cop['cop_r_Y'][2][:-2], color='r')
        plt.yticks(np.arange(0.04, 0.22, step=0.02))
        plt.xticks(np.arange(0.7, 0.98, step=0.02))




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

    def plot_q_comp(self):
        q_ref = self.merged_reference(self.q_ref)
        q = self.sol_merged.states["q"]
        fig, ax = plt.subplots(3, 4)
        ax = ax.flatten()
        for t in range(3):
            ax[t].set_title(self.q_name[t])
            ax[t].plot(self.time, q[t, :], 'r')
            ax[t].plot(self.time, q_ref[t, :], 'k')
            ax[t].set_xlim([self.time[0], self.time[-1]])
            ax[t].set_ylim([self.q_min[t], self.q_max[t]])
            pt = 0
            for p in range(self.nb_phases):
                pt += self.ocp.nlp[p].tf
                ax[t].plot([pt, pt], [self.q_min[t], self.q_max[t]], 'k--')

        for r in range(3, q_ref.shape[0]):
            ax[r].set_title(self.q_name[r])
            ax[r].plot(self.time, q[r, :] * 180/np.pi, 'r')
            ax[r].plot(self.time, q_ref[r, :] * 180/np.pi, 'k')
            ax[r].set_xlim([self.time[0], self.time[-1]])
            ax[r].set_ylim([self.q_min[r] * 180/np.pi, self.q_max[r] * 180/np.pi])
            pt = 0
            for p in range(self.nb_phases):
                pt += self.ocp.nlp[p].tf
                ax[r].plot([pt, pt], [self.q_min[r]*180/np.pi, self.q_max[r]*180/np.pi], 'k--')

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
        markers_ref = self.merged_reference(self.data.markers_ref)
        q = self.sol_merged.states["q"]
        markers = biorbd.to_casadi_func("markers", self.model[0].markers, MX.sym("q", self.model[0].nbQ(), 1))
        markers_pos = np.zeros((3, self.model[0].nbMarkers(), q.shape[1]))
        for n in range(q.shape[1]):
            markers_pos[:, :, n] = markers(q[:, n:n + 1])

        diff_marker_tot = np.zeros((3, self.n_markers, q.shape[1]))
        for m in range(self.n_markers-4):
            diff_marker_tot[0, m, :] = np.sqrt((markers_ref[0, m, :] - markers_pos[0, m, :])**2)
            diff_marker_tot[1, m, :] = np.sqrt((markers_ref[1, m, :] - markers_pos[1, m, :]) ** 2)
            diff_marker_tot[2, m, :] = np.sqrt((markers_ref[2, m, :] - markers_pos[2, m, :]) ** 2)
        return diff_marker_tot

    def compute_mean_error_markers(self):
        diff_marker_tot = self.compute_error_markers_tracking()
        mean_diff_marker_tot = []
        for m in range(self.n_markers):
            x = np.mean(diff_marker_tot[0, m, :])
            y = np.mean(diff_marker_tot[1, m, :])
            z = np.mean(diff_marker_tot[2, m, :])
            mean_diff_marker_tot.append(np.mean([x, y, z]))
        return mean_diff_marker_tot

    def compute_error_markers_tracking_per_objectif(self):
        diff_marker_tot = {}
        err_markers = self.compute_mean_error_markers()

        diff_marker_tot["markers_pelvis"] = np.mean([err_markers[0], err_markers[1], err_markers[2], err_markers[3]])
        diff_marker_tot["markers_pied"] = np.mean([err_markers[19:]])
        diff_marker_tot["markers_anat"] = np.mean([err_markers[4], err_markers[9], err_markers[10], err_markers[11],
                                                   err_markers[12], err_markers[17], err_markers[18]])
        diff_marker_tot["markers_tissus"] = np.mean([err_markers[5], err_markers[6], err_markers[7], err_markers[8],
                                                     err_markers[13], err_markers[14], err_markers[15], err_markers[16]])
        return diff_marker_tot

    def plot_markers_error(self):
        err_markers = self.compute_mean_error_markers()
        label_markers = self.data.c3d_data.marker_names
        x = np.arange(len(label_markers))
        plt.bar(x, err_markers, color='tab:red')
        plt.xticks(x, labels=label_markers)

    def plot_heatmap_markers(self, markers_idx=range(26)):
        err_markers = self.compute_error_markers_tracking()
        mean_err_marker = (err_markers[0, :, :] + err_markers[1, :, :] +err_markers[2, :, :])/3

        # --- plot heatmap -- #
        fig, ax = plt.subplots()
        im = ax.imshow(mean_err_marker[markers_idx, :])

        # --- create colorbar --- #
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('error markers', rotation=-90)

        # --- label ticks --- #
        marker_label = self.data.c3d_data.marker_names
        marker_label_plot = []
        for m in markers_idx:
            marker_label_plot.append(marker_label[m])
        ax.set_yticks(np.arange(len(marker_label_plot)))
        ax.set_yticklabels(marker_label_plot)



    def plot_markers_error_per_objectif(self):
        err_markers = self.compute_error_markers_tracking_per_objectif()
        label_markers = ["pelvis", "anatomique", "tissus", "pied"]
        y = [err_markers["markers_pelvis"], err_markers["markers_anat"], err_markers["markers_tissus"], err_markers["markers_pied"]]
        x = np.arange(len(label_markers))
        plt.bar(x, y, color='tab:red')
        plt.xticks(x, labels=label_markers)

    def plot_comp_emg(self):
        emg_ref = self.merged_reference(self.data.excitation_ref)
        emg = self.sol_merged.controls["muscles"]
        fig, axes = plt.subplots(4, 5, sharex=True, sharey=True)
        axes = axes.flatten()
        for m in range(self.n_muscles):
            axes[m].set_title(self.model[0].muscle(m).name().to_string())
            axes[m].plot(self.time, emg_ref[m, :], 'k')
            axes[m].plot(self.time, emg[m, :], 'r')
            pt = 0
            for p in range(self.nb_phases):
                pt += self.ocp.nlp[p].tf
                axes[m].plot([pt, pt], [0, 1], 'k--')
            axes[m].set_xlim([self.time[0], self.time[-1]])
            axes[m].set_ylim([0, 1])
        plt.legend(['reference', 'simulation'])


