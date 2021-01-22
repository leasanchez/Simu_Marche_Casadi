import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from bioptim import Data
from casadi import MX, Function


class Affichage:
    def __init__(self, ocp, sol, muscles=False, two_leg=False):
        self.muscles = muscles
        self.two_leg = two_leg
        self.ocp = ocp
        self.sol = sol
        states, controls = Data.get_data(ocp, sol)
        self.q = states["q"]
        self.q_dot = states["q_dot"]
        self.tau = controls["tau"]
        if self.muscles:
            self.activations = controls["muscles"]
        self.nb_phases = ocp.nb_phases
        self.nb_shooting = self.q.shape[1] - 1
        self.nb_q = ocp.nlp[0].model.nbQ()
        self.nb_markers = ocp.nlp[0].model.nbMarkers()
        self.t = self.get_time_vector()


    def get_q_name(self):
        model = self.ocp.nlp[0].model
        q_name = []
        for s in range(model.nbSegment()):
            seg_name = model.segment(s).name().to_string()
            for d in range(model.segment(s).nbDof()):
                dof_name = model.segment(s).nameDof(d).to_string()
                q_name.append(seg_name + "_" + dof_name)
        return q_name

    def get_q_range(self):
        model = self.ocp.nlp[0].model
        q_max = []
        q_min = []
        for s in range(model.nbSegment()):
            q_range = model.segment(s).QRanges()
            for r in q_range:
                q_max.append(r.max())
                q_min.append(r.min())
        return q_max, q_min

    def get_time_vector(self):
        if (self.nb_phases > 1):
            t = np.linspace(0, self.ocp.nlp[0].tf, self.ocp.nlp[0].ns + 1)
            for p in range(1, self.nb_phases):
                t = np.concatenate((t[:-1], t[-1] + np.linspace(0, self.ocp.nlp[p].tf, self.ocp.nlp[p].ns + 1)))
        else:
            t = np.linspace(0, self.ocp.nlp[0].tf, self.ocp.nlp[0].ns + 1)
        return t

    def compute_markers_position(self):
        # compute contact point position
        position = {}
        position["heel_R"] = np.zeros((3, self.nb_shooting + 1))
        position["meta1_R"] = np.zeros((3, self.nb_shooting + 1))
        position["meta5_R"] = np.zeros((3, self.nb_shooting + 1))
        position["toe_R"] = np.zeros((3, self.nb_shooting + 1))
        if self.two_leg:
            position["heel_L"] = np.zeros((3, self.nb_shooting + 1))
            position["meta1_L"] = np.zeros((3, self.nb_shooting + 1))
            position["meta5_L"] = np.zeros((3, self.nb_shooting + 1))

        symbolic_q = MX.sym("q", self.ocp.nlp[0].model.nbQ(), 1)
        markers_func = []
        for m in range(self.nb_markers):
            markers_func.append(Function(
                "ForwardKin",
                [symbolic_q],[self.ocp.nlp[0].model.marker(symbolic_q,m).to_mx()],
                ["q"],
                ["markers"],
                    ).expand())

        for n in range(self.nb_shooting + 1):
            Q = self.q[:, n]
            position["heel_R"][:, n:n+1] = markers_func[26](Q)
            position["meta1_R"][:, n:n+1] = markers_func[27](Q)
            position["meta5_R"][:, n:n+1] = markers_func[28](Q)
            position["toe_R"][:, n:n + 1] = markers_func[29](Q)
            if self.two_leg:
                position["heel_R"][:, n:n + 1] = markers_func[41](Q)
                position["meta1_R"][:, n:n + 1] = markers_func[43](Q)
                position["meta5_R"][:, n:n + 1] = markers_func[46](Q)
        return position

    def compute_individual_forces(self):
        if self.two_leg:
            labels_forces = ['Heel_r_X', 'Heel_r_Y', 'Heel_r_Z',
                             'Meta_1_r_X', 'Meta_1_r_Y', 'Meta_1_r_Z',
                             'Meta_5_r_X', 'Meta_5_r_Y', 'Meta_5_r_Z',
                             'Heel_l_X', 'Heel_l_Y', 'Heel_l_Z',
                             'Meta_1_l_X', 'Meta_1_l_Y', 'Meta_1_l_Z',
                             'Meta_5_l_X', 'Meta_5_l_Y', 'Meta_5_l_Z'
                             ]
        else:
            labels_forces = ['Heel_r_X', 'Heel_r_Y', 'Heel_r_Z',
                             'Meta_1_r_X', 'Meta_1_r_Y', 'Meta_1_r_Z',
                             'Meta_5_r_X', 'Meta_5_r_Y', 'Meta_5_r_Z',
                             'Toe_r_X', 'Toe_r_Y', 'Toe_r_Z']
        # --- dictionary for forces ---
        forces = {}
        for label in labels_forces:
            forces[label] = np.zeros(self.nb_shooting + 1)

        # COMPUTE FORCES FOR EACH PHASE
        if (self.nb_phases>1):
            n_shoot = 0
            for p in range(self.nb_phases):
                cn = self.ocp.nlp[p].model.contactNames() # get contact names for each model
                if (len(cn) > 0):
                    states_p, controls_p = Data.get_data(self.ocp, self.sol["x"], phase_idx=p)
                    Q = states_p["q"]
                    Qdot = states_p["q_dot"]
                    Tau = controls_p["tau"]
                    if self.muscles:
                        Activation = controls_p["muscles"]
                    for n in range(self.ocp.nlp[p].ns + 1):
                        s = np.concatenate([Q[:, n], Qdot[:, n]])
                        if self.muscles:
                            c = np.concatenate([Tau[:, n], Activation[:, n]])
                        else:
                            c=Tau[:, n]
                        forces_sim = self.ocp.nlp[p].contact_forces_func(s, c, 0)
                        for i, c in enumerate(cn):
                            if c.to_string() in forces:
                                forces[c.to_string()][n_shoot + n] = forces_sim[i]  # put corresponding forces in dictionnary
                    n_shoot += self.ocp.nlp[p].ns
        else:
            cn = self.ocp.nlp[0].model.contactNames()
            states_p, controls_p = Data.get_data(self.ocp, self.sol["x"])
            Q = states_p["q"]
            Qdot = states_p["q_dot"]
            Tau = controls_p["tau"]
            if self.muscles:
                Activation = controls_p["muscles"]
            for n in range(self.ocp.nlp[0].ns + 1):
                s = np.concatenate([Q[:, n], Qdot[:, n]])
                if self.muscles:
                    c=np.concatenate([Tau[:, n], Activation[:, n]])
                else:
                    c = Tau[:, n]
                forces_sim = self.ocp.nlp[0].contact_forces_func(s, c, 0)
                for i, c in enumerate(cn):
                    if c.to_string() in forces:
                        forces[c.to_string()][n] = forces_sim[i]
        return forces

    def compute_sum_forces_simu(self):
        forces = self.compute_individual_forces()
        coord_label = ['X', 'Y', 'Z']
        if self.two_leg:
            grf_simu = np.zeros((2, 3, self.nb_shooting))
            for i in range(3):
                grf_simu[0, i, :] = forces[f"Heel_r_{coord_label[i]}"] \
                                 + forces[f"Meta_1_r_{coord_label[i]}"] \
                                 + forces[f"Meta_5_r_{coord_label[i]}"] \
                                 + forces[f"Toe_r_{coord_label[i]}"],
                grf_simu[1, i, :] = forces[f"Heel_l_{coord_label[i]}"] \
                                 + forces[f"Meta_1_l_{coord_label[i]}"] \
                                 + forces[f"Meta_5_l_{coord_label[i]}"] \
                                 + forces[f"Toe_l_{coord_label[i]}"],
        else:
            grf_simu = np.zeros((3, self.nb_shooting + 1))
            for i in range(3):
                grf_simu[i:i+1, :] = forces[f"Heel_r_{coord_label[i]}"] \
                                 + forces[f"Meta_1_r_{coord_label[i]}"] \
                                 + forces[f"Meta_5_r_{coord_label[i]}"] \
                                 + forces[f"Toe_r_{coord_label[i]}"],
        return grf_simu

    def compute_contact_moments(self):
        moments = {} # init
        forces = self.compute_individual_forces()  # get forces
        position = self.compute_markers_position() # get markers position

        # compute moments
        moments["moments_X_R"] = position["heel_R"][1, :]*forces["Heel_r_Z"] \
                                 + position["meta1_R"][1, :]*forces["Meta_1_r_Z"] \
                                 + position["meta5_R"][1, :]*forces["Meta_5_r_Z"] \
                                 + position["toe_R"][1, :]*forces["Toe_r_Z"]
        moments["moments_Y_R"] = -position["heel_R"][0, :]*forces["Heel_r_Z"] \
                                 - position["meta1_R"][0, :]*forces["Meta_1_r_Z"] \
                                 - position["meta5_R"][0, :]*forces["Meta_5_r_Z"] \
                                 - position["toe_R"][0, :]*forces["Toe_r_Z"]
        moments["moments_Z_R"] = position["heel_R"][0, :]*forces["Heel_r_Y"] - position["heel_R"][1, :]*forces["Heel_r_X"]\
                                 + position["meta1_R"][0, :]*forces["Meta_1_r_Y"] - position["meta1_R"][1, :]*forces["Meta_1_r_X"]\
                                 + position["meta5_R"][0, :]*forces["Meta_5_r_Y"] - position["meta5_R"][1, :]*forces["Meta_5_r_X"] \
                                 + position["toe_R"][0, :]*forces["Toe_r_Y"] - position["toe_R"][1, :]*forces["Toe_r_X"]
        if self.two_leg:
            moments["moments_X_L"] = position["heel_L"][1, :]*forces["Heel_l_Z"] + position["meta1_L"][1, :]*forces["Meta_1_l_Z"] + position["meta5_L"][1, :]*forces["Meta_5_l_Z"]
            moments["moments_Y_L"] = -position["heel_L"][0, :]*forces["Heel_l_Z"] - position["meta1_L"][0, :]*forces["Meta_1_l_Z"] - position["meta5_L"][0, :]*forces["Meta_5_l_Z"]
            moments["moments_Z_L"] = position["heel_L"][0, :]*forces["Heel_l_Y"] - position["heel_L"][1, :]*forces["Heel_l_X"]\
                                 + position["meta1_L"][0, :]*forces["Meta_1_l_Y"] - position["meta1_L"][1, :]*forces["Meta_1_l_X"]\
                                 + position["meta5_L"][0, :]*forces["Meta_5_l_Y"] - position["meta5_L"][1, :]*forces["Meta_5_l_X"]
        return moments

    def compute_contact_forces_ref(self, grf_ref):
        if (self.nb_phases>1):
            forces_ref = {}
            if self.two_leg:
                forces_ref["force_X_R"] = grf_ref[0][0][0, :]
                forces_ref["force_Y_R"] = grf_ref[0][0][1, :]
                forces_ref["force_Z_R"] = grf_ref[0][0][2, :]
                forces_ref["force_X_L"] = grf_ref[0][1][0, :]
                forces_ref["force_Y_L"] = grf_ref[0][1][1, :]
                forces_ref["force_Z_L"] = grf_ref[0][1][2, :]
            else:
                forces_ref["force_X_R"] = grf_ref[0][0, :]
                forces_ref["force_Y_R"] = grf_ref[0][1, :]
                forces_ref["force_Z_R"] = grf_ref[0][2, :]
            for p in range(1, self.nb_phases):
                for i, f in enumerate(forces_ref):
                    forces_ref[f] = np.concatenate([forces_ref[f][:-1], grf_ref[p][i, :]])
                    if self.two_leg:
                        if (i<3): #R
                            forces_ref[f] = np.concatenate([forces_ref[f][:-1], grf_ref[p][0][i, :]])
                        else:
                            forces_ref[f] = np.concatenate([forces_ref[f][:-1], grf_ref[p][1][i-3, :]])
        else:
            forces_ref = {}
            if self.two_leg:
                forces_ref["force_X_R"] = grf_ref[0][0, :]
                forces_ref["force_Y_R"] = grf_ref[0][1, :]
                forces_ref["force_Z_R"] = grf_ref[0][2, :]
                forces_ref["force_X_L"] = grf_ref[1][0, :]
                forces_ref["force_Y_L"] = grf_ref[1][1, :]
                forces_ref["force_Z_L"] = grf_ref[1][2, :]
            else:
                forces_ref["force_X_R"] = grf_ref[0, :]
                forces_ref["force_Y_R"] = grf_ref[1, :]
                forces_ref["force_Z_R"] = grf_ref[2, :]
        return forces_ref

    def compute_CoP(self):
        CoP = np.zeros((3, self.nb_shooting + 1))
        moments = self.compute_contact_moments()
        forces = self.compute_individual_forces()
        FZ=forces["Heel_r_Z"] + forces["Meta_1_r_Z"] + forces["Meta_5_r_Z"] + forces["Toe_r_Z"]
        FZ[FZ == 0] = np.nan
        CoP[0, :] = -moments["moments_Y_R"] / FZ
        CoP[1, :] = moments["moments_X_R"] / FZ
        return CoP

    def compute_moments_at_CoP(self):
        CoP = self.compute_CoP()
        position = self.compute_markers_position()
        forces = self.compute_individual_forces()
        moments={}

        # compute moments
        moments["moments_X_R"] = (position["heel_R"][1, :] - CoP[1, :]) * forces["Heel_r_Z"] \
                                 + (position["meta1_R"][1, :] - CoP[1, :]) * forces["Meta_1_r_Z"] \
                                 + (position["meta5_R"][1, :] - CoP[1, :]) * forces["Meta_5_r_Z"] \
                                 + (position["toe_R"][1, :] - CoP[1, :]) * forces["Toe_r_Z"]
        moments["moments_Y_R"] = -(position["heel_R"][0, :] - CoP[0, :]) * forces["Heel_r_Z"] \
                                 - (position["meta1_R"][0, :] - CoP[0, :]) * forces["Meta_1_r_Z"] \
                                 - (position["meta5_R"][0, :] - CoP[0, :]) * forces["Meta_5_r_Z"] \
                                 - (position["toe_R"][0, :] - CoP[0, :]) * forces["Toe_r_Z"]
        moments["moments_Z_R"] = (position["heel_R"][0, :] - CoP[0, :]) * forces["Heel_r_Y"] - (position["heel_R"][1, :] - CoP[1, :])*forces["Heel_r_X"]\
                                 + (position["meta1_R"][0, :] - CoP[0, :])*forces["Meta_1_r_Y"] - (position["meta1_R"][1, :] - CoP[1, :])*forces["Meta_1_r_X"]\
                                 + (position["meta5_R"][0, :] - CoP[0, :])*forces["Meta_5_r_Y"] - (position["meta5_R"][1, :] - CoP[1, :])*forces["Meta_5_r_X"] \
                                 + (position["toe_R"][0, :] - CoP[0, :])*forces["Toe_r_Y"] - (position["toe_R"][1, :] - CoP[1, :])*forces["Toe_r_X"]
        if self.two_leg:
            moments["moments_X_L"] = position["heel_L"][1, :]*forces["Heel_l_Z"] + position["meta1_L"][1, :]*forces["Meta_1_l_Z"] + position["meta5_L"][1, :]*forces["Meta_5_l_Z"]
            moments["moments_Y_L"] = -position["heel_L"][0, :]*forces["Heel_l_Z"] - position["meta1_L"][0, :]*forces["Meta_1_l_Z"] - position["meta5_L"][0, :]*forces["Meta_5_l_Z"]
            moments["moments_Z_L"] = position["heel_L"][0, :]*forces["Heel_l_Y"] - position["heel_L"][1, :]*forces["Heel_l_X"]\
                                 + position["meta1_L"][0, :]*forces["Meta_1_l_Y"] - position["meta1_L"][1, :]*forces["Meta_1_l_X"]\
                                 + position["meta5_L"][0, :]*forces["Meta_5_l_Y"] - position["meta5_L"][1, :]*forces["Meta_5_l_X"]
        return moments

    def compute_mean_difference(self, x, x_ref):
        nb_x = x.shape[0]
        nb_phases = len(x_ref)
        mean_diff = []
        std = []
        for i in range(nb_x):
            if (nb_phases > 1):
                X_ref = x_ref[0][i, :]
                for p in range(1, nb_phases):
                    X_ref = np.concatenate([X_ref[:-1], x_ref[p][i, :]])
            else:
                X_ref = x_ref
            mean_diff.append(np.mean(np.sqrt((x[i, :] - X_ref) ** 2)))
            std.append(np.std(np.sqrt((x[i, :] - X_ref) ** 2)))
        return std, mean_diff

    def compute_max_difference(self, x, x_ref):
        nb_x = x.shape[0]
        nb_phases = len(x_ref)
        max_diff = []
        idx_max = []
        for i in range(nb_x):
            if (nb_phases > 1):
                X_ref = x_ref[0][i, :]
                for p in range(1, nb_phases):
                    X_ref = np.concatenate([X_ref[:-1], x_ref[p][i, :]])
            else:
                X_ref = x_ref
            diff = np.sqrt((x[i, :] - X_ref) ** 2)
            max_diff.append(np.max(diff))
            idx_max.append(np.where(diff == np.max(diff))[0][0])
        return idx_max, max_diff

    def compute_R2(self, x, x_ref):
        nb_x = x.shape[0]
        nb_phases = len(x_ref)
        R2 = []
        for i in range(nb_x):
            mean_x = np.repeat(np.mean(x[i, :]), x.shape[1])
            if (nb_phases > 1):
                X_ref = x_ref[0][i, :]
                for p in range(1, nb_phases):
                    X_ref = np.concatenate([X_ref[:-1], x_ref[p][i, :]])
            else:
                X_ref = x_ref
            diff_q_square = (x[i, :] - X_ref)**2
            diff_q_mean = (x[i, :] - mean_x)**2
            s = np.sum(diff_q_square) / np.sum(diff_q_mean)
            R2.append(np.abs(1-s))
        return R2


    def plot_q(self, q_ref=(), R2=False, RMSE=False):
        # --- plot q VS q_ref ---
        q_name = self.get_q_name()  # dof names
        q_max, q_min = self.get_q_range()
        n_column = int(self.nb_q/3) + (self.nb_q % 3 > 0)

        if R2:
            R2_value = self.compute_R2(x=self.q, x_ref=q_ref)
        if RMSE:
            std_value, RMSE_value = self.compute_mean_difference(x=self.q, x_ref=q_ref)

        figure, axes = plt.subplots(3,n_column)
        axes = axes.flatten()
        for i in range(self.nb_q):
            axes[i].plot(self.t, self.q[i, :], 'r-', alpha=0.5)
            # compute q_ref
            if (self.nb_phases>1):
                Q_ref = q_ref[0][i, :]
                for p in range(1, self.nb_phases):
                    Q_ref = np.concatenate([Q_ref[:-1], q_ref[p][i, :]])
            else:
                Q_ref = q_ref[i, :]
            axes[i].plot(self.t, Q_ref, 'b-', alpha=0.5)
            axes[i].scatter(self.t, self.q[i, :], color='r', s=3)
            axes[i].scatter(self.t, Q_ref, color='b', s=3)

            # plot phase transition
            if (self.nb_phases>1):
                pt=0
                for p in range(self.nb_phases):
                    pt += self.ocp.nlp[p].tf
                    axes[i].plot([pt, pt], [q_min[i], q_max[i]], 'k--')

            title = q_name[i]
            if R2:
                title = title + f" - R2 : {round(R2_value[i], 2)}"
            if RMSE :
                title = title + f" - RMSE : {round(RMSE_value[i], 2)} +/- {round(std_value[i], 2)}"
            axes[i].set_title(title)
            axes[i].set_ylim([q_min[i], q_max[i]])
        plt.legend(['simulated', 'reference'])

    def plot_tau(self):
        # --- tau
        q_name = self.get_q_name()  # dof names
        n_column = int(self.nb_q/3) + (self.nb_q % 3 > 0)
        figure, axes = plt.subplots(3,n_column)
        axes = axes.flatten()
        for i in range(self.nb_q):
            axes[i].plot(self.t, self.tau[i, :], 'r-')
            axes[i].plot([self.t[0], self.t[-1]], [0, 0], 'k--')
            # plot phase transition
            if (self.nb_phases>1):
                pt=0
                for p in range(self.nb_phases):
                    pt += self.ocp.nlp[p].tf
                    axes[i].plot([pt, pt], [np.min(self.tau[i, :]), np.max(self.tau[i, :])], 'k--')
            axes[i].set_title(q_name[i])

    def plot_qdot(self):
        # --- plot qdot
        q_name = self.get_q_name()  # dof names
        n_column = int(self.nb_q/3) + (self.nb_q % 3 > 0)
        figure, axes = plt.subplots(3,n_column)
        axes = axes.flatten()
        for i in range(self.nb_q):
            axes[i].plot(self.t, self.q_dot[i, :], 'r-')
            axes[i].plot([self.t[0], self.t[-1]], [0, 0], 'k--')
            # plot phase transition
            if (self.nb_phases>1):
                pt=0
                for p in range(self.nb_phases):
                    pt += self.ocp.nlp[p].tf
                    axes[i].plot([pt, pt], [np.min(self.q_dot[i, :]), np.max(self.q_dot[i, :])], 'k--')
            axes[i].set_title(q_name[i])

    def plot_activation(self, excitations_ref):
        nb_mus = self.ocp.nlp[0].model.nbMuscleTotal()
        figure, axes = plt.subplots(3, 6)
        axes = axes.flatten()
        for i in range(nb_mus):
            axes[i].plot(self.t, self.activations[i, :], 'r-')
            if (self.nb_phases > 1):
                a_plot = excitations_ref[0][i, :]
                for p in range(1, self.nb_phases):
                    a_plot = np.concatenate([a_plot[:-1], excitations_ref[p][i, :]])
            else:
                a_plot = excitations_ref[i, :]
            axes[i].plot(self.t, a_plot, 'b--')
            if (self.nb_phases > 1):
                pt = 0
                for p in range(self.nb_phases):
                    pt += self.ocp.nlp[p].tf
                    axes[i].plot([pt, pt], [np.min(a_plot), np.max(a_plot)], 'k--')
            axes[i].set_title(self.ocp.nlp[0].model.muscle(i).name().to_string())
            axes[i].set_ylim([0.0, 1.01])

    def plot_individual_forces(self):
        forces=self.compute_individual_forces()
        if self.two_leg:
            figureR, axesR = plt.subplots(3, 3)
            axesR = axesR.flatten()
            figureL, axesL = plt.subplots(3, 3)
            axesL = axesL.flatten()

            for i, f in enumerate(forces):
                if (i<9):
                    ax = axesR[i]
                else:
                    ax = axesL[i-9]
                ax.scatter(self.t, forces[f], color='r', s=3)
                ax.plot(self.t, forces[f], 'r-', alpha=0.5)
                if (i == 2) or (i == 5) or (i == 8):
                    ax.plot([self.t[0], self.t[-1]], [0, 0], 'k--')
                ax[i].set_title(f)
                if (self.nb_phases > 1):
                    pt = 0
                    for p in range(self.nb_phases):
                        pt += self.ocp.nlp[p].tf
                        ax.plot([pt, pt], [np.min(forces[f]), np.max(forces[f])], 'k--')
        else:
            figure, axes = plt.subplots(4, 3)
            axes = axes.flatten()
            for i, f in enumerate(forces):
                axes[i].scatter(self.t, forces[f], color='r', s=3)
                axes[i].plot(self.t, forces[f], 'r-', alpha=0.5)
                if (i==2) or (i==5) or (i==8):
                    axes[i].plot([self.t[0], self.t[-1]], [0, 0], 'k--')
                axes[i].set_title(f)
                if (self.nb_phases>1):
                    pt = 0
                    for p in range(self.nb_phases):
                        pt += self.ocp.nlp[p].tf
                        axes[i].plot([pt, pt], [np.min(forces[f]), np.max(forces[f])], 'k--')

    def plot_sum_forces(self, grf_ref):
        # PLOT SUM FORCES VS PLATEFORME FORCES
        forces = self.compute_individual_forces() # get individual forces
        forces_ref = self.compute_contact_forces_ref(grf_ref) # get forces_ref

        coord_label = ['X', 'Y', 'Z']
        if self.two_leg:
            figureR, axesR = plt.subplots(1, 3)
            axesR = axesR.flatten()
            figureL, axesL = plt.subplots(1, 3)
            axesL = axesL.flatten()

            for i in range(3):
                axesR[i].plot(self.t,
                             forces[f"Heel_r_{coord_label[i]}"]
                             + forces[f"Meta_1_r_{coord_label[i]}"]
                             + forces[f"Meta_5_r_{coord_label[i]}"]
                             + forces[f"Toe_r_{coord_label[i]}"],
                             'r-', alpha=0.5)
                axesR[i].plot(self.t, forces_ref[f"force_{coord_label[i]}_R"], 'b-', alpha=0.5)
                axesR[i].scatter(self.t,
                                forces[f"Heel_r_{coord_label[i]}"]
                                + forces[f"Meta_1_r_{coord_label[i]}"]
                                + forces[f"Meta_5_r_{coord_label[i]}"]
                                + forces[f"Toe_r_{coord_label[i]}"],
                                color='r', s=3)
                axesR[i].scatter(self.t, forces_ref[f"force_{coord_label[i]}_R"], color='b', s=3)
                axesR[i].set_title("Forces in " + coord_label[i] + " R")

                axesL[i].plot(self.t,
                             forces[f"Heel_l_{coord_label[i]}"]
                             + forces[f"Meta_1_l_{coord_label[i]}"]
                             + forces[f"Meta_5_l_{coord_label[i]}"],
                             'r-', alpha=0.5)
                axesL[i].plot(self.t, forces_ref[f"force_{coord_label[i]}_L"], 'b-', alpha=0.5)
                axesL[i].scatter(self.t,
                                forces[f"Heel_l_{coord_label[i]}"]
                                + forces[f"Meta_1_l_{coord_label[i]}"]
                                + forces[f"Meta_5_l_{coord_label[i]}"],
                                color='r', s=3)
                axesL[i].scatter(self.t, forces_ref[f"force_{coord_label[i]}_L"], color='b', s=3)
                axesL[i].set_title("Forces in " + coord_label[i] + " L")
                if (self.nb_phases > 1):
                    pt = 0
                    for p in range(self.nb_phases):
                        pt += self.ocp.nlp[p].tf
                        axesR[i].plot([pt, pt],
                                     [np.min(forces_ref[f"force_{coord_label[i]}_R"]),
                                      np.max(forces_ref[f"force_{coord_label[i]}_R"])],
                                     'k--')
                        axesL[i].plot([pt, pt],
                                     [np.min(forces_ref[f"force_{coord_label[i]}_L"]),
                                      np.max(forces_ref[f"force_{coord_label[i]}_L"])],
                                     'k--')
        else:
            figure, axes = plt.subplots(1, 3)
            axes = axes.flatten()
            for i in range(3):
                axes[i].plot(self.t,
                             forces[f"Heel_r_{coord_label[i]}"]
                             + forces[f"Meta_1_r_{coord_label[i]}"]
                             + forces[f"Meta_5_r_{coord_label[i]}"]
                             + forces[f"Toe_r_{coord_label[i]}"],
                             'r-', alpha=0.5)
                axes[i].plot(self.t, forces_ref[f"force_{coord_label[i]}_R"], 'b-', alpha=0.5)
                axes[i].scatter(self.t,
                             forces[f"Heel_r_{coord_label[i]}"]
                             + forces[f"Meta_1_r_{coord_label[i]}"]
                             + forces[f"Meta_5_r_{coord_label[i]}"]
                             + forces[f"Toe_r_{coord_label[i]}"],
                             color='r', s=3)
                axes[i].scatter(self.t, forces_ref[f"force_{coord_label[i]}_R"], color='b', s=3)
                axes[i].set_title("Forces in " + coord_label[i] + " R")
                if (self.nb_phases > 1):
                    pt = 0
                    for p in range(self.nb_phases):
                        pt += self.ocp.nlp[p].tf
                        axes[i].plot([pt, pt],
                                     [np.min(forces_ref[f"force_{coord_label[i]}_R"]), np.max(forces_ref[f"force_{coord_label[i]}_R"])],
                                     'k--')
        plt.legend(['simulated', 'reference'])


    def plot_sum_moments(self, M_ref):
        # PLOT MOMENTS
        moments = self.compute_moments_at_CoP() # compute moments at CoP from solution
        moments_ref = self.compute_contact_forces_ref(M_ref) # get moments_ref

        coord_label = ['X', 'Y', 'Z']
        figure, axes = plt.subplots(1, 3)
        axes = axes.flatten()
        for i in range(3):
            axes[i].plot(self.t, moments[f"moments_{coord_label[i]}_R"], 'r-', alpha=0.5)
            axes[i].plot(self.t, moments_ref[f"force_{coord_label[i]}_R"], 'b-', alpha=0.5)
            axes[i].scatter(self.t, moments[f"moments_{coord_label[i]}_R"], color='r', s=3)
            axes[i].scatter(self.t, moments_ref[f"force_{coord_label[i]}_R"], color='b', s=3)
            axes[i].set_title(f"Moments in {coord_label[i]} R at CoP")
        if (self.nb_phases > 1):
            for i in range(3):
                pt = 0
                for p in range(self.nb_phases):
                    pt += self.ocp.nlp[p].tf
                    axes[i].plot([pt, pt],
                                 [np.min(moments_ref[f"force_{coord_label[i]}_R"]),
                                  np.max(moments_ref[f"force_{coord_label[i]}_R"])],
                                 'k--')
        plt.legend(['simulated', 'reference'])

    def plot_CoP(self, CoP_ref):
        CoP = self.compute_CoP()
        position = self.compute_markers_position()
        color = ['green', 'red', 'blue']
        n_shoot = 0

        plt.figure()
        plt.title("CoP comparison")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")

        if (self.nb_phases > 1):
            for p in range(self.nb_phases - 1):
                plt.scatter(CoP[0, n_shoot: n_shoot + self.ocp.nlp[p].ns],
                            CoP[1, n_shoot: n_shoot + self.ocp.nlp[p].ns], c=color[p], marker='v', s=15)
                plt.scatter(CoP_ref[p][0, :-1], CoP_ref[p][1, :-1], c=color[p], marker='o', s=15)
                for i in range(self.ocp.nlp[p].ns):
                    plt.plot([CoP[0, n_shoot + i], CoP_ref[p][0, i]], [CoP[1, n_shoot + i], CoP_ref[p][1, i]],
                             color=color[p], alpha=0.2, linestyle='--')
                n_shoot += self.ocp.nlp[p].ns

            plt.plot(position["heel_R"][0, :self.ocp.nlp[0].ns + self.ocp.nlp[1].ns],
                     position["heel_R"][1, :self.ocp.nlp[0].ns + self.ocp.nlp[1].ns], 'k+', alpha=0.5)
            plt.plot(position["meta1_R"][0, self.ocp.nlp[0].ns:(self.nb_shooting - self.ocp.nlp[-1].ns)],
                     position["meta1_R"][1, self.ocp.nlp[0].ns:(self.nb_shooting - self.ocp.nlp[-1].ns)], 'k+', alpha=0.5)
            plt.plot(position["meta5_R"][0, self.ocp.nlp[0].ns :(self.nb_shooting - self.ocp.nlp[-1].ns)],
                     position["meta5_R"][1, self.ocp.nlp[0].ns :(self.nb_shooting - self.ocp.nlp[-1].ns)], 'k+', alpha=0.5)
            plt.plot(position["toe_R"][0, self.ocp.nlp[0].ns + self.ocp.nlp[1].ns :(self.nb_shooting - self.ocp.nlp[-1].ns)],
                     position["toe_R"][1, self.ocp.nlp[0].ns + self.ocp.nlp[1].ns :(self.nb_shooting - self.ocp.nlp[-1].ns)], 'k+', alpha=0.5)
        else:
            plt.scatter(CoP[0, :], CoP[1, :] , c=color[0], marker='v', s=15)
            plt.scatter(CoP_ref[0, :-1], CoP_ref[1, :-1], c=color[0], marker='o', s=15)
            for i in range(self.ocp.nlp[0].ns):
                plt.plot([CoP[0, n_shoot + i], CoP_ref[0, i]], [CoP[1, n_shoot + i], CoP_ref[1, i]],
                         color=color[0], alpha=0.2, linestyle='--')

            plt.plot(position["heel_R"][0, :], position["heel_R"][1, :], 'k+', alpha=0.5)
            plt.plot(position["meta1_R"][0, :], position["meta1_R"][1, :], 'k+', alpha=0.5)
            plt.plot(position["meta5_R"][0, :], position["meta5_R"][1, :], 'k+', alpha=0.5)
            plt.plot(position["toe_R"][0, :], position["toe_R"][1, :], 'k+', alpha=0.5)
        # plt.legend(['contact talon simulation', 'contact talon reference',
        #             'flatfoot simulation', 'flatfoot reference',
        #             'forefoot simulation', 'forefoot reference'])

    def plot_foot(self, ax, markers, idx_node, color="black", alpha=1.0):
        # plot pelvis
        ax.plot([markers[0, 0, idx_node], markers[0, 1, idx_node]],
                [markers[1, 0, idx_node], markers[1, 1, idx_node]],
                [markers[2, 0, idx_node], markers[2, 1, idx_node]],
                c=color, alpha=alpha, linestyle='dashed')
        ax.plot([markers[0, 1, idx_node], markers[0, 2, idx_node]],
                [markers[1, 1, idx_node], markers[1, 2, idx_node]],
                [markers[2, 1, idx_node], markers[2, 2, idx_node]],
                c=color, alpha=alpha, linestyle='dashed')
        ax.plot([markers[0, 2, idx_node], markers[0, 3, idx_node]],
                [markers[1, 2, idx_node], markers[1, 3, idx_node]],
                [markers[2, 2, idx_node], markers[2, 3, idx_node]],
                c=color, alpha=alpha, linestyle='dashed')
        ax.plot([markers[0, 0, idx_node], markers[0, 3, idx_node]],
                [markers[1, 0, idx_node], markers[1, 3, idx_node]],
                [markers[2, 0, idx_node], markers[2, 3, idx_node]],
                c=color, alpha=alpha, linestyle='dashed')
        # plot cuisse
        ax.plot([markers[0, 4, idx_node], (markers[0, 9, idx_node] + markers[0, 10, idx_node])/2],
                [markers[1, 4, idx_node], (markers[1, 9, idx_node] + markers[1, 10, idx_node])/2],
                [markers[2, 4, idx_node], (markers[2, 9, idx_node] + markers[2, 10, idx_node])/2],
                c=color, alpha=alpha, linestyle='dashed')
        #plot jambe
        ax.plot([(markers[0, 17, idx_node] + markers[0, 18, idx_node])/2, (markers[0, 9, idx_node] + markers[0, 10, idx_node])/2],
                [(markers[1, 17, idx_node] + markers[1, 18, idx_node])/2, (markers[1, 9, idx_node] + markers[1, 10, idx_node])/2],
                [(markers[2, 17, idx_node] + markers[2, 18, idx_node])/2, (markers[2, 9, idx_node] + markers[2, 10, idx_node])/2],
                c=color, alpha=alpha, linestyle='dashed')
        ax.plot([(markers[0, 17, idx_node] + markers[0, 18, idx_node])/2, markers[0, 19, idx_node]],
                [(markers[1, 17, idx_node] + markers[1, 18, idx_node])/2, markers[1, 19, idx_node]],
                [(markers[2, 17, idx_node] + markers[2, 18, idx_node])/2, markers[2, 19, idx_node]],
                c=color, alpha=alpha, linestyle='dashed')
        # plot pied
        ax.plot([markers[0, 19, idx_node], markers[0, 21, idx_node]],
                [markers[1, 19, idx_node], markers[1, 21, idx_node]],
                [markers[2, 19, idx_node], markers[2, 21, idx_node]],
                c=color, alpha=alpha, linestyle='dashed')
        ax.plot([markers[0, 21, idx_node], markers[0, 20, idx_node]],
                [markers[1, 21, idx_node], markers[1, 20, idx_node]],
                [markers[2, 21, idx_node], markers[2, 20, idx_node]],
                c=color, alpha=alpha, linestyle='dashed')
        ax.plot([markers[0, 20, idx_node], markers[0, 22, idx_node]],
                [markers[1, 20, idx_node], markers[1, 22, idx_node]],
                [markers[2, 20, idx_node], markers[2, 22, idx_node]],
                c=color, alpha=alpha, linestyle='dashed')
        ax.plot([markers[0, 22, idx_node], markers[0, 24, idx_node]],
                [markers[1, 22, idx_node], markers[1, 24, idx_node]],
                [markers[2, 22, idx_node], markers[2, 24, idx_node]],
                c=color, alpha=alpha, linestyle='dashed')
        ax.plot([markers[0, 24, idx_node], markers[0, 25, idx_node]],
                [markers[1, 24, idx_node], markers[1, 25, idx_node]],
                [markers[2, 24, idx_node], markers[2, 25, idx_node]],
                c=color, alpha=alpha, linestyle='dashed')
        ax.plot([markers[0, 25, idx_node], markers[0, 19, idx_node]],
                [markers[1, 25, idx_node], markers[1, 19, idx_node]],
                [markers[2, 25, idx_node], markers[2, 19, idx_node]],
                c=color, alpha=alpha, linestyle='dashed')

    def plot_stance_phase(self, markers, CoP, grf, markers_ref, CoP_ref=(), grf_ref=()):
        MAX_GRF = np.max(grf[2, :]) * 10
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        self.plot_foot(ax, markers=markers_ref, idx_node=0, color="black", alpha=0.5)
        for i in range(markers.shape[2]-1):
            self.plot_foot(ax, markers=markers, idx_node=i, color="red", alpha=0.5)
            ax.plot([CoP[0, i], CoP[0, i] + grf[0, i] / MAX_GRF],
                    [CoP[1, i], CoP[1, i] + grf[1, i] / MAX_GRF],
                    [CoP[2, i], grf[2, i] / MAX_GRF],
                    c='g')
            if len(CoP_ref) > 0:
                MAX_GRF_ref = np.max(grf_ref[2, :]) * 10
                ax.scatter(CoP_ref[0, i], CoP_ref[1, i], CoP_ref[2, i], c='b')
                ax.plot([CoP_ref[0, i], CoP_ref[0, i] + grf_ref[0, i] / MAX_GRF_ref],
                        [CoP_ref[1, i], CoP_ref[1, i] + grf_ref[1, i] / MAX_GRF_ref],
                        [CoP_ref[2, i], grf_ref[2, i] / MAX_GRF_ref],
                        c='b')
        for m in range(17, 26):
            # plot computed markers position
            ax.scatter(markers[0, m, 0], markers[1, m, 0], markers[2, m, 0], c="red", alpha=0.7)
            ax.scatter(markers[0, m, 1:], markers[1, m, 1:], markers[2, m, 1:], c="red", alpha=0.5)

            # plot reference markers position
            ax.scatter(markers_ref[0, m, 0],  markers_ref[1, m, 0],  markers_ref[2, m, 0],  c="black", alpha=0.7)
            ax.scatter(markers_ref[0, m, 1:], markers_ref[1, m, 1:], markers_ref[2, m, 1:], c="black", alpha=0.5)

        ax.scatter(CoP[0, :-1], CoP[1, :-1], CoP[2, :-1], c='g')
        min_x = np.round(np.min([markers[0, 19, 0], markers_ref[0, 2, 0]]), 1) - 0.1
        max_x = np.round(np.max([markers[0, 0, -1], markers_ref[0, 22, -1]]), 1) + 0.1
        ax.set_xlabel('x')
        ax.set_xlim([min_x, max_x])
        ax.set_ylabel('y')
        ax.set_ylim([-0.2, max_x - min_x - 0.2])
        ax.set_zlabel('z')
        # ax.set_zlim([-0.05, 0.2])


    def heatmap(self, data, row_labels, col_labels, ax=None,
                cbar_kw={}, cbarlabel="", **kwargs):
        """
        Create a heatmap from a numpy array and two lists of labels.

        Parameters
        ----------
        data
            A 2D numpy array of shape (N, M).
        row_labels
            A list or array of length N with the labels for the rows.
        col_labels
            A list or array of length M with the labels for the columns.
        ax
            A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
            not provided, use current axes or create a new one.  Optional.
        cbar_kw
            A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
        cbarlabel
            The label for the colorbar.  Optional.
        **kwargs
            All other arguments are forwarded to `imshow`.
        """

        if not ax:
            ax = plt.gca()

        # Plot the heatmap
        im = ax.imshow(data, **kwargs)

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        # We want to show all ticks...
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        # ... and label them with the respective list entries.
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)

        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False,
                       labeltop=True, labelbottom=False)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                 rotation_mode="anchor")

        # Turn spines off and create white grid.
        for edge, spine in ax.spines.items():
            spine.set_visible(False)

        ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        return im, cbar