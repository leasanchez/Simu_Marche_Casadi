import numpy as np
import seaborn
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from casadi import MX, Function

class Affichage:
    def __init__(self, ocp, sol, muscles=False):
        seaborn.set_style("whitegrid")
        seaborn.color_palette('deep')

        # params function
        self.muscles = muscles
        self.ocp = ocp
        self.sol = sol

        # results data
        self.q = sol.states["q"]
        self.q_dot = sol.states["qdot"]
        self.tau = sol.controls["tau"]
        if self.muscles:
            self.activations = sol.controls["muscles"]

        # params simulation
        self.nb_shooting = self.q.shape[1] - 1
        self.nb_q = ocp.nlp[0].model.nbQ()
        if self.muscles:
            self.nb_muscle = ocp.nlp[0].model.nbMuscleTotal()
        self.nb_markers = ocp.nlp[0].model.nbMarkers()
        self.t = np.linspace(0, self.ocp.nlp[0].tf, self.ocp.nlp[0].ns + 1)


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

    def get_qdot_range(self):
        model = self.ocp.nlp[0].model
        qdot_max = []
        qdot_min = []
        for s in range(model.nbSegment()):
            qdot_range = model.segment(s).QDotRanges()
            for r in qdot_range:
                qdot_max.append(r.max())
                qdot_min.append(r.min())
        return qdot_max, qdot_min

    def get_contact_name(self):
        model = self.ocp.nlp[0].model
        contact_name=[]
        C_names = model.contactNames()
        for name in C_names:
            contact_name.append(name.to_string())
        return contact_name

    def compute_individual_forces(self):
        labels_forces = ['Heel_r_X', 'Heel_r_Y', 'Heel_r_Z',
                         'Meta_1_r_X', 'Meta_1_r_Y', 'Meta_1_r_Z',
                         'Meta_5_r_X', 'Meta_5_r_Y', 'Meta_5_r_Z',
                         'Heel_l_X', 'Heel_l_Y', 'Heel_l_Z',
                         'Meta_1_l_X', 'Meta_1_l_Y', 'Meta_1_l_Z',
                         'Meta_5_l_X', 'Meta_5_l_Y', 'Meta_5_l_Z'
                         ]
        # --- dictionary for forces ---
        forces = {}
        for label in labels_forces:
            forces[label] = np.zeros(self.nb_shooting + 1)

        # --- COMPUTE FORCES FOR EACH PHASE ---
        contact_name = self.get_contact_name()
        for n in range(self.ocp.nlp[0].ns + 1):
            x = np.concatenate([self.q[:, n], self.q_dot[:, n]])
            if self.muscles:
                u = np.concatenate([self.tau[:, n], self.activations[:, n]])
            else:
                u = self.tau[:, n]
            forces_sim = self.ocp.nlp[0].contact_forces_func(x, u, 0)
            for (i, name) in enumerate(contact_name):
                if name in forces:
                    forces[name][n] = forces_sim[i]
        return forces

    def plot_forces(self):
        forces = self.compute_individual_forces()
        figure, axes = plt.subplots(3, 3, sharey=True, sharex=True)
        figure.suptitle('Contact forces')

        # --- plot heel --- #
        axes[0, 0].set_title("Heel")
        axes[0, 0].plot(self.t, forces["Heel_r_X"], color="red")
        axes[0, 0].plot(self.t, forces["Heel_l_X"], color="blue")
        axes[0, 0].set_ylabel("Forces in X (N)")

        axes[1, 0].plot(self.t, forces["Heel_r_Y"], color="red")
        axes[1, 0].plot(self.t, forces["Heel_l_Y"], color="blue")
        axes[1, 0].set_ylabel("Forces in Y (N)")

        axes[2, 0].plot(self.t, forces["Heel_r_Z"], color="red")
        axes[2, 0].plot(self.t, forces["Heel_l_Z"], color="blue")
        axes[2, 0].plot([self.t[0], self.t[-1]], [0, 0], color="black")
        axes[2, 0].set_ylabel("Forces in Z (N)")
        axes[2, 0].set_xlabel("Time (s)")

        # --- plot meta 1 --- #
        axes[0, 1].set_title("Meta 1")
        axes[0, 1].plot(self.t, forces["Meta_1_r_X"], color="red")
        axes[0, 1].plot(self.t, forces["Meta_1_l_X"], color="blue")

        axes[1, 1].plot(self.t, forces["Meta_1_r_Y"], color="red")
        axes[1, 1].plot(self.t, forces["Meta_1_l_Y"], color="blue")

        axes[2, 1].plot(self.t, forces["Meta_1_r_Z"], color="red")
        axes[2, 1].plot(self.t, forces["Meta_1_l_Z"], color="blue")
        axes[2, 1].plot([self.t[0], self.t[-1]], [0, 0], color="black")
        axes[2, 1].set_xlabel("Time (s)")

        # --- plot meta 5 --- #
        axes[0, 2].set_title("Meta 5")
        axes[0, 2].plot(self.t, forces["Meta_5_r_X"], color="red")
        axes[0, 2].plot(self.t, forces["Meta_5_l_X"], color="blue")

        axes[1, 2].plot(self.t, forces["Meta_5_r_Y"], color="red")
        axes[1, 2].plot(self.t, forces["Meta_5_l_Y"], color="blue")

        axes[2, 2].plot(self.t, forces["Meta_5_r_Z"], color="red")
        axes[2, 2].plot(self.t, forces["Meta_5_l_Z"], color="blue")
        axes[2, 2].plot([self.t[0], self.t[-1]], [0, 0], color="black")
        axes[2, 2].set_xlabel("Time (s)")
        axes[2, 2].legend(["Right", "Left"])


    def plot_q_symetry(self):
        q_max, q_min = self.get_q_range()

        # --- plot pelvis --- #
        figure, axes = plt.subplots(2, 3, sharex=True)
        figure.suptitle('Q pelvis')
        pelvis_label = ["Pelvis translation X",
                        "Pelvis translation Y",
                        "Pelvis translation Z",
                        "Pelvis Rotation",
                        "Pelvis Elevation/Depression Right",
                        "Pelvis Anterior/Posterior"]
        axes=axes.flatten()
        for i in range(6):
            axes[i].set_title(pelvis_label[i])
            if (i<3):
                axes[i].plot(self.t, self.q[i, :], color="red")
                axes[i].set_ylim([q_min[i], q_max[i]])
                axes[i].set_ylabel("distance (m)")
                axes[i].set_xlim([self.t[0], self.t[-1]])
            else:
                axes[i].plot(self.t, self.q[i, :] * 180/np.pi, color="red")
                axes[i].set_ylim([q_min[i] * 180/np.pi, q_max[i] * 180/np.pi])
                axes[i].set_ylabel("rotation (degrees)")
                axes[i].set_xlim([self.t[0], self.t[-1]])
        axes[4].set_xlabel("time (s)")

        # --- plot leg Dofs --- #
        figure, axes = plt.subplots(2, 3, sharex=True)
        figure.suptitle('Q legs')
        leg_label = ["Hip Abduction/Adduction",
                        "Hip Rotation interne/externe",
                        "Hip Flexion/Extension",
                        "Knee Flexion/Extension",
                        "Ankle Inversion/Eversion",
                        "Ankle Flexion/Extension"]
        axes=axes.flatten()
        for i in range(6):
            axes[i].set_title(leg_label[i])
            axes[i].plot(self.t, self.q[i + 6, :] * 180/np.pi, color="red")
            axes[i].plot(self.t, self.q[i + 12, :] * 180 / np.pi, color="blue")
            axes[i].set_ylim([q_min[i + 6] * 180/np.pi, q_max[i + 6] * 180/np.pi])
            axes[i].set_xlim([self.t[0], self.t[-1]])
            axes[i].set_ylabel("rotation (degrees)")
        axes[4].set_xlabel("time (s)")
        axes[-1].legend(["Right", "Left"])


    def plot_qdot_symetry(self):
        qdot_max, qdot_min = self.get_qdot_range()
        # --- plot pelvis --- #
        figure, axes = plt.subplots(2, 3, sharex=True)
        figure.suptitle('Qdot pelvis')
        pelvis_label = ["Pelvis translation X",
                        "Pelvis translation Y",
                        "Pelvis translation Z",
                        "Pelvis Rotation",
                        "Pelvis Elevation/Depression Right",
                        "Pelvis Anterior/Posterior"]
        axes=axes.flatten()
        for i in range(6):
            axes[i].set_title(pelvis_label[i])
            if (i < 3):
                axes[i].plot(self.t, self.q_dot[i, :], color="red")
                axes[i].set_ylabel("speed (m/s)")
                axes[i].set_ylim([qdot_min[i], qdot_max[i]])
                axes[i].set_xlim([self.t[0], self.t[-1]])
            else:
                axes[i].plot(self.t, self.q_dot[i, :] * 180 / np.pi, color="red")
                axes[i].set_ylabel("rotation speed (degrees/s)")
                axes[i].set_ylim([qdot_min[i] * 180 / np.pi, qdot_max[i] * 180 / np.pi])
                axes[i].set_xlim([self.t[0], self.t[-1]])
        axes[4].set_xlabel("time (s)")

        # --- plot leg Dofs --- #
        figure, axes = plt.subplots(2, 3, sharex=True)
        figure.suptitle('Qdot legs')
        leg_label = ["Hip Abduction/Adduction",
                     "Hip Rotation interne/externe",
                     "Hip Flexion/Extension",
                     "Knee Flexion/Extension",
                     "Ankle Inversion/Eversion",
                     "Ankle Flexion/Extension"]
        axes=axes.flatten()
        for i in range(6):
            axes[i].set_title(leg_label[i])
            axes[i].plot(self.t, self.q_dot[i + 6, :] * 180 / np.pi, color="red")
            axes[i].plot(self.t, self.q_dot[i + 12, :] * 180 / np.pi, color="blue")
            axes[i].set_ylabel("rotation speed (degrees/s)")
            axes[i].set_ylim([qdot_min[i + 6] * 180 / np.pi, qdot_max[i + 6] * 180 / np.pi])
            axes[i].set_xlim([self.t[0], self.t[-1]])
        axes[4].set_xlabel("time (s)")
        axes[-1].legend(["Right", "Left"])


    def plot_tau_symetry(self):
        # --- plot pelvis --- #
        figure, axes = plt.subplots(2, 3, sharex=True)
        figure.suptitle('Residual torque pelvis')
        pelvis_label = ["Pelvis translation X",
                        "Pelvis translation Y",
                        "Pelvis translation Z",
                        "Pelvis Rotation",
                        "Pelvis Elevation/Depression Right",
                        "Pelvis Anterior/Posterior"]
        axes=axes.flatten()
        for i in range(6):
            axes[i].set_title(pelvis_label[i])
            if (i<3):
                axes[i].plot(self.t, self.tau[i, :], color="red")
                axes[i].set_ylabel("Force (N)")
                axes[i].set_xlim([self.t[0], self.t[-1]])
            else:
                axes[i].plot(self.t, self.tau[i, :], color="red")
                axes[i].set_ylabel("Torque (N.m)")
                axes[i].set_xlim([self.t[0], self.t[-1]])
        axes[4].set_xlabel("time (s)")

        # --- plot leg Dofs --- #
        figure, axes = plt.subplots(2, 3, sharex=True)
        figure.suptitle('Residual torque legs')
        leg_label = ["Hip Abduction/Adduction",
                        "Hip Rotation interne/externe",
                        "Hip Flexion/Extension",
                        "Knee Flexion/Extension",
                        "Ankle Inversion/Eversion",
                        "Ankle Flexion/Extension"]
        axes=axes.flatten()
        for i in range(6):
            axes[i].set_title(leg_label[i])
            axes[i].plot(self.t, self.tau[i + 6, :], color="red")
            axes[i].plot(self.t, self.tau[i + 12, :], color="blue")
            axes[i].set_xlim([self.t[0], self.t[-1]])
            axes[i].set_ylabel("Torque (N.m)")
        axes[4].set_xlabel("time (s)")
        axes[-1].legend(["Right", "Left"])

    def plot_muscles_symetry(self):
        figure, axes = plt.subplots(4, 5, sharex=True, sharey=True)
        figure.suptitle("Muscle activity")
        axes = axes.flatten()
        for i in range(len(axes) - 1):
            axes[i].set_title(self.ocp.nlp[0].model.muscle(i).name().to_string())
            axes[i].plot(self.t, self.activations[i, :], color="red")
            axes[i].plot(self.t, self.activations[i + 19, :], color="blue")
            axes[i].set_xlim([self.t[0], self.t[-1]])
            axes[i].set_ylim([0.0, 1.0])
        axes[-2].legend(["Right", "Left"])
        for i in range(4):
            axes[i*5].set_ylabel("activation")
            axes[15 + i].set_xlabel("time (s)")