import numpy as np
import seaborn
from matplotlib import pyplot as plt
from .Utils_start import utils
from .Contact_Forces import contact
from .Muscles import muscle

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

        # params model
        self.model=ocp.nlp[0].model
        self.q_name=utils.get_q_name(self.model)
        self.q_min, self.q_max = utils.get_q_range(self.model)
        self.qdot_min, self.qdot_max = utils.get_qdot_range(self.model)
        self.contact_name = utils.get_contact_name(self.model)

        # contact
        self.contact_data = contact(self.ocp, self.sol, self.muscles)

        # muscle params
        self.muscle_params = muscle(self.ocp, self.sol)

    def plot_individual_forces(self):
        forces = self.contact_data.individual_forces
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

    def plot_sum_forces(self):
        figure, axes = plt.subplots(1, 3, sharey=True)
        figure.suptitle('Contact forces')

        # --- plot x --- #
        axes[0, 0].set_title("forces X")
        axes[0, 0].plot(self.t, self.contact_data.forces["forces_r_X"], color="red")
        axes[0, 0].plot(self.t, self.contact_data.forces["forces_l_X"], color="blue")
        axes[0, 0].set_xlim([self.t[0], self.t[-1]])
        axes[0, 0].set_ylabel("Forces (N)")
        axes[0, 0].set_xlabel("Time (s)")

        # --- plot y --- #
        axes[0, 1].set_title("forces Y")
        axes[0, 1].plot(self.t, self.contact_data.forces["forces_r_Y"], color="red")
        axes[0, 1].plot(self.t, self.contact_data.forces["forces_l_Y"], color="blue")
        axes[0, 1].set_xlim([self.t[0], self.t[-1]])
        axes[0, 1].set_ylabel("Forces (N)")
        axes[0, 1].set_xlabel("Time (s)")

        # --- plot z --- #
        axes[0, 2].set_title("forces Z")
        axes[0, 2].plot(self.t, self.contact_data.forces["forces_r_Z"], color="red")
        axes[0, 2].plot(self.t, self.contact_data.forces["forces_l_Z"], color="blue")
        axes[0, 2].set_xlim([self.t[0], self.t[-1]])
        axes[0, 2].set_ylabel("Forces (N)")
        axes[0, 2].set_xlabel("Time (s)")
        axes[0, 2].legend(["Right", "Left"])

    def plot_cop(self):
        figure=plt.figure()
        figure.suptitle('Center of pressure')

        # --- pied droit --- #
        plt.scatter(self.contact_data.position["Heel_r"][0, :], self.contact_data.position["Heel_r"][1, :], marker='+', color="red")
        plt.scatter(self.contact_data.position["Meta_1_r"][0, :], self.contact_data.position["Meta_1_r"][1, :], marker='+',color="red")
        plt.scatter(self.contact_data.position["Meta_5_r"][0, :], self.contact_data.position["Meta_5_r"][1, :], marker='+', color="red")
        plt.scatter(self.contact_data.cop["cop_r_X"], self.contact_data.cop["cop_r_Y"], marker='o', color="red")

        # --- pied gauche --- #
        plt.scatter(self.contact_data.position["Heel_l"][0, :], self.contact_data.position["Heel_l"][1, :], marker='+', color="blue")
        plt.scatter(self.contact_data.position["Meta_1_l"][0, :], self.contact_data.position["Meta_1_l"][1, :], marker='+', color="blue")
        plt.scatter(self.contact_data.position["Meta_5_l"][0, :], self.contact_data.position["Meta_5_l"][1, :], marker='+', color="blue")
        plt.scatter(self.contact_data.cop["cop_l_X"], self.contact_data.cop["cop_l_Y"], marker='o', color="blue")

        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.axis("equal")

    def plot_q_symetry(self):
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
                axes[i].set_ylim([self.q_min[i], self.q_max[i]])
                axes[i].set_ylabel("distance (m)")
                axes[i].set_xlim([self.t[0], self.t[-1]])
            else:
                axes[i].plot(self.t, self.q[i, :] * 180/np.pi, color="red")
                axes[i].set_ylim([self.q_min[i] * 180/np.pi, self.q_max[i] * 180/np.pi])
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
            axes[i].set_ylim([self.q_min[i + 6] * 180/np.pi, self.q_max[i + 6] * 180/np.pi])
            axes[i].set_xlim([self.t[0], self.t[-1]])
            axes[i].set_ylabel("rotation (degrees)")
        axes[4].set_xlabel("time (s)")
        axes[-1].legend(["Right", "Left"])


    def plot_qdot_symetry(self):
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
                axes[i].set_ylim([self.qdot_min[i], self.qdot_max[i]])
                axes[i].set_xlim([self.t[0], self.t[-1]])
            else:
                axes[i].plot(self.t, self.q_dot[i, :] * 180 / np.pi, color="red")
                axes[i].set_ylabel("rotation speed (degrees/s)")
                axes[i].set_ylim([self.qdot_min[i] * 180 / np.pi, self.qdot_max[i] * 180 / np.pi])
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
            axes[i].set_ylim([self.qdot_min[i + 6] * 180 / np.pi, self.qdot_max[i + 6] * 180 / np.pi])
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
            axes[i].plot(self.t, self.muscle_params.muscle_tau[i, :], color="red")
            axes[i].plot(self.t, self.tau[i, :], color="green")
            axes[i].set_xlim([self.t[0], self.t[-1]])
            if (i<3):
                axes[i].set_ylabel("Force (N)")
            else:
                axes[i].set_ylabel("Torque (N.m)")
        axes[4].set_xlabel("time (s)")
        plt.legend(["muscular torque", "residual torque"])

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
            axes[i].plot(self.t, self.muscle_params.muscle_tau[i + 6, :], color="red")
            axes[i].plot(self.t, self.tau[i + 6, :], color="red", linestyle="--")
            axes[i].plot(self.t, self.muscle_params.muscle_tau[i + 12, :], color="blue")
            axes[i].plot(self.t, self.tau[i + 12, :], color="blue", linestyle="--")
            axes[i].set_xlim([self.t[0], self.t[-1]])
            axes[i].set_ylabel("Torque (N.m)")
        axes[4].set_xlabel("time (s)")
        axes[-1].legend(["Right muscular torque", "Right residual torque", "Left muscular torque", "Left residual torque"])

    def plot_muscles_activation_symetry(self):
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

    def plot_muscles_force_symetry(self):
        figure, axes = plt.subplots(4, 5, sharex=True)
        figure.suptitle("Muscle force")
        axes = axes.flatten()
        for i in range(len(axes) - 1):
            axes[i].set_title(self.ocp.nlp[0].model.muscle(i).name().to_string())
            axes[i].plot(self.t, self.muscle_params.muscle_force[i, :], color="red")
            axes[i].plot(self.t, self.muscle_params.muscle_force[i + 19, :], color="blue")
            axes[i].plot([self.t[0], self.t[-1]], [self.muscle_params.f_iso[i], self.muscle_params.f_iso[i]], color="black", linestyle="--")
            axes[i].set_xlim([self.t[0], self.t[-1]])
            # axes[i].set_ylim([0.0, self.muscle_params.f_iso[i]])
        axes[-2].legend(["Right", "Left"])
        for i in range(4):
            axes[i*5].set_ylabel("force (N)")
            axes[15 + i].set_xlabel("time (s)")

    def plot_momentarm(self, idx_muscle):
        idx_q = np.where(np.sqrt(self.muscle_params.muscle_jacobian[idx_muscle, :, 0] ** 2) > 1e-10)[0]
        for i_q in idx_q:
           figure = plt.figure(self.q_name[i_q])
           figure.suptitle(self.model.muscle(idx_muscle).name().to_string())
           plt.plot(self.t, self.muscle_params.muscle_jacobian[idx_muscle, i_q, :], color="red")
           plt.xlabel("time (s)")
           plt.ylabel("moment arm (m)")