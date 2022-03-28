import matplotlib.pyplot as plt
import numpy as np

def get_colors():
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    return colors


class plot_function:
    def __init__(self):
        self.colors = get_colors()
        self.conditions = ['squat controle', 'squat 3cm', 'squat 4cm', 'squat 5cm']

    def plot_initial_kinematic(self, kin_test, q_idx, condition_idx, title):
        fig, axes = plt.subplots(2, 5)
        axes = axes.flatten()
        fig.suptitle(title)
        for (i, kin) in enumerate(kin_test):
            axes[i].plot(kin.q_kalman[condition_idx][q_idx, :] * 180 / np.pi, color=self.colors[i])
            axes[i].plot([0, kin.q_kalman[condition_idx].shape[1]], [0, 0], 'k--')
            if q_idx == 12 or q_idx==18:
                axes[i].plot([0, kin.q_kalman[condition_idx].shape[1]], [-90, -90], 'k--')
            else:
                axes[i].plot([0, kin.q_kalman[condition_idx].shape[1]], [90, 90], 'k--')

    def plot_initial_com_displacement(self, kin_test, condition_idx, title, norm=None):
        fig, axes = plt.subplots(2, 5)
        axes = axes.flatten()
        fig.suptitle(title)
        for (i, kin) in enumerate(kin_test):
            if norm is None:
                axes[i].plot(kin.com_position[condition_idx][2, :], color=self.colors[i])
            else:
                axes[i].plot(kin.com_position[condition_idx][2, :] / norm[i], color=self.colors[i])
            axes[i].plot([0, kin.com_position[condition_idx].shape[1]], [1.0, 1.0], 'k--')
            axes[i].plot([0, kin.com_position[condition_idx].shape[1]], [0.65, 0.65], 'k--')

    def plot_condition(self, diff_control, diff_perturbation, title, conditions_idx=None):
        cond_ticks = np.linspace(1, len(diff_perturbation)+1, len(diff_perturbation)+1) if conditions_idx is None else [i+1 for i in conditions_idx]
        diff_perturbation.insert(0, diff_control)
        cond_data = np.array(diff_perturbation)

        plt.figure()
        plt.title(title)
        for i in range(len(diff_control)):
            plt.plot(cond_ticks, cond_data[:, i], 'o-', color=self.colors[i])
        plt.xticks([1, 2, 3, 4], self.conditions)
        plt.ylabel('angle difference between legs (deg)')

    def plot_pelvis_markers(self, mark_test):
        fig = plt.figure()
        fig.suptitle('pelvis markers')
        ax = plt.axes(projection='3d')
        for (i, mark) in enumerate(mark_test):
            ax.plot3D(mark.mean_markers_position[0][0, [0, 2, 3, 5, 0], 0],
                      mark.mean_markers_position[0][1, [0, 2, 3, 5, 0], 0],
                      mark.mean_markers_position[0][2, [0, 2, 3, 5, 0], 0],
                      'o-', c=self.colors[i])

    def plot_anova(self, Firm, y, condition, condition_name, title='anova'):
        x = np.linspace(0, 100, 200)
        plt.figure(figsize=(8, 3.5))
        plt.title(title)
        ax0 = plt.axes((0.1, 0.15, 0.35, 0.8))
        ax1 = plt.axes((0.55, 0.15, 0.35, 0.8))
        # plot mean subject trajectories:
        for i in range(len(condition_name)):
            ax0.plot(x, y[condition == i].T, c=self.colors[i])
        ax0.set_xlim([0, 100])
        ax0.set_xlabel('Time (%)')
        ax0.legend(condition_name)

        # plot SPM results:
        Firm.plot(ax=ax1, color='r', facecolor=(0.8, 0.3, 0.3))
        plt.show()