from casadi import DM
import numpy as np
from matplotlib import pyplot as plt

class Plot_results():
    def __init__(self, sol, params):
        self.params = params
        self.U = sol[:params.nbU * params.nbNoeuds]
        self.X = sol[params.nbU * params.nbNoeuds: -params.nP]
        self.P = sol[-params.nP:]
        self.t = np.linspace(0, params.T, params.nbNoeuds + 1)

    def plot_q(self, q_kalman):
        q, dq = self.get_states()
        Labels = ['Pelvis_Trans_X', 'Pelvis_Trans_Y', 'Pelvis_Rot_Z', 'R_Hip_Rot_Z', 'R_Knee_Rot_Z', 'R_Ankle_Rot_Z']
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        axes = axes.flatten()

        for i in range(self.params.nbQ):
            if i > 1:
                q[i, :] = q[i, :] * 180 / np.pi
                q_kalman[i, :] = q_kalman[i, :] * 180 / np.pi
                axes[i].set_ylabel('angle (deg)')
            else:
                axes[i].set_ylabel('position (m)')

            axes[i].set_title(Labels[i])
            axes[i].plot(self.t, q[i, :], 'b+', label='simu')
            axes[i].plot(self.t, q_kalman[i, :], 'r+', label='kalman')
            axes[i].plot([self.params.T_stance, self.params.T_stance], [min(q[i, :]), max(q[i, :])], 'k:')
            axes[i].set_xlabel('time (s)')
            axes[i].set_legend()

    def plot_dq(self):
        q, dq = self.get_states()
        Labels = ['Pelvis_Trans_X', 'Pelvis_Trans_Y', 'Pelvis_Rot_Z', 'R_Hip_Rot_Z', 'R_Knee_Rot_Z', 'R_Ankle_Rot_Z']
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        axes = axes.flatten()

        for i in range(self.params.nbQ):
            if i > 1:
                axes[i].set_ylabel('angular speed (rad/s)')
            else:
                axes[i].set_ylabel('speed (m/s)')

            axes[i].set_title(Labels[i])
            axes[i].plot(self.t, dq[i, :], '+')
            axes[i].plot([self.params.T_stance, self.params.T_stance], [min(dq[i, :]), max(dq[i, :])], 'k:')
            axes[i].set_xlabel('time (s)')

    def plot_u(self, U_real):
        activation = self.get_activation()
        torque = self.get_torque()

        fig, axes = plt.subplots(6, 4, figsize=(10, 10))
        Labels = ['GLUT_MAX1', 'GLUT_MAX2', 'GLUT_MAX3', 'GLUT_MED1', 'GLUT_MED2', 'GLUT_MED3',
                  'R_SEMIMEM', 'R_SEMITEN', 'R_BI_FEM_LH', 'R_RECTUS_FEM', 'R_VAS_MED', 'R_VAS_INT',
                  'R_VAS_LAT', 'R_GAS_MED', 'R_GAS_LAT', 'R_SOLEUS', 'R_TIB_ANT', 'Pelvis Tx', 'Pelvis Ty', 'Pelvis Rz',
                  'Hip Rz', 'Knee Rz', 'Ankle Rz']
        axes = axes.flatten()
        u_emg = 9

        for i in range(self.params.nbMus):
            axes[i].set_title(Labels[i])
            self.plot_control(axes[i], self.t[:-1], activation[i, :])
            axes[i].plot([self.params.T_stance, self.params.T_stance], [0, 1], 'k:')
            axes[i].set_xlabel('time (s)')
            axes[i].set_ylim(0, 1)
            axes[i].grid(True)
            if (i != 1) and (i != 2) and (i != 3) and (i != 5) and (i != 6) and (i != 11) and (i != 12):
                axes[i].plot(self.t[:-1], U_real[u_emg, :], 'r')
                u_emg -= 1
        for i in range(self.params.nbQ):
            axes[self.params.nbMus + i].set_title(Labels[self.params.nbMus + i])
            self.plot_control(axes[self.params.nbMus + i], self.t[:-1], torque[i, :])
            axes[self.params.nbMus + i].plot([self.params.T_stance, self.params.T_stance], [min(torque[i, :]), max(torque[i, :])], 'k:')
            axes[self.params.nbMus + i].set_xlabel('time (s)')
            axes[self.params.nbMus + i].grid(True)


    def plot_control(ax, t, x):
        nbPoints = len(np.array(x))
        for n in range(nbPoints - 1):
            ax.plot([t[n], t[n + 1], t[n + 1]], [x[n], x[n], x[n + 1]], 'b')


    def get_states(self):
        q = DM(self.params.nbQ, self.params.nbNoeuds + 1)
        dq = DM(self.params.nbQ, self.params.nbNoeuds + 1)
        for i in range(self.params.nbQ):
            q[i, :] = self.X[i::self.params.nbX]
            q[i, :] = self.X[self.params.nbQ + i::self.params.nbX]
        return q, dq

    def get_torque(self):
        tau = DM(self.params.nbQ, self.params.nbNoeuds)
        for i in range(self.params.nbQ):
            tau[i, :] = self.U[self.params.nbMus + i::self.params.nbU]
        return tau

    def get_activation(self):
        activation = DM(self.params.nbMus, self.params.nbNoeuds)
        for i in range(self.params.nbMus):
            activation[i, :] = self.U[i::self.params.nbU]
        return activation
