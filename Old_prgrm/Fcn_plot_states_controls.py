from matplotlib import pyplot as plt
import numpy as np
from Read_Muscod import Muscod


# plot control
def plot_control(ax, t, x, color='b'):
    nbPoints = len(np.array(x))
    for n in range(nbPoints - 1):
        ax.plot([t[n], t[n + 1], t[n + 1]], [x[n], x[n], x[n + 1]], color)

def plot_q(q, T, nbNoeuds, gait = False, impact = False, params = 0):
    # JOINT POSITIONS
    fig, axes = plt.subplots(2, 3, figsize=(10, 5))
    axes = axes.flatten()
    Labels = ['Pelvis_Trans_X', 'Pelvis_Trans_Y', 'Pelvis_Rot_Z', 'R_Hip_Rot_Z', 'R_Knee_Rot_Z', 'R_Ankle_Rot_Z']
    if gait:
        t_stance = np.linspace(0, params.T_stance, params.nbNoeuds_stance + 1)
        t_swing = params.T_stance + np.linspace(0, params.T_swing, params.nbNoeuds_swing + 1)
        t = np.hstack([t_stance[:-1], t_swing])
        if impact:
            t = np.hstack([t, t[-1]])
    else:
        t = np.linspace(0, T, nbNoeuds + 1)

    for i in range(len(q[:, 0])):
        axes[i].set_title(Labels[i])
        axes[i].set_xlabel('time (s)')
        axes[i].grid = True
        if (i < 2):
            axes[i].plot(t, q[i, :], '+')
            axes[i].set_ylabel('position (m)')
        else:
            q[i, :] = q[i, :] * 180 / np.pi
            axes[i].plot(t, q[i, :], '+')
            axes[i].set_ylabel('angle (deg)')
        if gait:
            axes[i].plot([params.T_stance, params.T_stance], [min(q[i, :]), max(q[i, :])], 'k:')
    plt.show(block=False)

def plot_q_muscod(q, params, muscod, Gaitphase = 'gait', impact = False):
    # JOINT POSITIONS
    fig, axes = plt.subplots(2, 3, figsize=(10, 5))
    axes = axes.flatten()
    Labels = ['Pelvis_Trans_X', 'Pelvis_Trans_Y', 'Pelvis_Rot_Z', 'R_Hip_Rot_Z', 'R_Knee_Rot_Z', 'R_Ankle_Rot_Z']
    if Gaitphase == 'gait':
        t_stance = np.linspace(0, params.T_stance, params.nbNoeuds_stance + 1)
        t_swing = params.T_stance + np.linspace(0, params.T_swing, params.nbNoeuds_swing + 1)
        t = np.hstack([t_stance[:-1], t_swing])
        t_muscod = muscod.t
        q_muscod = muscod.X_muscod[:muscod.nbQ, :-1]
        if impact:
            t = np.hstack([t, t[-1]])
            t_muscod = np.hstack([muscod.t, muscod.t[-1]])
            q_muscod = muscod.X_muscod[:muscod.nbQ, :]
    elif Gaitphase == 'stance':
        t = np.linspace(0, params.T_stance, params.nbNoeuds_stance + 1)
        t_muscod = muscod.t_stance
        q_muscod = muscod.X_muscod[:muscod.nbQ, :muscod.nbNoeuds_stance + 1]
    else:
        t = np.linspace(0, params.T_swing, params.nbNoeuds_swing + 1)
        t_muscod = muscod.t_swing
        q_muscod = muscod.X_muscod[:muscod.nbQ, muscod.nbNoeuds_stance:-1]

    for i in range(len(q[:, 0])):
        axes[i].set_title(Labels[i])
        axes[i].set_xlabel('time (s)')
        axes[i].grid = True
        if (i < 2):
            axes[i].plot(t, q[i, :], 'b+', Label='simu')
            axes[i].plot(t_muscod, q_muscod[i, :], 'g+', Label='muscod')
            axes[i].set_ylabel('position (m)')
        else:
            q[i, :] = q[i, :] * 180 / np.pi
            q_muscod[i, :] = q_muscod[i, :] * 180 / np.pi
            axes[i].plot(t, q[i, :], 'b+',  Label='simu')
            axes[i].plot(t_muscod, q_muscod[i, :], 'g+', Label='muscod')
            axes[i].set_ylabel('angle (deg)')
        if Gaitphase == 'gait':
            axes[i].plot([params.T_stance, params.T_stance], [min(q[i, :]), max(q[i, :])], 'k:')
        axes[i].legend()
    plt.show(block=False)


def plot_q_int(q, q_int, T, nbNoeuds):
    # JOINT POSITIONS
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.flatten()
    dt = T/nbNoeuds
    dt2 = dt/5
    t = np.linspace(0, T + dt, nbNoeuds + 1)
    t2 = np.linspace(0, dt, 5)

    def plot_int(q_int, ax):
        for n in range(nbNoeuds):
            t_init = t[n]
            t_int = t_init + t2
            ax.plot(t_int, q_int[:, n], 'g-')

    axes[0].set_title('Pelvis_Trans_X')
    axes[0].plot(t, q[0, :], '+')
    plot_int(q_int[0, :, :], axes[0])
    axes[0].plot([T, T], [min(q[0, :]), max(q[0, :])], 'k:')
    axes[0].set_xlabel('time (s)')
    axes[0].set_ylabel('position (m)')

    axes[1].set_title('Pelvis_Trans_Y')
    axes[1].plot([T, T], [min(q[1, :]), max(q[1, :])], 'k:')
    axes[1].plot(t, q[1, :], '+')
    plot_int(q_int[1, :, :], axes[1])
    axes[1].set_xlabel('time (s)')
    axes[1].set_ylabel('position (m)')

    axes[2].set_title('Pelvis_Rot_Z')
    axes[2].plot(t, q[2, :]*180/np.pi, '+')
    plot_int(q_int[2, :, :]*180/np.pi, axes[2])
    axes[2].plot([T, T], [min(q[2, :]*180/np.pi), max(q[2, :])*180/np.pi], 'k:')
    axes[2].set_xlabel('time (s)')
    axes[2].set_ylabel('angle (deg)')

    axes[3].set_title('R_Hip_Rot_Z')
    axes[3].plot(t, q[3, :]*180/np.pi, '+')
    plot_int(q_int[3, :, :] * 180 / np.pi, axes[3])
    axes[3].plot([T, T], [min(q[3, :] * 180 / np.pi), max(q[3, :]) * 180 / np.pi], 'k:')
    axes[3].set_xlabel('time (s)')
    axes[3].set_ylabel('angle (deg)')

    axes[4].set_title('R_Knee_Rot_Z')
    axes[4].plot(t, q[4, :]*180/np.pi, '+')
    plot_int(q_int[4, :, :] * 180 / np.pi, axes[4])
    axes[4].plot([T, T], [min(q[4, :] * 180 / np.pi), max(q[4, :]) * 180 / np.pi], 'k:')
    axes[4].set_xlabel('time (s)')
    axes[4].set_ylabel('angle (deg)')

    axes[5].set_title('R_Ankle_Rot_Z')
    axes[5].plot(t, q[5, :]*180/np.pi, '+')
    plot_int(q_int[5, :, :] * 180 / np.pi, axes[5])
    axes[5].plot([T, T], [min(q[5, :] * 180 / np.pi), max(q[5, :]) * 180 / np.pi], 'k:')
    axes[5].set_xlabel('time (s)')
    axes[5].set_ylabel('angle (deg)')

    plt.show(block=False)



def plot_dq(dq, T, nbNoeuds, gait = False, impact = False, params = 0):
    # JOINT VELOCITIES
    fig, axes = plt.subplots(2, 3, figsize=(10, 5))
    axes = axes.flatten()
    Labels = ['Pelvis_Trans_X', 'Pelvis_Trans_Y', 'Pelvis_Rot_Z', 'R_Hip_Rot_Z', 'R_Knee_Rot_Z', 'R_Ankle_Rot_Z']

    if gait:
        t_stance = np.linspace(0, params.T_stance, params.nbNoeuds_stance + 1)
        t_swing = params.T_stance + np.linspace(0, params.T_swing, params.nbNoeuds_swing + 1)
        t = np.hstack([t_stance[:-1], t_swing])
        if impact:
            t = np.hstack([t, t[-1]])
    else:
        t = np.linspace(0, T, nbNoeuds + 1)

    for i in range(len(dq[:, 0])):
        axes[i].set_title(Labels[i])
        axes[i].set_xlabel('time (s)')
        axes[i].grid = True
        if (i < 2):
            axes[i].plot(t, dq[i, :], '+')
            axes[i].set_ylabel('spend (m/s)')
        else:
            dq[i, :] = dq[i, :] * 180 / np.pi
            axes[i].plot(t, dq[i, :], '+')
            axes[i].set_ylabel('angular speed (deg/s)')
        if gait:
            axes[i].plot([params.T_stance, params.T_stance], [min(dq[i, :]), max(dq[i, :])], 'k:')
    plt.show(block=False)

def plot_torque(torque, T, nbNoeuds, gait = False, impact = False, params = 0):
    # JOINT TORQUES
    fig, axes = plt.subplots(2, 3, sharex=True, figsize=(10, 5))
    axes = axes.flatten()
    Labels = ['Pelvis_Trans_X', 'Pelvis_Trans_Y', 'Pelvis_Rot_Z', 'R_Hip_Rot_Z', 'R_Knee_Rot_Z', 'R_Ankle_Rot_Z']

    # Set time
    if gait:
        t_stance = np.linspace(0, params.T_stance, params.nbNoeuds_stance + 1)
        t_swing = params.T_stance + np.linspace(0, params.T_swing, params.nbNoeuds_swing + 1)
        t = np.hstack([t_stance[:-1], t_swing[:-1]])
        if impact:
            t = np.hstack([t, t[-1]])
    else:
        dt = T/nbNoeuds
        t = np.linspace(0, T - dt, nbNoeuds)

    for i in range(len(torque[:, 0])):
        axes[i].set_title(Labels[i])
        plot_control(axes[i], t, torque[i, :])
        axes[i].set_ylabel('Force (N)')
        if gait:
            axes[i].plot([params.T_stance, params.T_stance], [np.min([-25, round(np.min(torque[i, :]))]), np.max([round(np.max(torque[i, :])), 25])], 'k:')
        if (i > 2):
            axes[i].set_xlabel('time (s)')
            axes[i].set_ylim([-25, 25])
    plt.show(block=False)



def plot_markers_heatmap(diff_M):
    nbNoeuds = len(diff_M[2, 0, :])

    Labels_M = ["L_IAS", "L_IPS", "R_IPS", "R_IAS", "R_FTC",
                "R_Thigh_Top", "R_Thigh_Down", "R_Thigh_Front", "R_Thigh_Back", "R_FLE", "R_FME",
                "R_FAX", "R_TTC", "R_Shank_Top", "R_Shank_Down", "R_Shank_Front", "R_Shank_Tibia", "R_FAL", "R_TAM",
                "R_FCC", "R_FM1", "R_FMP1", "R_FM2", "R_FMP2", "R_FM5", "R_FMP5"]
    node = np.linspace(0, nbNoeuds, nbNoeuds, dtype=int)

    fig4, ax = plt.subplots()
    im = ax.imshow(diff_M[2, :, :])

    # Create labels
    ax.set_xticks(np.arange(len(node)))
    ax.set_yticks(np.arange(len(Labels_M)))
    ax.set_xticklabels(node)
    ax.set_yticklabels(Labels_M)
    ax.set_title('Markers differences')

    # Create grid
    ax.set_xticks(np.arange(diff_M[0, :, :].shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(diff_M[0, :, :].shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="k", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('squared differences', rotation=-90, va="bottom")

def plot_emg_heatmap(diff_U):
    nbNoeuds   = len(diff_U[0, :])
    Labels_emg = ['GLUT_MAX1', 'GLUT_MED2', 'R_SEMITEN', 'R_BI_FEM_LH', 'R_RECTUS_FEM', 'R_VAS_MED', 'R_GAS_MED', 'R_GAS_LAT', 'R_SOLEUS', 'R_TIB_ANT']
    node       = np.linspace(0, nbNoeuds, nbNoeuds, dtype=int)

    fig, ax = plt.subplots()
    im_emg = ax.imshow(diff_U)

    # Create labels
    ax.set_xticks(np.arange(len(node)))
    ax.set_yticks(np.arange(len(Labels_emg)))
    ax.set_xticklabels(node)
    ax.set_yticklabels(Labels_emg)
    ax.set_title('Muscular activations differences')

    # Create grid
    ax.set_xticks(np.arange(diff_U[:, :].shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(diff_U[:, :].shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="k", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Create colorbar
    cbar = ax.figure.colorbar(im_emg, ax=ax)
    cbar.ax.set_ylabel('squared differences ', rotation=-90, va="bottom")


def plot_activation(params, u, U_real, T, nbNoeuds, gait = False, impact = False):
    # Muscular activation
    nbMus = params.nbMus
    fig1, axes1 = plt.subplots(6, 3, sharex=True, sharey=True, figsize=(10, 10))
    Labels = ['GLUT_MAX1', 'GLUT_MAX2', 'GLUT_MAX3', 'GLUT_MED1', 'GLUT_MED2', 'GLUT_MED3',
              'R_SEMIMEM', 'R_SEMITEN', 'R_BI_FEM_LH', 'R_RECTUS_FEM', 'R_VAS_MED', 'R_VAS_INT',
              'R_VAS_LAT', 'R_GAS_MED', 'R_GAS_LAT', 'R_SOLEUS', 'R_TIB_ANT']
    axes1 = axes1.flatten()

    # Set time
    if gait:
        t_stance = np.linspace(0, params.T_stance, params.nbNoeuds_stance + 1)
        t_swing = params.T_stance + np.linspace(0, params.T_swing, params.nbNoeuds_swing + 1)
        t = np.hstack([t_stance[:-1], t_swing[:-1]])
        if impact:
            t = np.hstack([t, t[-1]])
    else:
        dt = T/nbNoeuds
        t = np.linspace(0, T - dt, nbNoeuds)

    u_emg = 9
    for i in range(nbMus):
        ax = axes1[i]
        ax.set_title(Labels[i])
        plot_control(ax, t, u[i, :])
        ax.grid(True)
        ax.plot([0, t[-1]], [0, 0], 'k--')  # lower bound
        ax.plot([0, t[-1]], [1, 1], 'k--')  # upper bound
        ax.yaxis.set_ticks(np.arange(0, 1.2, 0.2))
        if i > (nbMus - 3):
            ax.set_xlabel('time (s)')
        if (i == 0) or (i == 3) or (i == 6) or (i == 9) and (i == 12):
            ax.set_ylabel('activation ')
        if (i != 1) and (i != 2) and (i != 3) and (i != 5) and (i != 6) and (i != 11) and (i != 12):
            ax.plot(t, U_real[u_emg, :], 'r')
            u_emg -= 1
        if gait:
            ax.plot([params.T_stance, params.T_stance],[0, 1], 'k:')
    plt.show(block=False)

def plot_activation_muscod(params, u, U_real, muscod, Gaitphase = 'gait', impact = False):
    # Muscular activation
    nbMus = params.nbMus
    fig1, axes1 = plt.subplots(6, 3, sharex=True, sharey=True, figsize=(10, 10))
    Labels = ['GLUT_MAX1', 'GLUT_MAX2', 'GLUT_MAX3', 'GLUT_MED1', 'GLUT_MED2', 'GLUT_MED3',
              'R_SEMIMEM', 'R_SEMITEN', 'R_BI_FEM_LH', 'R_RECTUS_FEM', 'R_VAS_MED', 'R_VAS_INT',
              'R_VAS_LAT', 'R_GAS_MED', 'R_GAS_LAT', 'R_SOLEUS', 'R_TIB_ANT']
    axes1 = axes1.flatten()
    if Gaitphase == 'gait':
        t_stance = np.linspace(0, params.T_stance, params.nbNoeuds_stance + 1)
        t_swing = params.T_stance + np.linspace(0, params.T_swing, params.nbNoeuds_swing + 1)
        t = np.hstack([t_stance[:-1], t_swing])
        t_muscod = muscod.t[:-1]
        activation_muscod = muscod.X_muscod[2*muscod.nbQ:, :-2]
        if impact:
            t = np.hstack([t, t[-1]])
            t_muscod = np.hstack([t_muscod, t_muscod[-1]])
            activation_muscod = muscod.X_muscod[2*muscod.nbQ:, :-1]
    elif Gaitphase == 'stance':
        t = np.linspace(0, params.T_stance, params.nbNoeuds_stance + 1)
        t_muscod = muscod.t_stance[:-1]
        activation_muscod = muscod.X_muscod[2*muscod.nbQ:, :muscod.nbNoeuds_stance]
    else:
        t = np.linspace(0, params.T_swing, params.nbNoeuds_swing + 1)
        t_muscod = muscod.t_swing[:-1]
        activation_muscod = muscod.X_muscod[2*muscod.nbQ:, muscod.nbNoeuds_stance:-2]

    u_emg = 9
    for i in range(nbMus):
        ax = axes1[i]
        ax.set_title(Labels[i])
        plot_control(ax, t[:-1], u[i, :])
        plot_control(ax, t_muscod, activation_muscod[i, :], color='g')
        ax.grid(True)
        ax.plot([0, t[-1]], [0, 0], 'k--')  # lower bound
        ax.plot([0, t[-1]], [1, 1], 'k--')  # upper bound
        ax.yaxis.set_ticks(np.arange(0, 1.2, 0.2))
        if i > (nbMus - 3):
            ax.set_xlabel('time (s)')
        if (i == 0) or (i == 3) or (i == 6) or (i == 9) and (i == 12):
            ax.set_ylabel('activation ')
        if (i != 1) and (i != 2) and (i != 3) and (i != 5) and (i != 6) and (i != 11) and (i != 12):
            ax.plot(t[:-1], U_real[u_emg, :], 'r')
            u_emg -= 1
        if Gaitphase == 'gait':
            ax.plot([params.T_stance, params.T_stance],[0, 1], 'k:')
    plt.show(block=False)


def plot_GRF(GRF, GRF_real, T, nbNoeuds):
    # Set time
    t = np.linspace(0, T, nbNoeuds + 1)

    fig, axes = plt.subplots(2, 1, sharex=True)
    axes.flatten()

    ax_ap = axes[0]
    ax_ap.set_title('GRF A/P  during the gait')
    ax_ap.plot(t, GRF_real[1, :nbNoeuds + 1], 'r')
    ax_ap.plot([T, T], [min(GRF_real[1, :]), max(GRF_real[1, :])], 'k:')  # end of the stance phase
    ax_ap.plot(t, GRF[0, :], 'b', alpha = 0.5)
    ax_ap.grid(True)

    ax_v = axes[1]
    ax_v.set_title('GRF vertical')
    ax_v.plot(t, GRF_real[2, :nbNoeuds + 1], 'r')
    ax_v.plot([T, T], [min(GRF_real[2, :]), max(GRF_real[2, :])], 'k:')
    ax_v.plot(t, GRF[1, :], 'b', alpha = 0.5)
    ax_v.set_xlabel('time (s)')
    ax_v.grid(True)
    fig.tight_layout()
    plt.show(block=False)

def plot_markers(nbNoeuds, M, M_real, COM):
    # PLOT ARTIFICIAL SEGMENTS TO FOLLOW LEG MOVEMENT
    plt.figure()
    for k_stance in range(nbNoeuds):
        Mk = M[k_stance]
        CoM = COM[k_stance]
        M_aff = np.zeros((3, 5))
        M_aff[:, 0] = np.array(Mk[:, 2]).squeeze()
        M_aff[:, 1] = np.array(Mk[:, 4]).squeeze()
        M_aff[:, 2] = np.array(Mk[:, 11]).squeeze()
        M_aff[:, 3] = np.array(Mk[:, 19]).squeeze()
        M_aff[:, 4] = np.array(Mk[:, 22]).squeeze()
        plt.plot(M_aff[0, :], M_aff[2, :], 'bo-', alpha=0.5)
        # plt.plot(np.array(Mk[0, -1]), np.array(Mk[2, -1]), 'g+',)
        plt.plot(np.array(CoM[0]), np.array(CoM[2]), 'g+', markersize=10)
        plt.plot([M_real[0, 2, k_stance], M_real[0, 4, k_stance], M_real[0, 11, k_stance], M_real[0, 19, k_stance],
                  M_real[0, 22, k_stance]],
                 [M_real[2, 2, k_stance], M_real[2, 4, k_stance], M_real[2, 11, k_stance], M_real[2, 19, k_stance],
                  M_real[2, 22, k_stance]], 'r+')
    plt.show(block = False)
