from matplotlib import pyplot as plt
import numpy as np
import biorbd

def plot_q(q, T, nbNoeuds):
    # JOINT POSITIONS
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.flatten()
    dt = T/nbNoeuds
    t = np.linspace(0, T, nbNoeuds + 1)

    axes[0].set_title('Pelvis_Trans_X')
    axes[0].plot(t, q[0, :], '+')
    axes[0].plot([T, T], [min(q[0, :]), max(q[0, :])], 'k:')
    axes[0].set_xlabel('time (s)')
    axes[0].set_ylabel('position (m)')

    axes[1].set_title('Pelvis_Trans_Y')
    axes[1].plot([T, T], [min(q[1, :]), max(q[1, :])], 'k:')
    axes[1].plot(t, q[1, :], '+')
    axes[1].set_xlabel('time (s)')
    axes[1].set_ylabel('position (m)')

    axes[2].set_title('Pelvis_Rot_Z')
    axes[2].plot(t, q[2, :]*180/np.pi, '+')
    axes[2].plot([T, T], [min(q[2, :]*180/np.pi), max(q[2, :])*180/np.pi], 'k:')
    axes[2].set_xlabel('time (s)')
    axes[2].set_ylabel('angle (deg)')

    axes[3].set_title('R_Hip_Rot_Z')
    axes[3].plot(t, q[3, :]*180/np.pi, '+')
    axes[3].plot([T, T], [min(q[3, :] * 180 / np.pi), max(q[3, :]) * 180 / np.pi], 'k:')
    axes[3].set_xlabel('time (s)')
    axes[3].set_ylabel('angle (deg)')

    axes[4].set_title('R_Knee_Rot_Z')
    axes[4].plot(t, q[4, :]*180/np.pi, '+')
    axes[4].plot([T, T], [min(q[4, :] * 180 / np.pi), max(q[4, :]) * 180 / np.pi], 'k:')
    axes[4].set_xlabel('time (s)')
    axes[4].set_ylabel('angle (deg)')

    axes[5].set_title('R_Ankle_Rot_Z')
    axes[5].plot(t, q[5, :]*180/np.pi, '+')
    axes[5].plot([T, T], [min(q[5, :] * 180 / np.pi), max(q[5, :]) * 180 / np.pi], 'k:')
    axes[5].set_xlabel('time (s)')
    axes[5].set_ylabel('angle (deg)')

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



def plot_dq(dq, T, nbNoeuds):
    # JOINT VELOCITIES
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.flatten()

    # Set time
    dt = T/nbNoeuds
    t = np.linspace(0, T, nbNoeuds + 1)

    axes[0].set_title('Pelvis_Trans_X')
    axes[0].plot(t, dq[0, :], '+')
    axes[0].plot([T, T], [min(dq[0, :]), max(dq[0, :])], 'k:')
    axes[0].set_xlabel('time (s)')
    axes[0].set_ylabel('speed (m/s)')

    axes[1].set_title('Pelvis_Trans_Y')
    axes[1].plot(t, dq[1, :], '+')
    axes[1].plot([T, T], [min(dq[1, :]), max(dq[1, :])], 'k:')
    axes[1].set_xlabel('time (s)')
    axes[1].set_ylabel('speed (m/s)')

    axes[2].set_title('Pelvis_Rot_Z')
    axes[2].plot(t, dq[2, :], '+')
    axes[2].plot([T, T], [min(dq[2, :]), max(dq[2, :])], 'k:')
    axes[2].set_xlabel('time (s)')
    axes[2].set_ylabel('speed (rad/s)')

    axes[3].set_title('R_Hip_Rot_Z')
    axes[3].plot(t, dq[3, :], '+')
    axes[3].plot([T, T], [min(dq[3, :]), max(dq[3, :])], 'k:')
    axes[3].set_xlabel('time (s)')
    axes[3].set_ylabel('speed (rad/s)')

    axes[4].set_title('R_Knee_Rot_Z')
    axes[4].plot(t, dq[4, :], '+')
    axes[4].plot([T, T], [min(dq[4, :]), max(dq[4, :])], 'k:')
    axes[4].set_xlabel('time (s)')
    axes[4].set_ylabel('speed (rad/s)')

    axes[5].set_title('R_Ankle_Rot_Z')
    axes[5].plot(t, dq[5, :], '+')
    axes[5].plot([T, T], [min(dq[5, :]), max(dq[5, :])], 'k:')
    axes[5].set_xlabel('time (s)')
    axes[5].set_ylabel('speed (rad/s)')

    plt.show(block=False)

def plot_torque(torque, T, nbNoeuds):
    # JOINT TORQUES

    # Set time
    dt = T/nbNoeuds
    t = np.linspace(0, T - dt, nbNoeuds)

    plt.subplot(231)
    plt.title('Pelvis_Trans_X')
    plt.plot(t, torque[0, :], 'b+-')
    # plt.plot(t, F0[0, :], 'r-', label = 'control')
    plt.plot([T, T], [min(torque[0, :]), max(torque[0, :])], 'k:')
    plt.xlabel('time (s)')
    plt.ylabel('Force (N)')

    plt.subplot(232)
    plt.title('Pelvis_Trans_Y')
    plt.plot(t, torque[1, :], 'b+-')
    # plt.plot(t, F0[1, :], 'r-', label = 'control')
    plt.plot([T, T], [min(torque[1, :]), max(torque[1, :])], 'k:')
    plt.xlabel('time (s)')
    plt.ylabel('force (N)')

    plt.subplot(233)
    plt.title('Pelvis_Rot_Z')
    plt.plot(t, torque[2, :], 'b+-')
    # plt.plot(t, F0[2, :], 'r-', label = 'control')
    plt.plot([T, T], [min(torque[2, :]), max(torque[2, :])], 'k:')
    plt.xlabel('time (s)')
    plt.ylabel('torque (Nm)')

    plt.subplot(234)
    plt.title('R_Hip_Rot_Z')
    plt.plot(t, torque[3, :], 'b+-')
    plt.plot([T, T], [min(torque[3, :]), max(torque[3, :])], 'k:')
    plt.xlabel('time (s)')
    plt.ylabel('torque (Nm)')

    plt.subplot(235)
    plt.title('R_Knee_Rot_Z')
    plt.plot(t, torque[4, :], 'b+-')
    plt.plot([T, T], [min(torque[4, :]), max(torque[4, :])], 'k:')
    plt.xlabel('time (s)')
    plt.ylabel('torque (Nm)')

    plt.subplot(236)
    plt.title('R_Ankle_Rot_Z')
    plt.plot(t, torque[5, :], 'b+-')
    plt.plot([T, T], [min(torque[5, :]), max(torque[5, :])], 'k:')
    plt.xlabel('time (s)')
    plt.ylabel('torque (Nm)')

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


def plot_pelvis_force(F, T, nbNoeuds):
    # Set time
    dt = T/nbNoeuds
    t = np.linspace(0, T - dt, nbNoeuds)

    plt.figure()
    plt.subplot(311)
    plt.title('Force Pelvis TX')
    # plt.plot([0, t], [-1000, -1000], 'k--')  # lower bound
    # plt.plot([0, t], [1000, 1000], 'k--')  # upper bound
    for n in range(nbNoeuds - 1):
        plt.plot([t[n], t[n + 1], t[n + 1]], [F[0, n], F[0, n], F[0, n + 1]], 'b')

    plt.subplot(312)
    plt.title('Force Pelvis TY')
    # plt.plot([0, t[-1]], [-2000, -2000], 'k--')  # lower bound
    # plt.plot([0, t[-1]], [2000, 2000], 'k--')  # upper bound
    for n in range(nbNoeuds - 1):
        plt.plot([t[n], t[n + 1], t[n + 1]], [F[1, n], F[1, n], F[1, n + 1]], 'b')

    plt.subplot(313)
    plt.title('Force Pelvis RZ')
    # plt.plot([0, t[-1]], [-200, -200], 'k--')  # lower bound
    # plt.plot([0, t[-1]], [200, 200], 'k--')  # upper bound
    for n in range(nbNoeuds - 1):
        plt.plot([t[n], t[n + 1], t[n + 1]], [F[2, n], F[2, n], F[2, n + 1]], 'b')

    plt.show(block=False)


def plot_control(u, U_real, T, nbNoeuds):
    nbU = 20
    nbMus = 17

    # plot control
    def plot_control(ax, t, x):
        nbPoints = len(np.array(x))
        for n in range(nbPoints - 2):
            ax.plot([t[n], t[n + 1], t[n + 1]], [x[n], x[n], x[n + 1]], 'b')

    # Set time
    t = np.linspace(0, T, nbNoeuds + 1)

    # CONTROL
    fig1, axes1 = plt.subplots(5, 4, sharex=True, figsize=(10, 10))
    Labels = ['GLUT_MAX1', 'GLUT_MAX2', 'GLUT_MAX3', 'GLUT_MED1', 'GLUT_MED2', 'GLUT_MED3',
              'R_SEMIMEM', 'R_SEMITEN', 'R_BI_FEM_LH', 'R_RECTUS_FEM', 'R_VAS_MED', 'R_VAS_INT',
              'R_VAS_LAT', 'R_GAS_MED', 'R_GAS_LAT', 'R_SOLEUS', 'R_TIB_ANT', 'Pelvis Tx', 'Pelvis Ty', 'Pelvis Rz']    # Control labels
    axes1 = axes1.flatten()
    u_emg = 9
    for i in range(nbU):
        ax = axes1[i]  # get the correct subplot
        ax.set_title(Labels[i])  # put control label
        ax.plot([T, T], [0, 1], 'k:')  # end of the stance phase
        plot_control(ax, t, u[i, :])
        ax.grid(True)
        if (i != 1) and (i != 2) and (i != 3) and (i != 5) and (i != 6) and (i != 11) and (i != 12) and (i < nbMus):
            ax.plot(t, U_real[u_emg, :], 'r')  # plot emg if available
            u_emg -= 1
        if (i > nbU - 5):
            ax.set_xlabel('time (s)')
        if (i < (nbMus)):
            ax.plot([0, t[-1]], [0, 0], 'k--')  # lower bound
            ax.plot([0, t[-1]], [1, 1], 'k--')  # upper bound
            ax.yaxis.set_ticks(np.arange(0, 1.5, 0.5))
    plt.show(block=False)


def plot_GRF(GRF, GRF_real, T, nbNoeuds):
    # Set time
    t = np.linspace(0, T , nbNoeuds + 1)

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
    ax_v.plot(t, GRF[2, :], 'b', alpha = 0.5)
    ax_v.set_xlabel('time (s)')
    ax_v.grid(True)
    fig.tight_layout()
    plt.show(block=False)

def plot_markers_result(sol_q, T_phase, nbNoeuds, nbMarker, M_real):
    # Plot leg trajectory with 5 markers

    # INPUT
    # sol_q          = optimized joint position (nbQ x nbNoeuds)
    # nbNoeuds_phase = shooting points for each phase (nbPhase)
    # nbMarker       = number of markers

    # PARAMETERS
    M_simu   = np.zeros((3, nbMarker, nbNoeuds + 1))

    plt.figure()
    # FIND MARKERS POSITIONS
    for k_stance in range(nbNoeuds + 1):
        model   = biorbd.Model('/home/leasanchez/programmation/Marche_Florent/ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact.bioMod')
        markers = model.markers(sol_q[:, k_stance])
        for nMark in range(nbMarker):
            M_simu[:, nMark, k_stance] = markers[nMark].to_array()

        # PLOT ARTIFICIAL SEGMENTS TO FOLLOW LEG MOVEMENT
        M_aff = np.zeros((3, 5))
        M_aff[:, 0] = M_simu[:, 2, k_stance]
        M_aff[:, 1] = M_simu[:, 4, k_stance]
        M_aff[:, 2] = M_simu[:, 11, k_stance]
        M_aff[:, 3] = M_simu[:, 19, k_stance]
        M_aff[:, 4] = M_simu[:, 22, k_stance]
        plt.plot(M_aff[0, :], M_aff[2, :], 'bo-', alpha=0.5)
        plt.plot([M_real[0, 2, k_stance], M_real[0, 4, k_stance], M_real[0, 11, k_stance], M_real[0, 19, k_stance], M_real[0, 22, k_stance]],
                 [M_real[2, 2, k_stance], M_real[2, 4, k_stance], M_real[2, 11, k_stance], M_real[2, 19, k_stance], M_real[2, 22, k_stance]], 'r+')

    # for k_swing in range(nbNoeuds_phase[1] + 1):
    #     model = biorbd.Model('/home/leasanchez/programmation/Marche_Florent/ModelesS2M/ANsWER_Rleg_6dof_17muscle_0contact.bioMod')
    #     markers = model.markers(sol_q[:, nbNoeuds_phase[0] + k_swing])
    #     for nMark in range(nbMarker):
    #         M_simu[:, nMark, nbNoeuds_phase[0] + k_swing] = markers[nMark].to_array()
    #
    #     # PLOT ARTIFICIAL SEGMENTS TO FOLLOW LEG MOVEMENT
    #     M_aff = np.zeros((3, 5))
    #     M_aff[:, 0] = M_simu[:, 2, nbNoeuds_phase[0] + k_swing]
    #     M_aff[:, 1] = M_simu[:, 4, nbNoeuds_phase[0] + k_swing]
    #     M_aff[:, 2] = M_simu[:, 11, nbNoeuds_phase[0] + k_swing]
    #     M_aff[:, 3] = M_simu[:, 19, nbNoeuds_phase[0] + k_swing]
    #     M_aff[:, 4] = M_simu[:, 22, nbNoeuds_phase[0] + k_swing]
    #     plt.plot(M_aff[0, :], M_aff[2, :], 'go-', alpha=0.5)
    #     plt.plot([M_real[0, 2, nbNoeuds_phase[0] + k_swing], M_real[0, 4, nbNoeuds_phase[0] + k_swing], M_real[0, 11, nbNoeuds_phase[0] + k_swing], M_real[0, 19, nbNoeuds_phase[0] + k_swing], M_real[0, 22, nbNoeuds_phase[0] + k_swing]],
    #              [M_real[2, 2, nbNoeuds_phase[0] + k_swing], M_real[2, 4, nbNoeuds_phase[0] + k_swing], M_real[2, 11, nbNoeuds_phase[0] + k_swing], M_real[2, 19, nbNoeuds_phase[0] + k_swing], M_real[2, 22, nbNoeuds_phase[0] + k_swing]], 'm+')

    plt.plot([-0.5, 1.5], [0, 0], 'k--')

    plt.show(block = False)

