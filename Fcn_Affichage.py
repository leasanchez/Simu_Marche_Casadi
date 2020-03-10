import biorbd
from casadi import *
from pylab import *
import numpy as np

# AFFICHAGE DONNEES EXPERIMENTALES
def affichage_emg_real(U_real, T_stance, T_swing, nbNoeuds_stance, nbNoeuds_swing):
    plt.figure()
    plt.title("EMG for the gait cycle")
    nbMus    = len(U_real[:, 0])
    t_stance = np.linspace(0, T_stance, nbNoeuds_stance)
    t_swing = t_stance[-1] + np.linspace(0, T_swing, nbNoeuds_swing)
    t = np.hstack([t_stance, t_swing])

    Labels = ['R_Tibialis_Anterior', 'R_Soleus', 'R_Gastrocnemius_Lateralis', 'R_Gastrocnemius_Medialis', 'R_Vastus_Medialis', 'R_Rectus_Femoris', 'R_Biceps_Femoris', 'R_Semitendinous', 'R_Gluteus_Medius', 'R_Gluteus_Maximus']
    for nMus in range(nbMus):
        plt.plot(t, U_real[nMus, :], '+-', Label = Labels[nMus])

    plt.legend()

def affichage_markers_real(M_real):
    plt.figure()
    plt.title("Markers position during the gait")

    plt.plot(M_real[0, 2, :], M_real[2, 2, :], '+', Label='R_IAS')       # pelvis
    plt.plot(M_real[0, 4, :], M_real[2, 4, :], '+', Label='R_FTC')       # femur
    plt.plot(M_real[0, 11, :], M_real[2, 11, :], '+', Label='R_FAX')     # tibia
    plt.plot(M_real[0, 19, :], M_real[2, 19, :], '+', Label='R_FCC')     # pied
    plt.plot(M_real[0, 22, :], M_real[2, 22, :], '+', Label='R_FM2')     # pied

    plt.legend()

def affichage_GRF_real(GRF_real, T_stance, T_swing, nbNoeuds_stance, nbNoeuds_swing):
    t_stance = np.linspace(0, T_stance, nbNoeuds_stance)
    t_swing = t_stance[-1] + np.linspace(0, T_swing, nbNoeuds_swing)
    t = np.hstack([t_stance, t_swing])

    plt.figure()
    plt.title("GRF A/P  during the gait")
    plt.plot(t, GRF_real[1, :], '+-')
    plt.plot([T_stance, T_stance], [0, max(GRF_real[1, :])], 'k--')

    plt.figure()
    plt.title("GRF vertical during the gait")
    plt.plot(t, GRF_real[2, :], '+-')
    plt.plot([T_stance, T_stance], [min(GRF_real[1, :]), max(GRF_real[2, :])], 'k--')


# ---------------- PLOT RESULTS ----------------------------------------------------------------------------------------
def affichage_q_results(sol_q, T_phase, nbNoeuds_phase):
    # Print joint kinetic

    # INPUT
    # sol_q          = optimized q (nbQ x nbNoeuds)
    # T_phase        = vector with phases time
    # nbNoeuds_phase = shooting points for each phase

    # OUTPUT
    # hip flexion/extension
    # knee flexion/extension
    # ankle flexion/extension

    # PARAMETERS
    nbPhase  = len(T_phase)
    nbNoeuds = len(sol_q[0, :])
    nbQ      = len(sol_q[:, 0])

    # TIME VECTOR
    for nPhase in range(nbPhase):
        if nPhase == 0:
            t = np.linspace(0, T_phase[0], nbNoeuds_phase[0])
        else:
            t = np.hstack([t, t[-1] + np.linspace(0, T_phase[nPhase], nbNoeuds_phase[nPhase])])

    plt.figure()
    plt.title("flexion/extension hip")
    plt.plot(t, sol_q[3, :-1]*180/np.pi)

    plt.figure()
    plt.title("flexion/extension knee")
    plt.plot(t, sol_q[4, :-1]*180/np.pi)

    plt.figure()
    plt.title("flexion/extension ankle")
    plt.plot(t, sol_q[5, :-1]*180/np.pi)


def affichage_activation_results(sol_a, T_phase, nbNoeuds_phase, U_real):
    # Print optimized muscular activation VS emg

    # INPUT
    # sol_a          = optimized activation (nbMus x nbNoeuds)
    # T_phase        = vector with phases time (nbPhase)
    # nbNoeuds_phase = shooting points for each phase (nbPhase)

    # OUTPUT
    # subplot(nbMus)

    # PARAMETERS
    nbPhase  = len(T_phase)
    nbNoeuds = len(sol_a[0, :]) - 1
    nbMus    = len(sol_a[:, 0])
    Labels   = ['GLUT_MAX1', 'GLUT_MAX2', 'GLUT_MAX3', 'GLUT_MED1', 'GLUT_MED2', 'GLUT_MED3',
              'R_SEMIMEM', 'R_SEMITEN', 'R_BI_FEM_LH', 'R_RECTUS_FEM', 'R_VAS_MED', 'R_VAS_INT',
              'R_VAS_LAT', 'R_GAS_MED', 'R_GAS_LAT', 'R_SOLEUS', 'R_TIB_ANT']

    plt.figure()
    plt.title("Results muscular activation")

    # TIME VECTOR
    for nPhase in range(nbPhase):
        if nPhase == 0:
            t = np.linspace(0, T_phase[0], nbNoeuds_phase[0])
        else:
            t = np.hstack([t, t[-1] + np.linspace(0, T_phase[nPhase], nbNoeuds_phase[nPhase] + 1)])

    # PLOT MUSCULAR ACTIVATION AND EMG
    nMus_emg = 9
    for nMus in range(nbMus):
        plt.subplot(5, 4, nMus + 1)
        if nMus == 1 or nMus == 2 or nMus == 3 or nMus == 5 or nMus == 6 or nMus == 11 or nMus == 12:
            plt.plot(t, sol_a[nMus, :], 'b+-', alpha = 0.5)
        else:
            plt.plot(t, sol_a[nMus, :], 'b+-', alpha = 0.5)
            plt.plot(t, U_real[nMus_emg, :], 'r+')
            nMus_emg -= 1

        plt.title(Labels[nMus])


def affichage_markers_result(sol_q, T_phase, nbNoeuds_phase, nbMarker, M_real):
    # Plot leg trajectory with 5 markers

    # INPUT
    # sol_q          = optimized joint position (nbQ x nbNoeuds)
    # nbNoeuds_phase = shooting points for each phase (nbPhase)
    # nbMarker       = number of markers

    # PARAMETERS
    nbNoeuds = len(sol_q[0, :]) - 1
    nbPhase  = len(T_phase)
    nbQ      = len(sol_q[:, 0])
    M_simu   = np.zeros((3, nbMarker, nbNoeuds + 1))

    plt.figure()
    # FIND MARKERS POSITIONS
    for k_stance in range(nbNoeuds_phase[0]):
        model   = biorbd.Model('/home/leasanchez/programmation/Simu_Marche_Casadi/ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact.bioMod')
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

    for k_swing in range(nbNoeuds_phase[1] + 1):
        model = biorbd.Model('/home/leasanchez/programmation/Simu_Marche_Casadi/ModelesS2M/ANsWER_Rleg_6dof_17muscle_0contact.bioMod')
        markers = model.markers(sol_q[:, nbNoeuds_phase[0] + k_swing])
        for nMark in range(nbMarker):
            M_simu[:, nMark, nbNoeuds_phase[0] + k_swing] = markers[nMark].to_array()

        # PLOT ARTIFICIAL SEGMENTS TO FOLLOW LEG MOVEMENT
        M_aff = np.zeros((3, 5))
        M_aff[:, 0] = M_simu[:, 2, nbNoeuds_phase[0] + k_swing]
        M_aff[:, 1] = M_simu[:, 4, nbNoeuds_phase[0] + k_swing]
        M_aff[:, 2] = M_simu[:, 11, nbNoeuds_phase[0] + k_swing]
        M_aff[:, 3] = M_simu[:, 19, nbNoeuds_phase[0] + k_swing]
        M_aff[:, 4] = M_simu[:, 22, nbNoeuds_phase[0] + k_swing]
        plt.plot(M_aff[0, :], M_aff[2, :], 'go-', alpha=0.5)
        plt.plot([M_real[0, 2, nbNoeuds_phase[0] + k_swing], M_real[0, 4, nbNoeuds_phase[0] + k_swing], M_real[0, 11, nbNoeuds_phase[0] + k_swing], M_real[0, 19, nbNoeuds_phase[0] + k_swing], M_real[0, 22, nbNoeuds_phase[0] + k_swing]],
                 [M_real[2, 2, nbNoeuds_phase[0] + k_swing], M_real[2, 4, nbNoeuds_phase[0] + k_swing], M_real[2, 11, nbNoeuds_phase[0] + k_swing], M_real[2, 19, nbNoeuds_phase[0] + k_swing], M_real[2, 22, nbNoeuds_phase[0] + k_swing]], 'm+')

    plt.plot([-0.5, 1.5], [0, 0], 'k--')

    # TIME
    for nPhase in range(nbPhase):
        if nPhase == 0:
            t = np.linspace(0, T_phase[0], nbNoeuds_phase[0])
        else:
            t = np.hstack([t, t[-1] + np.linspace(0, T_phase[nPhase], nbNoeuds_phase[nPhase] + 1)])

    # PLOT MARKERS DIFFERENCES
    diff = M_simu[:, :, :] - M_real             # 3 x nMarker x nbNoeuds

    plt.figure()
    plt.title('Difference between simulated and real markers')

    for nMark in range(nbMarker):
        plt.subplot(6, 5, nMark + 1)
        plt.plot(t, diff[0, nMark, :])
        plt.plot(t, diff[2, nMark, :])



