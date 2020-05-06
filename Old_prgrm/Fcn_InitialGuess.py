from casadi import *
from pylab import *
import numpy as np
from scipy.interpolate import interp1d
import scipy.io as sio
from LoadData import Get_Event

def load_initialguess_muscularExcitation(U_real):
    # Create initial vector for muscular excitation (nbNoeuds x nbMus)
    # Based on EMG from the c3d file

    # INPUT
    # U_real          = muscular excitation from the c3d file

    # OUTPUT
    # U0             = initial guess for muscular excitation (3 x nbNoeuds)

    nbNoeuds = len(U_real[0, :])
    nbMus    = len(U_real[:, 0])

    U0 = np.zeros((nbMus + 7, nbNoeuds))

    U0[0, :]  = U_real[9, :]  # glut_max1_r
    U0[1, :]  = U_real[9, :]  # glut_max2_r
    U0[2, :]  = U_real[9, :]  # glut_max3_r
    U0[3, :]  = U_real[8, :]  # glut_med1_r
    U0[4, :]  = U_real[8, :]  # glut_med2_r
    U0[5, :]  = U_real[8, :]  # glut_med3_r
    U0[6, :]  = U_real[7, :]  # semimem_r
    U0[7, :]  = U_real[7, :]  # semiten_r
    U0[8, :]  = U_real[6, :]  # bi_fem_r
    U0[9, :]  = U_real[5, :]  # rectus_fem_r
    U0[10, :] = U_real[4, :]  # vas_med_r
    U0[11, :] = U_real[4, :]  # vas_int_r
    U0[12, :] = U_real[4, :]  # vas_lat_r
    U0[13, :] = U_real[3, :]  # gas_med_r
    U0[14, :] = U_real[2, :]  # gas_lat_r
    U0[15, :] = U_real[1, :]  # soleus_r
    U0[16, :] = U_real[0, :]  # tib_ant_r
    return U0

def load_initialguess_muscularExcitation_2(U_real):
    # Create initial vector for muscular excitation (nbNoeuds x nbMus)
    # Based on EMG from the c3d file

    # INPUT
    # U_real          = muscular excitation from the c3d file

    # OUTPUT
    # U0             = initial guess for muscular excitation (3 x nbNoeuds)

    nbNoeuds = len(U_real[0, :])
    nbMus    = len(U_real[:, 0])

    U0 = np.zeros((nbMus + 7, nbNoeuds))

    U0[0, :]  = U_real[9, :]  # glut_max1_r
    U0[1, :]  = np.zeros(nbNoeuds) + 0.1 # glut_max2_r
    U0[2, :]  = np.zeros(nbNoeuds) + 0.1  # glut_max3_r
    U0[3, :]  = np.zeros(nbNoeuds) + 0.1  # glut_med1_r
    U0[4, :]  = U_real[8, :]  # glut_med2_r
    U0[5, :]  = np.zeros(nbNoeuds) + 0.1  # glut_med3_r
    U0[6, :]  = np.zeros(nbNoeuds) + 0.1  # semimem_r
    U0[7, :]  = U_real[7, :]  # semiten_r
    U0[8, :]  = U_real[6, :]  # bi_fem_r
    U0[9, :]  = U_real[5, :]  # rectus_fem_r
    U0[10, :] = U_real[4, :]  # vas_med_r
    U0[11, :] = np.zeros(nbNoeuds) + 0.1  # vas_int_r
    U0[12, :] = np.zeros(nbNoeuds) + 0.1  # vas_lat_r
    U0[13, :] = U_real[3, :]  # gas_med_r
    U0[14, :] = U_real[2, :]  # gas_lat_r
    U0[15, :] = U_real[1, :]  # soleus_r
    U0[16, :] = U_real[0, :]  # tib_ant_r
    return U0

def load_initialguess_q(params, GaitPhase):
    # Create initial vector for joint position (nbNoeuds x nbQ)
    # Based on Kalman filter??

    # INPUT
    # c3d_file       = path and name of the c3d file -- get event to determine indexes of HS and TO
    # kalman_file    = path and name of the file containing Q value
    # T              = phase time
    # nbNoeuds       = number of shooting points
    # Gaitphase      = gait cycle phase : stance, swing

    # OUTPUT
    # Q0             = initial guess for joint position (nbQ x nbNoeuds)
    kalman_file   = params.kalman_file
    c3d_file      = params.file

    if GaitPhase == 'stance':
        T        = params.T_stance
        nbNoeuds = params.nbNoeuds_stance
    elif GaitPhase == 'swing':
        T        = params.T_swing
        nbNoeuds = params.nbNoeuds_swing
    else:
        T_stance = params.T_stance
        nbNoeuds_stance = params.nbNoeuds_stance
        T_swing = params.T_swing
        nbNoeuds_swing = params.nbNoeuds_swing
        T = params.T
        nbNoeuds = params.nbNoeuds

    # LOAD MAT FILE FOR GENERALIZED COORDINATES
    kalman = sio.loadmat(kalman_file)
    Q_real = kalman['Q2']

    [start, stop_stance, stop] = Get_Event(c3d_file)

    # INTERPOLATE AND GET KALMAN JOINT POSITION FOR SHOOTING POINT FOR THE CYCLE PHASE
    if GaitPhase == 'swing':
        # T = T_swing
        t      = np.linspace(0, T, int(stop - stop_stance) + 1)
        node_t = np.linspace(0, T, nbNoeuds + 1)
        f      = interp1d(t, Q_real[:, int(stop_stance): int(stop) + 1], kind='cubic')
        Q0     = f(node_t)

    elif GaitPhase == 'stance':
        # T = T_stance
        t      = np.linspace(0, T, int(stop_stance - start) + 1)
        node_t = np.linspace(0, T, nbNoeuds + 1)
        f      = interp1d(t, Q_real[:, int(start): int(stop_stance) + 1], kind='cubic')
        Q0     = f(node_t)

    else:
        t_stance      = np.linspace(0, T_stance, int(stop_stance - start) + 1)
        node_t_stance = np.linspace(0, T_stance, nbNoeuds_stance + 1)
        f_stance      = interp1d(t_stance, Q_real[:, int(start): int(stop_stance) + 1], kind='cubic')
        Q0_stance     = f_stance(node_t_stance)

        t_swing      = np.linspace(0, T_swing, int(stop - stop_stance) + 1)
        node_t_swing = np.linspace(0, T_swing, nbNoeuds_swing + 1)
        f_swing      = interp1d(t_swing, Q_real[:, int(stop_stance): int(stop) + 1], kind='cubic')
        Q0_swing     = f_swing(node_t_swing)

        Q0 = np.hstack([Q0_stance[:, :-1], Q0_swing])

    return Q0