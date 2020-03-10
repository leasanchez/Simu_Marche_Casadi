import biorbd
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

def load_initialguess_q(c3d_file, kalman_file, T, nbNoeuds, GaitPhase):  # A MODIFIER !!
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

    # LOAD MAT FILE FOR GENERALIZED COORDINATES
    kalman = sio.loadmat(kalman_file)
    Q_real = kalman['Q2']

    [start, stop_stance, stop] = Get_Event(c3d_file)

    # INTERPOLATE AND GET KALMAN JOINT POSITION FOR SHOOTING POINT FOR THE CYCLE PHASE
    if GaitPhase == 'swing':
        # T = T_swing
        t      = np.linspace(0, T, int(stop - stop_stance))
        node_t = np.linspace(0, T, nbNoeuds + 1)
        f      = interp1d(t, Q_real[:, int(stop_stance) + 1: int(stop) + 1], kind='cubic')
        Q0     = f(node_t)


    elif GaitPhase == 'stance':
        # T = T_stance
        t      = np.linspace(0, T, int(stop_stance - start) + 1)
        node_t = np.linspace(0, T, nbNoeuds)
        f      = interp1d(t, Q_real[:, int(start): int(stop_stance) + 1], kind='cubic')
        Q0     = f(node_t)

    return Q0