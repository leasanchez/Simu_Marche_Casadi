import biorbd
from casadi import *
from pylab import *
import numpy as np
import scipy.io as sio

def fcn_objective_activation(wL, u):
    # Minimize muscular excitation of muscle without emg

    # INPUT
    # wL            = weighting factor for muscular activation
    # u             = controls (including muscular excitation)

    # OUTPUT
    # Ja            = activation cost

    Ja = wL*(u[1]*u[1]) + wL*(u[2]*u[2]) + wL*(u[3]*u[3]) + wL*(u[5]*u[5]) + wL*(u[6]*u[6]) + wL*(u[11]*u[11]) + wL*(u[12]*u[12])
    #     GLUT_MAX2     +    GLUT_MAX3   +  GLUT_MED1     +   GLUT_MED3    +  R_SEMIMEM     +  R_VAS_INT       +    R_VAS_LAT

    return Ja


def fcn_objective_emg(wU, u, U_real):
    # Tracking muscular excitation for muscle with emg

    # INPUT
    # wU            = weighting factor for muscular excitation
    # u             = controls (including muscular excitation)
    # U_real        = measured muscular excitations

    # OUTPUT
    # Je            = cost of the difference between real and simulated muscular excitation

    Je = wU * ((u[0] - U_real[9]) * (u[0] - U_real[9]))           # GLUT_MAX1
    # Je += we*(Uk[1] - U_real[1, k])     # GLUT_MAX2
    # Je += we*(Uk[2] - U_real[2, k])     # GLUT_MAX3
    # Je += we*(Uk[3] - U_real[3, k])     # GLUT_MED1
    Je += wU * ((u[4] - U_real[8]) * (u[4] - U_real[8]))          # GLUT_MED2
    # Je += we*(Uk[5] - U_real[5, k])     # GLUT_MED3
    # Je += we*(Uk[6] - U_real[6, k])     # R_SEMIMEM
    Je += wU * ((u[7] - U_real[7]) * (u[7] - U_real[7]))          # R_SEMITEN
    Je += wU * ((u[8] - U_real[6]) * (u[8] - U_real[6]))          # R_BI_FEM_LH
    Je += wU * ((u[9] - U_real[5]) * (u[9] - U_real[5]))          # R_RECTUS_FEM
    Je += wU * ((u[10] - U_real[4]) * (u[10] - U_real[4]))        # R_VAS_MED
    # Je += we*(Uk[11] - U_real[11, k])   # R_VAS_INT
    # Je += we*(Uk[12] - U_real[12, k])   # R_VAS_LAT
    Je += wU * ((u[13] - U_real[3]) * (u[13] - U_real[3]))        # R_GAS_MED
    Je += wU * ((u[14] - U_real[2]) * (u[14] - U_real[2]))        # R_GAS_LAT
    Je += wU * ((u[15] - U_real[1]) * (u[15] - U_real[1]))        # R_SOLEUS
    Je += wU * ((u[16] - U_real[0]) * (u[16] - U_real[0]))        # R_TIB_ANT
    return Je

def fcn_objective_markers(wMa, wMt, Q, M_real, Gaitphase):
    # Tracking markers position

    # INPUT
    # wMa           = weighting factor for anatomical markers
    # wMt           = weighting factor for technical markers
    # Q             = generalized positions (state)
    # M_real        = real markers position (x, y, z)
    # Gaitphase     = gait cycle phase (used to define which model is used)

    # OUTPUT
    # Jm            = cost of the difference between real and simulated markers positions (x & z)

    if Gaitphase == 'stance':
        # SET MODEL
        model = biorbd.Model('/home/leasanchez/programmation/Simu_Marche_Casadi/ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact.bioMod')
        Get_Markers = external('libmarche_Florent_markers_stance', 'libmarche_Florent_markers_stance.so', {'enable_fd': True})
    else:
        # SET MODEL
        model = biorbd.Model('/home/leasanchez/programmation/Simu_Marche_Casadi/ModelesS2M/ANsWER_Rleg_6dof_17muscle_0contact.bioMod')
        Get_Markers = external('libmarche_Florent_markers_swing', 'libmarche_Florent_markers_swing.so', {'enable_fd': True})

    markers = Get_Markers(Q, 1, 1)                                                  # markers position
    Jm = 0
    for nMark in range(model.nbMarkers()):
        if model.marker(nMark).isAnatomical():
            Jm += wMa * ((markers[3*nMark] - M_real[0, nMark]) * (markers[3*nMark] - M_real[0, nMark]))             # x
            Jm += wMa * ((markers[3*nMark + 2] - M_real[2, nMark]) * (markers[3*nMark + 2] - M_real[2, nMark]))     # z
        else:
            Jm += wMt * ((markers[3*nMark] - M_real[0, nMark]) * (markers[3*nMark] - M_real[0, nMark]))
            Jm += wMt * ((markers[3*nMark + 2] - M_real[2, nMark]) * (markers[3*nMark + 2] - M_real[2, nMark]))
    return Jm


def fcn_objective_GRF(wR, x, u, GRF_real):
    # Tracking ground reaction forces

    # INPUT
    # wR            = weighting factor for ground reaction forces
    # x             = state : generalized positions Q, velocities dQ and muscular activation a
    # GRF_real      = real ground reaction forces from force platform

    # OUTPUT
    # JR            = cost of the difference between real and simulated ground reaction forces

    # SET MODEL
    model = biorbd.Model('/home/leasanchez/programmation/Simu_Marche_Casadi/ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact.bioMod')

    activations = u[: model.nbMuscleTotal()]
    Q           = x[:model.nbQ()]
    dQ          = x[model.nbQ(): 2 * model.nbQ()]


    muscularJointTorque = external('libmuscular_joint_torque', 'libmuscular_joint_torque.so', {'enable_fd': True})
    joint_torque    = muscularJointTorque(activations, Q, dQ)
    joint_torque[0] = u[model.nbMuscleTotal() + 0]           # ajout des forces au pelvis
    joint_torque[1] = u[model.nbMuscleTotal() + 1]
    joint_torque[2] = u[model.nbMuscleTotal() + 2]

    # COMPUTE THE GROUND REACTION FORCES
    Forward_Dynamics_Contact = external('libforward_dynamics_contact', 'libforward_dynamics_contact.so',{'enable_fd': True})
    GRF                      = Forward_Dynamics_Contact(Q, dQ, joint_torque)[2*model.nbQ():]

    JR  = wR * ((GRF[0] - GRF_real[1]) * (GRF[0] - GRF_real[1]))         # Fx
    JR += wR * ((GRF[2] - GRF_real[2]) * (GRF[2] - GRF_real[2]))         # Fz
    return GRF, JR