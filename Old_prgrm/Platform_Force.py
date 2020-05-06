from ezc3d import c3d
from pylab import *
import numpy as np

def GetForces(file):
    measurements  = c3d(file)
    analog        = measurements['data']['analogs']
    labels_analog = measurements['parameters']['ANALOG']['LABELS']['value']
    nbPF          = measurements['parameters']['FORCE_PLATFORM']['USED']['value'][0][0]

    F = np.zeros((len(analog[0, 0,:]), 3, nbPF))
    for p in range(nbPF):
        F[:, 0, p] = analog[0, labels_analog.index("Fx" + str(p + 1)), :].squeeze()  # Fx
        F[:, 1, p] = analog[0, labels_analog.index("Fy" + str(p + 1)), :].squeeze()  # Fy
        F[:, 2, p] = analog[0, labels_analog.index("Fz" + str(p + 1)), :].squeeze()  # Fz

    return F

def GetMoment(file):
    measurements  = c3d(file)
    analog        = measurements['data']['analogs']
    labels_analog = measurements['parameters']['ANALOG']['LABELS']['value']
    nbPF          = measurements['parameters']['FORCE_PLATFORM']['USED']['value'][0][0]

    M = np.zeros((len(analog[0, 0,:]), 3, nbPF))
    for p in range(nbPF):
        M[:, 0, p] = analog[0, labels_analog.index("Mx" + str(p + 1)), :].squeeze() * 1e-3  # Fx
        M[:, 1, p] = analog[0, labels_analog.index("My" + str(p + 1)), :].squeeze() * 1e-3  # Fy
        M[:, 2, p] = analog[0, labels_analog.index("Mz" + str(p + 1)), :].squeeze() * 1e-3  # Fz

    return M


def GetGroundReactionForces(file):
    measurements  = c3d(file)
    analog        = measurements['data']['analogs']
    labels_analog = measurements['parameters']['ANALOG']['LABELS']['value']
    nbPF          = measurements['parameters']['FORCE_PLATFORM']['USED']['value'][0][0]

    GRF = np.zeros((len(analog[0, 0,:]), 3, nbPF))
    for p in range(nbPF):
        GRF[:, 0, p] = analog[0, labels_analog.index("Fx" + str(p + 1)), :].squeeze()    # Fx
        GRF[:, 1, p] = analog[0, labels_analog.index("Fy" + str(p + 1)), :].squeeze()    # Fy
        GRF[:, 2, p] = - analog[0, labels_analog.index("Fz" + str(p + 1)), :].squeeze()  # Fz
    return GRF


def ComputeCoP(file):
    measurements  = c3d(file)
    analog        = measurements['data']['analogs']
    nbPF          = measurements['parameters']['FORCE_PLATFORM']['USED']['value'][0][0]
    corners       = np.reshape( np.reshape(measurements['parameters']['FORCE_PLATFORM']['CORNERS']['value'] * 1e-3, (3 * 4 * 2, 1)), (2, 4, 3)) #platform x corners x coord

    CoP1 = np.zeros(((len(analog[0, 0, :]), 3, nbPF)))
    CoP = np.zeros(((len(analog[0, 0,:]), 3, nbPF)))
    F   = GetForces(file)
    M   = GetMoment(file)

    for p in range(nbPF):
        # Attention X et Y sont inversés sur la plaque !!!
        CoP1[:, 0, p] = np.divide(M[:, 0, p], F[:, 2, p])                  # Mx/Fz
        CoP1[:, 1, p] = - np.divide(M[:, 1, p], F[:, 2, p])                # -My/Fz
        CoP1[:, :, p][np.isnan(CoP1[:, :, p])] = 0

        # Center of the platform
        if p == 0:
            CoP[:, 0, p] = (corners[p, 1, 0] - corners[p, 2, 0]) / 2 + CoP1[:, 0, p]  # xcenter + CoPx
            CoP[:, 1, p] = (corners[p, 0, 1] - corners[p, 1, 1]) / 2 + CoP1[:, 1, p]  # ycenter + CoPy
        else :
            CoP[:, 0, p] = corners[p, 2, 0] + (corners[p, 1, 0] - corners[p, 2, 0]) / 2 + CoP1[:, 0, p]
            CoP[:, 1, p] = (corners[p, 0, 1] - corners[p, 1, 1]) / 2 + CoP1[:, 1, p]

    return CoP