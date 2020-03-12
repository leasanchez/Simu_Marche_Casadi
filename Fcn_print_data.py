import os
import numpy as np

def save_GRF_real(name_subject, save_dir, GRF_real):
    # SAVE GRF FROM PLATFORM
    filename_GRF = name_subject + '_GRF.txt'
    nbNoeuds = len(GRF_real[0, :])
    if filename_GRF not in os.listdir(save_dir):
        f = open(save_dir + filename_GRF, 'a')
        f.write("Ground Reaction Forces from force plateform \n\n")
        for n in range(nbNoeuds):
            np.savetxt(f, GRF_real[:, n], delimiter=' , ')
            f.write("\n")
        f.close()

def save_Markers_real(name_subject, save_dir, M_real):
    # SAVE MARKERS POSITIONS FROM MOTION CAPTURE
    filename_M = name_subject + '_Markers.txt'
    nbNoeuds   = len(M_real[0, 0, :])
    nbMarkers  = len(M_real[0, :, 0])

    if filename_M not in os.listdir(save_dir):
        f = open(save_dir + filename_M, 'a')
        f.write("Markers position from motion capture \n\n")
        for n in range(nbNoeuds):
            for m in range(nbMarkers):
                f.write(str(m + 1) + "  :  ")
                np.savetxt(f, M_real[:, m, n], delimiter=',')
                f.write("\n")
            f.write("\n")
        f.close()

def save_EMG_real(name_subject, save_dir, U_real):
    # SAVE EMG
    filename_EMG = name_subject + '_EMG.txt'
    nbNoeuds     = len(U_real[0, :])
    nbMus        = len(U_real[:, 0])
    if filename_EMG not in os.listdir(save_dir):
        f = open(save_dir + filename_EMG, 'a')
        f.write("Muscular excitations from emg \n\n")
        for n in range(nbNoeuds):
            for m in range(nbMus):
                f.write(str(m + 1) + "  :  " + str(U_real[m, n]) + " \n")
            f.write("\n")
        f.close()


def save_params(name_subject, save_dir, params):
    # SAVE PARAMETERS USED FOR SIMULATIONS
    filename_param = name_subject + '_params.txt'
    if filename_param in os.listdir(save_dir):
        os.remove(save_dir + filename_param)

    f = open(save_dir + filename_param, 'a')
    f.write("Parameters for the simulation \n\n\n")
    f.write('MODEL\n')
    f.write('model stance : ' + str(params.model_stance_dir))
    f.write('\nmodel swing  : ' + str(params.model_swing_dir))

    f.write('\n\nMODEL PARAMETERS\n')
    f.write('\nnbNoeuds stance : ' + str(params.nbNoeuds_stance))
    f.write('\nnbNoeuds swing  : ' + str(params.nbNoeuds_swing))
    f.write('\nnbNoeuds        : ' + str(params.nbNoeuds))
    f.write('\nT stance        : ' + str(params.T_stance))
    f.write('\nT swing         : ' + str(params.T_swing))
    f.write('\nT               : ' + str(params.T))
    f.write('\nnbMus           : ' + str(params.nbMus))
    f.write('\nnbQ             : ' + str(params.nbQ))
    f.write('\nnbMarker        : ' + str(params.nbMarker))
    f.write('\nnbBody          : ' + str(params.nbBody))
    f.write('\nnbContact       : ' + str(params.nbContact))
    f.write('\nnbU             : ' + str(params.nbU))
    f.write('\nnbX             : ' + str(params.nbX))
    f.write('\nnP              : ' + str(params.nP))

    f.write('\n\nWEIGHTING FACTORS\n')
    f.write('wL   : ' + str(params.wL))
    f.write('\nwMa : ' + str(params.wMa))
    f.write('\nwMt : ' + str(params.wMt))
    f.write('\nwU  : ' + str(params.wU))
    f.write('\nwR  : ' + str(params.wR))
    f.close()


def save_bounds(name_subject, save_dir, lbx, ubx, params):
    # SAVE BOUNDS
    filename_param = name_subject + '_params.txt'

    upperbound_u = ubx[:params.nbU]
    lowerbound_u = lbx[:params.nbU]
    upperbound_x = ubx[params.nbU * params.nbNoeuds : params.nbU * params.nbNoeuds + params.nbX]
    lowerbound_x = lbx[params.nbU * params.nbNoeuds: params.nbU * params.nbNoeuds + params.nbX]
    upperbound_p = ubx[- params.nbP :]
    lowerbound_p = ubx[- params.nbP :]


    f = open(save_dir + filename_param, 'a')
    f.write('\n\nBOUNDS\n')
    f.write('Control max\n')
    np.savetxt(f, upperbound_u, delimiter='\n')
    f.write('\nControl min\n')
    np.savetxt(f, lowerbound_u, delimiter='\n')
    f.write('\n\nState max \n')
    np.savetxt(f, upperbound_x, delimiter='\n')
    f.write('\nState min\n')
    np.savetxt(f, lowerbound_x, delimiter='\n')
    f.write('\n\nParameter max \n')
    np.savetxt(f, upperbound_p, delimiter='\n')
    f.write('\nParameter min\n')
    np.savetxt(f, lowerbound_p, delimiter='\n')
    f.close()


def save_initialguess(name_subject, save_dir, u0, x0, p0):
    # SAVE INITIAL GUESS
    filename_init = name_subject + '_initialguess.txt'
    nbNoeuds      = len(u0[0, :])

    if filename_init not in os.listdir(save_dir):
        f = open(save_dir + filename_init, 'a')
        f.write('Initial guess\n\n\n')
        f.write('CONTROL\n')
        for n in range(nbNoeuds):
            f.write('\nu0   ' + str(n) + '\n')
            np.savetxt(f, u0[:, n], delimiter='\n')
        f.write('\n\nSTATE\n')
        for n in range(nbNoeuds + 1):
            f.write('\nx0   ' + str(n) + '\n')
            np.savetxt(f, x0[:, n], delimiter='\n')
        f.write('\n\nPARAMETER\n')
        np.savetxt(f, p0, delimiter='\n')
    f.close()

def save_objective_values(name_subject, save_dir, objectif, constraint):
    # SAVE OBJECTIVE FUNCTION AND CONSTRAINTS VALUE FOR EACH ITERATION IN TXT
    filename_J = name_subject + '_objvalue.txt'
    f = open(save_dir + '/RES/' + filename_J, 'a')
    f.write('Global                 : ' + str(objectif[0]) + '\n')
    f.write('activation             : ' + str(objectif[1]) + '\n')
    f.write('emg                    : ' + str(objectif[2]) + '\n')
    f.write('marker                 : ' + str(objectif[3]) + '\n')
    f.write('ground reaction forces : ' + str(objectif[4]) + '\n')
    f.write('constraints            : ' + str(sum(constraint)) + '\n\n')
    f.close()