import numpy as np
import biorbd
from matplotlib import pyplot as plt
from Fcn_Objective import fcn_objective_activation, fcn_objective_emg


def plot_callback(callback, params, ubx, lbx, u0, x0, U_real, GRF_real, M_real):
    # INITIALISATION GRAPH WITH INITIAL GUESS
    upperbound_u = ubx[:params.nbU]
    lowerbound_u = lbx[:params.nbU]
    upperbound_x = ubx[params.nbU * params.nbNoeuds : params.nbU * params.nbNoeuds + params.nbX]
    lowerbound_x = lbx[params.nbU * params.nbNoeuds: params.nbU * params.nbNoeuds + params.nbX]

    # TIME
    t_stance = np.linspace(0, params.T_stance, params.nbNoeuds_stance)
    t_swing = t_stance[-1] + np.linspace(0, params.T_swing, params.nbNoeuds_swing)
    t = np.hstack([t_stance, t_swing])

    def plot_control(ax, t, x):
        nbPoints = len(np.array(x))
        for n in range(nbPoints - 1):
            ax.plot([t[n], t[n + 1], t[n + 1]], [x[n], x[n], x[n + 1]], 'b')

    # CONTROL
    fig1, axes1 = plt.subplots(5, 4, sharex=True, figsize=(10, 10))                                                     # Create figure
    Labels = ['GLUT_MAX1', 'GLUT_MAX2', 'GLUT_MAX3', 'GLUT_MED1', 'GLUT_MED2', 'GLUT_MED3',
              'R_SEMIMEM', 'R_SEMITEN', 'R_BI_FEM_LH', 'R_RECTUS_FEM', 'R_VAS_MED', 'R_VAS_INT',
              'R_VAS_LAT', 'R_GAS_MED', 'R_GAS_LAT', 'R_SOLEUS', 'R_TIB_ANT', 'Pelvis Tx', 'Pelvis Ty', 'Pelvis Rz']    # Control labels
    axes1 = axes1.flatten()                                                                                             # Get axes figure (vector)
    u_emg = 9                                                                                                           # init variable for muscle with emg
    for i in range(params.nbU):
        ax = axes1[i]                                                            # get the correct subplot
        ax.set_title(Labels[i])                                                  # put control label
        ax.plot([0, params.T], [lowerbound_u[i], lowerbound_u[i]], 'k--')               # lower bound
        ax.plot([0, params.T], [upperbound_u[i], upperbound_u[i]], 'k--')               # upper bound
        ax.plot([params.T_stance, params.T_stance], [lowerbound_u[i], upperbound_u[i]], 'k:')  # end of the stance phase
        ax.grid(True)
        if (i != 1) and (i != 2) and (i != 3) and (i != 5) and (i != 6) and (i != 11) and (i != 12) and (i < (nbMus - 1)):
            ax.plot(t, U_real[u_emg, :], 'r')                                    # plot emg if available
            u_emg -= 1
        if (i > params.nbU - 5):
            ax.set_xlabel('time (s)')
        if (i < (params.nbMus - 1)):
            ax.yaxis.set_ticks(np.arange(0, 1, 0.5))
        plot_control(ax, t, u0[i, :])                                            # plot initial guess


    # STATES
    ts = np.hstack([t_stance, t_swing, t_swing[-1] + (t_swing[-1] - t_swing[-2])])                                      # Adjusted time (T + dt)

    fig2, axes2 = plt.subplots(2, 6, sharex=True, figsize=(20, 10))                                                     # Create figure
    axes2       = axes2.flatten()                                                                                       # Get axes figure (vector)
    Labels_X    = ['Pelvis_TX', 'Pelvis_TY', 'Pelvis_RZ', 'Hip', 'Knee', 'Ankle']                                       # Control labels

    for q in range(params.nbQ):
        ax1 = axes2[q]
        ax1.set_title('Q ' + Labels_X[q])
        if q != 0 and q != 1:
            ax1.plot([0, params.T], [lowerbound_x[q] * (180 / np.pi), lowerbound_x[q] * (180 / np.pi)], 'k--')          # lower bound
            ax1.plot([0, params.T], [upperbound_x[q] * (180 / np.pi), upperbound_x[q] * (180 / np.pi)], 'k--')          # upper bound
            ax1.plot([params.T_stance, params.T_stance], [lowerbound_x[q] * (180 / np.pi), upperbound_x[q] * (180 / np.pi)], 'k:')    # end of the stance phase
            ax1.plot(ts, x0[q, :] * (180 / np.pi), 'b')                                                                 # plot initial guess q (degre)
        else:
            ax1.plot([0, params.T], [lowerbound_x[q], lowerbound_x[q]], 'k--')
            ax1.plot([0, params.T], [upperbound_x[q], upperbound_x[q]], 'k--')
            ax1.plot([params.T_stance, params.T_stance], [lowerbound_x[q], upperbound_x[q]], 'k:')                      # end of the stance phase
            ax1.plot(ts, x0[q, :], 'b')                                                                                 # plot initial guess q (trans)
        ax1.grid(True)

        ax2 = axes2[q + params.nbQ]
        ax2.set_title('dQ ' + Labels_X[q])
        ax2.plot([0, params.T], [lowerbound_x[q], lowerbound_x[q]], 'k--')                                              # lower bound
        ax2.plot([0, params.T], [upperbound_x[q], upperbound_x[q]], 'k--')                                              # upper bound
        ax2.plot([params.T_stance, params.T_stance], [lowerbound_x[q], upperbound_x[q]], 'k:')                          # end of the stance phase
        ax2.plot(ts, x0[(q + params.nbQ), :], 'b')                                                                      # plot initial guess dq
        ax2.set_xlabel('time (s)')
        ax2.grid(True)

    # GROUND REACTION FORCES
    GRF = np.zeros((3, params.nbNoeuds))
    m   = biorbd.Model('/home/leasanchez/programmation/Simu_Marche_Casadi/ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact.bioMod')
    for k in range(params.nbNoeuds_stance):
        activations = u0[: params.nbMus]
        F           = u0[params.nbMus:]
        Q           = x0[:params.nbQ()]
        dQ          = x0[params.nbQ(): 2 * params.nbQ()]

        states = biorbd.VecBiorbdMuscleStateDynamics(params.nbMus)
        n_muscle = 0
        for state in states:
            state.setActivation(activations[n_muscle])  # Set muscles activations
            n_muscle += 1

        joint_torque    = m.muscularJointTorque(activations, Q, dQ)
        joint_torque[0] = F[0]
        joint_torque[1] = F[1]
        joint_torque[2] = F[2]

        C = m.getConstraints()
        m.ForwardDynamicsConstraintsDirect(Q, dQ, joint_torque, C)
        GRF[:, k] = C.getForce().to_array()

    # GROUND REACTION FORCES
    fig3, axes3 = plt.subplots(2, 1, sharex=True)
    axes3.flatten()

    ax_ap = axes3[0]
    ax_ap.set_title('GRF A/P  during the gait')
    ax_ap.plot(t, GRF_real[1, :], 'r')
    ax_ap.plot([params.T_stance, params.T_stance], [min(GRF_real[1, :]), max(GRF_real[1, :])], 'k:')  # end of the stance phase
    ax_ap.plot(t, GRF[0, :], 'b')
    ax_ap.grid(True)

    ax_v = axes3[1]
    ax_v.set_title('GRF vertical')
    ax_v.plot(t, GRF_real[2, :], 'r')
    ax_v.plot([params.T_stance, params.T_stance], [min(GRF_real[2, :]), max(GRF_real[2, :])], 'k:')
    ax_v.plot(t, GRF[2, :], 'b')
    ax_v.set_xlabel('time (s)')
    ax_v.grid(True)
    fig3.tight_layout()
    plt.show(block=False)

    def fcn_dyn_contact(x, u):
        activations = u[: params.nbMus]
        F           = u[params.nbMus:]
        Q           = x[:params.nbQ()]
        dQ          = x[params.nbQ(): 2 * params.nbQ()]

        states   = biorbd.VecBiorbdMuscleStateDynamics(params.nbMus)
        n_muscle = 0
        for state in states:
            state.setActivation(activations[n_muscle])  # Set muscles activations
            n_muscle += 1

        joint_torque = params.model_stance.muscularJointTorque(activations, Q, dQ)
        joint_torque[0] = F[0]
        joint_torque[1] = F[1]
        joint_torque[2] = F[2]

        ddQ = params.model_stance.ForwardDynamicsConstraintsDirect(Q, dQ, joint_torque)
        return np.hstack(dQ, ddQ)

    def fcn_dyn_nocontact(x, u):
        activations = u[: params.nbMus]
        F = u[params.nbMus:]
        Q = x[:params.nbQ()]
        dQ = x[params.nbQ(): 2 * params.nbQ()]

        states = biorbd.VecBiorbdMuscleStateDynamics(params.nbMus)
        n_muscle = 0
        for state in states:
            state.setActivation(activations[n_muscle])  # Set muscles activations
            n_muscle += 1

        joint_torque = params.model_swing.muscularJointTorque(activations, Q, dQ)
        joint_torque[0] = F[0]
        joint_torque[1] = F[1]
        joint_torque[2] = F[2]

        ddQ = params.model_swing.ForwardDynamics(Q, dQ, joint_torque)
        return np.hstack(dQ, ddQ)

    # Integration Runge Kutta 4
    def int_RK4(fcn, T, nbNoeuds, x, u):
        dn = T / nbNoeuds                 # Time step for shooting point
        dt = dn / 4                       # Time step for iteration
        xj = x
        for i in range(4):
            k1 = fcn(xj, u)
            x2 = xj + (dt / 2) * k1
            k2 = fcn(x2, u)
            x3 = xj + (dt / 2) * k2
            k3 = fcn(x3, u)
            x4 = xj + dt * k3
            k4 = fcn(x4, u)

            xj += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return xj


    while plt.get_fignums():            # figures opened
        if callback.update_sol:         # optimized values modified (new iteration)
            # STRUCTURE DATA
            sol_U = callback.sol_data[:params.nbU * params.nbNoeuds]
            sol_X = callback.sol_data[params.nbU * params.nbNoeuds: -params.nP]

            sol_q  = np.hstack([sol_X[0::params.nbX], sol_X[1::params.nbX], sol_X[2::params.nbX], sol_X[3::params.nbX], sol_X[4::params.nbX], sol_X[5::params.nbX]])
            sol_dq = np.hstack([sol_X[6::params.nbX], sol_X[7::params.nbX], sol_X[8::params.nbX], sol_X[9::params.nbX], sol_X[10::params.nbX], sol_X[11::params.nbX]])
            sol_a  = np.hstack([sol_U[0::params.nbU], sol_U[1::params.nbU], sol_U[2::params.nbU], sol_U[3::params.nbU], sol_U[4::params.nbU], sol_U[5::params.nbU],
                               sol_U[6::params.nbU], sol_U[7::params.nbU], sol_U[8::params.nbU], sol_U[9::params.nbU], sol_U[10::params.nbU],
                               sol_U[11::params.nbU], sol_U[12::params.nbU], sol_U[13::params.nbU], sol_U[14::params.nbU], sol_U[15::params.nbU],
                               sol_U[16::params.nbU]])
            sol_F  = np.hstack([sol_U[17::params.nbU], sol_U[18::params.nbU], sol_U[19::params.nbU]])

            # CONVERGENCE
            JR = Je = Jm = Ja = constraint = 0
            GRF = np.zeros((3, params.nbNoeuds))

            for k in range(params.nbNoeuds_stance):
                uk  = np.array(sol_U[params.nbU * k : params.nbU * (k + 1)])
                xk  = np.array(sol_X[params.nbX * k : params.nbX * (k + 1)])
                xk1 = np.array(sol_X[params.nbX * (k + 1) : params.nbX * (k + 2)])

                # GROUND REACTION FORCES
                activations = uk[: params.nbMus]
                F           = uk[params.nbMus:]
                Q           = xk[:params.nbQ()]
                dQ          = xk[params.nbQ(): 2 * params.nbQ()]

                states = biorbd.VecBiorbdMuscleStateDynamics(params.nbMus)
                n_muscle = 0
                for state in states:
                    state.setActivation(activations[n_muscle])  # Set muscles activations
                    n_muscle += 1

                joint_torque    = params.model_stance.muscularJointTorque(activations, Q, dQ)
                joint_torque[0] = F[0]
                joint_torque[1] = F[1]
                joint_torque[2] = F[2]

                C = m.getConstraints()
                m.ForwardDynamicsConstraintsDirect(Q, dQ, joint_torque, C)
                GRF[:, k] = C.getForce().to_array()

                # OBJECTIF
                JR += params.wR * ((GRF[0, k] - GRF_real[1, k]) * (GRF[0, k] - GRF_real[1, k]))
                JR += params.wR * ((GRF[2, k] - GRF_real[2, k]) * (GRF[2, k] - GRF_real[2, k]))

                Ja += fcn_objective_activation(params.wL, uk)
                Je += fcn_objective_emg(params.wU, uk)

                markers = params.model_stance.markers(xk[:params.nbQ])
                for nMark in range(params.nbMarkers):
                    marker = markers[nMark].to_array()
                    if params.model_stance.marker(nMark).isAnatomical():
                        Jm += params.wMa * ((marker[0] - M_real[0, nMark]) * (marker[0] - M_real[0, nMark]))    # x
                        Jm += params.wMa * ((marker[2] - M_real[2, nMark]) * (marker[2] - M_real[2, nMark]))    # z
                    else:
                        Jm += params.wMt * ((marker[0] - M_real[0, nMark]) * (marker[0] - M_real[0, nMark]))
                        Jm += params.wMt * ((marker[2] - M_real[2, nMark]) * (marker[2] - M_real[2, nMark]))

                # CONTRAINTES
                constraint += xk1 - int_RK4(fcn_dyn_contact, params.T_stance, params.nbNoeuds_stance, xk, uk)

            for k in range(params.nbNoeuds_swing):
                uk  = np.array(sol_U[params.nbU * k : params.nbU * (k + 1)])
                xk  = np.array(sol_X[params.nbX * k : params.nbX * (k + 1)])
                xk1 = np.array(sol_X[params.nbX * (k + 1) : params.nbX * (k + 2)])

                # CONSTRAINT
                constraint += xk1 - int_RK4(fcn_dyn_nocontact, params.T_swing, params.nbNoeuds_swing, xk, uk)

                # OBJECTIF
                Ja += fcn_objective_activation(params.wL, uk)
                Je += fcn_objective_emg(params.wU, uk)

                markers = params.model_swing.markers(xk[:params.nbQ])
                for nMark in range(params.nbMarkers):
                    marker = markers[nMark].to_array()
                    if params.model_swing.marker(nMark).isAnatomical():
                        Jm += params.wMa * ((marker[0] - M_real[0, nMark]) * (marker[0] - M_real[0, nMark]))    # x
                        Jm += params.wMa * ((marker[2] - M_real[2, nMark]) * (marker[2] - M_real[2, nMark]))    # z
                    else:
                        Jm += params.wMt * ((marker[0] - M_real[0, nMark]) * (marker[0] - M_real[0, nMark]))
                        Jm += params.wMt * ((marker[2] - M_real[2, nMark]) * (marker[2] - M_real[2, nMark]))

            J = Ja + Je + Jm + JR

            # PRINT VALUES
            print('\n \nGlobal                 : ' + str(J))
            print('activation             : ' + str(Ja))
            print('emg                    : ' + str(Je))
            print('marker                 : ' + str(Jm))
            print('ground reaction forces : ' + str(JR))
            print('constraints            : ' + str(sum(constraint)) + '\n')


            def plot_control_update(ax, t, x):
                nbPoints = len(np.array(x))
                for n in range(nbPoints - 1):
                    lines = ax.get_lines()
                    if size(lines) > 52:
                        lines[4 + n].set_xdata([t[n], t[n + 1], t[n + 1]])
                        lines[4 + n].set_ydata([x[n], x[n], x[n + 1]])
                    else:
                        lines[3 + n].set_xdata([t[n], t[n + 1], t[n + 1]])
                        lines[3 + n].set_ydata([x[n], x[n], x[n + 1]])

            # CONTROL
            axes1 = plt.figure(1).axes
            for i in range(params.nbMus):
                ax = axes1[i]
                plot_control_update(ax, t, sol_a[:, i])
            for j in range(3):
                ax = axes1[i + 1 + j]
                plot_control_update(ax, t, sol_F[:, j])

            # STATE
            axes2 = plt.figure(2).axes
            for q in range(params.nbQ):
                ax1 = axes2[q]
                lines = ax1.get_lines()
                if q != 0 and q != 1:
                    lines[3].set_ydata(sol_q[:, q] * (180 / np.pi))
                else:
                    lines[3].set_ydata(sol_q[:, q])
            for dq in range(params.nbQ):
                ax1 = axes2[q + 1 + dq]
                lines = ax1.get_lines()
                lines[3].set_ydata(sol_dq[:, dq])

            # GRF
            axes3 = plt.figure(3).axes
            ax_ap = axes3[0]
            lines = ax_ap.get_lines()
            lines[2].set_ydata(GRF[0, :])

            ax_v  = axes3[1]
            lines = ax_v.get_lines()
            lines[2].set_ydata(GRF[2, :])

            callback.update_sol = False         # can get new iteration
        plt.draw()                              # update plots
        plt.pause(.001)
