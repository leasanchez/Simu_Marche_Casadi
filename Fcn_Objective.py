import biorbd

class Fcn_Objective:
    @staticmethod
    def fcn_objective_activation(wL, activation):
        # Minimize muscular excitation of muscle without emg

        # INPUT
        # wL            = weighting factor for muscular activation
        # activation    = controls (muscular activation)

        # OUTPUT
        # Ja            = activation cost

        Ja = wL*(activation[1]*activation[1]) + wL*(activation[2]*activation[2]) + wL*(activation[3]*activation[3]) + wL*(activation[5]*activation[5]) + wL*(activation[6]*activation[6]) + wL*(activation[11]*activation[11]) + wL*(activation[12]*activation[12])
        #              GLUT_MAX2              +               GLUT_MAX3          +              GLUT_MED1           +              GLUT_MED3           +             R_SEMIMEM            +                R_VAS_INT           +    R_VAS_LAT

        return Ja

    @staticmethod
    def fcn_objective_residualtorque(wt, torque):
        # Minimize residual joint torque

        # INPUT
        # wt            = weighting factor for residual joint torque (should be high)
        # torque        = controls (residual joint torques)

        # OUTPUT
        # Jt            = residual joint torque cost

        Jt = wt * (torque[3] * torque[3]) + wt * (torque[4] * torque[4]) + wt * (torque[5] * torque[5])
        return Jt

    @staticmethod
    def fcn_objective_emg(wU, activation, U_real):
        # Tracking muscular excitation for muscle with emg

        # INPUT
        # wU            = weighting factor for muscular excitation
        # activation    = controls (muscular activation)
        # U_real        = measured muscular excitations

        # OUTPUT
        # Je            = cost of the difference between real and simulated muscular excitation

        Je = wU * ((activation[0] - U_real[9]) * (activation[0] - U_real[9]))           # GLUT_MAX1
        # Je += we*(Uk[1] - U_real[1, k])     # GLUT_MAX2
        # Je += we*(Uk[2] - U_real[2, k])     # GLUT_MAX3
        # Je += we*(Uk[3] - U_real[3, k])     # GLUT_MED1
        Je += wU * ((activation[4] - U_real[8]) * (activation[4] - U_real[8]))          # GLUT_MED2
        # Je += we*(Uk[5] - U_real[5, k])     # GLUT_MED3
        # Je += we*(Uk[6] - U_real[6, k])     # R_SEMIMEM
        Je += wU * ((activation[7] - U_real[7]) * (activation[7] - U_real[7]))          # R_SEMITEN
        Je += wU * ((activation[8] - U_real[6]) * (activation[8] - U_real[6]))          # R_BI_FEM_LH
        Je += wU * ((activation[9] - U_real[5]) * (activation[9] - U_real[5]))          # R_RECTUS_FEM
        Je += wU * ((activation[10] - U_real[4]) * (activation[10] - U_real[4]))        # R_VAS_MED
        # Je += we*(Uk[11] - U_real[11, k])   # R_VAS_INT
        # Je += we*(Uk[12] - U_real[12, k])   # R_VAS_LAT
        Je += wU * ((activation[13] - U_real[3]) * (activation[13] - U_real[3]))        # R_GAS_MED
        Je += wU * ((activation[14] - U_real[2]) * (activation[14] - U_real[2]))        # R_GAS_LAT
        Je += wU * ((activation[15] - U_real[1]) * (activation[15] - U_real[1]))        # R_SOLEUS
        Je += wU * ((activation[16] - U_real[0]) * (activation[16] - U_real[0]))        # R_TIB_ANT
        return Je

    @staticmethod
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
        else:
            # SET MODEL
            model = biorbd.Model('/home/leasanchez/programmation/Simu_Marche_Casadi/ModelesS2M/ANsWER_Rleg_6dof_17muscle_0contact.bioMod')

        markers = model.markers(Q)                                                                                          # markers position
        Jm = 0
        for nMark in range(model.nbMarkers()):
            if model.marker(nMark).isAnatomical():
                Jm += wMa * ((markers[0, nMark] - M_real[0, nMark]) * (markers[0, nMark] - M_real[0, nMark]))               # x
                Jm += wMa * ((markers[2, nMark] - M_real[2, nMark]) * (markers[2, nMark] - M_real[2, nMark]))               # z
            else:
                Jm += wMt * ((markers[0, nMark] - M_real[0, nMark]) * (markers[0, nMark] - M_real[0, nMark]))
                Jm += wMt * ((markers[2, nMark] - M_real[2, nMark]) * (markers[2, nMark] - M_real[2, nMark]))
        return Jm

    @staticmethod
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
        torque      = u[model.nbMuscleTotal():]
        Q           = x[:model.nbQ()]
        dQ          = x[model.nbQ(): 2 * model.nbQ()]

        # SET ISOMETRIC FORCE

        # COMPUTE MOTOR JOINT TORQUES
        # compute joint torque from muscular activation
        states = biorbd.VecBiorbdMuscleStateDynamics(model.nbMuscleTotal())
        for n_muscle in range(model.nbMuscleTotal()):
            states[n_muscle].setActivation(activations[n_muscle])
        joint_torque = model.muscularJointTorque(states, Q, dQ).to_mx()
        # add residual torques
        joint_torque += torque

        # COMPUTE THE GROUND REACTION FORCES
        C = model.getConstraints()
        model.ForwardDynamicsConstraintsDirect(Q, dQ, joint_torque, C)
        GRF = C.getForce().to_mx()

        JR  = wR * ((GRF[0] - GRF_real[1]) * (GRF[0] - GRF_real[1]))         # Fx
        JR += wR * ((GRF[1] - GRF_real[2]) * (GRF[1] - GRF_real[2]))         # Fz
        return GRF, JR

    @staticmethod
    def fcn_objective_GRF_casadi(wR, GRF, GRF_real):
        # Tracking ground reaction forces

        # INPUT
        # wR            = weighting factor for ground reaction forces
        # GRF           = computed GRF from x, u and p
        # GRF_real      = real ground reaction forces from force platform

        # OUTPUT
        # JR            = cost of the difference between real and simulated ground reaction forces

        JR  = wR * ((GRF[0] - GRF_real[1]) * (GRF[0] - GRF_real[1]))         # Fx
        JR += wR * ((GRF[1] - GRF_real[2]) * (GRF[1] - GRF_real[2]))         # Fz
        return JR