import biorbd
from casadi import *

class Dynamics:
    @staticmethod
    def Set_parameter_forceiso(model, p):
        n_muscle = 0
        for nGrp in range(model.nbMuscleGroups()):
            for nMus in range(model.muscleGroup(nGrp).nbMuscles()):
                fiso_init = model.muscleGroup(nGrp).muscle(nMus).characteristics().forceIsoMax().to_mx()
                model.muscleGroup(nGrp).muscle(nMus).characteristics().setForceIsoMax(p[n_muscle] * fiso_init)
                n_muscle += 1
        return [0]

    @staticmethod
    def articular_torque(model, activations, Q, dQ):
        states = biorbd.VecBiorbdMuscleStateDynamics(model.nbMuscleTotal())
        for n_muscle in range(model.nbMuscleTotal()):
            states[n_muscle].setActivation(activations[n_muscle])
        joint_torque = model.muscularJointTorque(states, Q, dQ).to_mx()
        return joint_torque

    @staticmethod
    def ffcn_no_contact(x, u, p):
        # SET MODEL
        model = biorbd.Model('/home/leasanchez/programmation/Simu_Marche_Casadi/ModelesS2M/ANsWER_Rleg_6dof_17muscle_0contact.bioMod')

        # FIND THE PARAMETERS P OPTIMISING THE MAXIMUM ISOMETRIC FORCES -- MODEL SWING
        Dynamics.Set_parameter_forceiso(model, p)

        # INPUT
        Q              = x[:model.nbQ()]                            # states
        dQ             = x[model.nbQ():2 * model.nbQ()]
        activations    = u[: model.nbMuscleTotal()]                 # controls
        torque         = u[model.nbMuscleTotal():]

        # COMPUTE MOTOR JOINT TORQUES
        joint_torque = Dynamics.articular_torque(model, activations, Q, dQ)
        joint_torque += torque                  # add residual torques

        # COMPUTE THE ACCELERATION -- FORWARD DYNAMICS
        ddQ = model.ForwardDynamics(Q, dQ, joint_torque).to_mx()

        return vertcat(dQ, ddQ)

    @staticmethod
    def ffcn_contact(x, u, p):
        # SET MODEL
        model = biorbd.Model('/home/leasanchez/programmation/Simu_Marche_Casadi/ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact.bioMod')

        # FIND THE PARAMETERS P OPTIMISING THE MAXIMUM ISOMETRIC FORCES -- MODEL STANCE
        Dynamics.Set_parameter_forceiso(model, p)

        # INPUT
        Q           = x[:model.nbQ()]                            # states
        dQ          = x[model.nbQ():2*model.nbQ()]
        activations = u[: model.nbMuscleTotal()]                 # controls
        torque      = u[model.nbMuscleTotal():]

        # COMPUTE MOTOR JOINT TORQUES
        joint_torque  = Dynamics.articular_torque(model, activations, Q, dQ)
        joint_torque += torque                  # add residual torques

        # COMPUTE THE ACCELERATION -- FORWARD DYNAMICS
        ddQ = model.ForwardDynamicsConstraintsDirect(Q, dQ, joint_torque).to_mx()

        return vertcat(dQ, ddQ)