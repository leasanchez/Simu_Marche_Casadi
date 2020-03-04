import biorbd
from casadi import *

def ffcn_no_contact(x, u):
    # SET MODEL
    model = biorbd.Model('/home/leasanchez/programmation/Marche_Florent/ModelesS2M/ANsWER_Rleg_6dof_17muscle_0contact.bioMod')

    # INPUT
    Q              = x[:model.nbQ()]                            # states
    dQ             = x[model.nbQ():2 * model.nbQ()]
    activations    = x[2 * model.nbQ():]
    excitations    = u[: model.nbMuscleTotal()]                 # controls

    # COMPUTE THE FIRST DERIVATIVE OF MUSCLE ACTIVATIONS
    # def timeDerivationActivation(excitations, activations):
    #     activationsDot = []
    #     states = biorbd.VecBiorbdMuscleStateDynamics(model.nbMuscleTotal())
    #     n_muscle = 0
    #     for nGrp in range(model.nbMuscleGroups()):
    #         for nMus in range(model.muscleGroup(nGrp).nbMuscles()):
    #             states[n_muscle].timeDerivativeActivation(excitations[n_muscle], activations[n_muscle], model.muscleGroup(nGrp).muscle(nMus).characteristics()
    #             activationsDot.append()
    #             n_muscle += 1
    #     return vertcat(*activationsDot)
    #
    # timeDerivationActivation = Function('activationDot', [excitations, activations], timeDerivationActivation(excitations, activations)).expand()

    timeDerivativeActivation = external('libtime_Derivative_Activation', 'libtime_Derivative_Activation.so', {'enable_fd': True})
    activationsDot           = timeDerivativeActivation(excitations, activations)


    # COMPUTE MOTOR JOINT TORQUES
    # muscularJointTorque = Function('muscular_joint_torque', [activations, Q, dQ], model.muscularJointTorque(activations, Q, dQ)).expand()
    muscularJointTorque = external('libmuscular_joint_torque', 'libmuscular_joint_torque.so', {'enable_fd': True})

    joint_torque    = muscularJointTorque(activations, Q, dQ)
    joint_torque[0] = u[model.nbMuscleTotal() + 0]           # ajout des forces au pelvis
    joint_torque[1] = u[model.nbMuscleTotal() + 1]
    joint_torque[2] = u[model.nbMuscleTotal() + 2]

    # COMPUTE THE ANGULAR ACCELERATION BY FORWARD DYNAMICS
    # Forward_Dynamics_SansContact = Function('ffcn_no_contact', [Q, dQ, joint_torque], vertcat(dQ, model.ForwardDynamics(Q, dQ, joint_torque))).expand()
    Forward_Dynamics_SansContact = external('libforward_dynamics_no_contact', 'libforward_dynamics_no_contact.so', {'enable_fd': True})
    d_xQ                         = Forward_Dynamics_SansContact(Q, dQ, joint_torque)[:2*model.nbQ()]    # q et dq

    # STATE DERIVATIVE
    dX = vertcat(d_xQ, activationsDot)
    return dX



def ffcn_contact(x, u):
    # SET MODEL
    model = biorbd.Model('/home/leasanchez/programmation/Marche_Florent/ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact.bioMod')

    # INPUT
    Q           = x[:model.nbQ()]                            # states
    dQ          = x[model.nbQ():2*model.nbQ()]
    activations = x[2*model.nbQ():]
    excitations = u[: model.nbMuscleTotal()]                 # controls

    # # COMPUTE ACTIVATION DERIVATIVE
    # def timeDerivationActivation(excitations, activations):
    #     # states   = biorbd.VecBiorbdMuscleStateDynamics(model.nbMuscleTotal())
    #     activationsDot = []
    #     n_muscle = 0
    #     for nGrp in range(model.nbMuscleGroups()):
    #         for nMus in range(model.muscleGroup(nGrp).nbMuscles()):
    #             activationsDot.append(1)   # states[n_muscle].timeDerivativeActivation(excitations[n_muscle], activations[n_muscle], model.muscleGroup(nGrp).muscle(nMus).characteristics())
    #             n_muscle += 1
    #     return vertcat(*activationsDot)
    #
    # # timeDerivationActivation = Function('activationDot', [excitations, activations], timeDerivationActivation(excitations, activations)).expand()

    timeDerivativeActivation = external('libtime_Derivative_Activation_stance', 'libtime_Derivative_Activation_stance.so', {'enable_fd': True})
    activationsDot           = timeDerivativeActivation(excitations, activations)

    # COMPUTE MOTOR JOINT TORQUES
    muscularJointTorque = external('libmuscular_joint_torque_stance', 'libmuscular_joint_torque_stance.so', {'enable_fd': True})
    # muscularJointTorque = Function('muscular_joint_torque', [activations, Q, dQ], model.muscularJointTorque(activations, Q, dQ)).expand()
    joint_torque        = muscularJointTorque(activations, Q, dQ)
    joint_torque[0]     = u[model.nbMuscleTotal() + 0]                             # ajout des forces au pelvis
    joint_torque[1]     = u[model.nbMuscleTotal() + 1]
    joint_torque[2]     = u[model.nbMuscleTotal() + 2]

    # COMPUTE THE ANGULAR ACCELERATION BY FORWARD DYNAMICS
    # Forward_Dynamics_Contact = Function('ffcn_contact', [Q, dQ, joint_torque], vertcat(dQ, model.ForwardDynamicsConstraintsDirect(Q, dQ, joint_torque))).expand()
    Forward_Dynamics_Contact = external('libforward_dynamics_contact', 'libforward_dynamics_contact.so',{'enable_fd': True})
    d_xQ                     = Forward_Dynamics_Contact(Q, dQ, joint_torque)[:2*model.nbQ()]                # q et dq

    # STATE DERIVATIVE
    dX = vertcat(d_xQ, activationsDot)
    return dX