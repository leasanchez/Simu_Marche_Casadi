import biorbd
from casadi import *

def ffcn_no_contact(x, u):
    # SET MODEL
    model = biorbd.Model('/home/leasanchez/programmation/Marche_Florent/ModelesS2M/ANsWER_Rleg_6dof_17muscle_0contact.bioMod')

    # INPUT
    Q              = x[:model.nbQ()]                            # states
    dQ             = x[model.nbQ():2 * model.nbQ()]
    activations    = u[: model.nbMuscleTotal()]                 # controls



    # COMPUTE MOTOR JOINT TORQUES
    muscularJointTorque = external('libmuscular_joint_torque', 'libmuscular_joint_torque.so', {'enable_fd': True})

    joint_torque    = muscularJointTorque(activations, Q, dQ)
    joint_torque[0] = u[model.nbMuscleTotal() + 0]           # ajout des forces au pelvis
    joint_torque[1] = u[model.nbMuscleTotal() + 1]
    joint_torque[2] = u[model.nbMuscleTotal() + 2]

    # COMPUTE THE ANGULAR ACCELERATION BY FORWARD DYNAMICS
    Forward_Dynamics_SansContact = external('libforward_dynamics_no_contact', 'libforward_dynamics_no_contact.so', {'enable_fd': True})
    d_xQ                         = Forward_Dynamics_SansContact(Q, dQ, joint_torque)[:2*model.nbQ()]    # q et dq

    # STATE DERIVATIVE
    dX = vertcat(d_xQ)
    return dX



def ffcn_contact(x, u):
    # SET MODEL
    model = biorbd.Model('/home/leasanchez/programmation/Marche_Florent/ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact.bioMod')

    # INPUT
    Q           = x[:model.nbQ()]                            # states
    dQ          = x[model.nbQ():2*model.nbQ()]
    activations = u[: model.nbMuscleTotal()]                 # controls

    # COMPUTE MOTOR JOINT TORQUES
    muscularJointTorque = external('libmuscular_joint_torque', 'libmuscular_joint_torque.so', {'enable_fd': True})

    joint_torque    = muscularJointTorque(activations, Q, dQ)
    joint_torque[0] = u[model.nbMuscleTotal() + 0]           # ajout des forces au pelvis
    joint_torque[1] = u[model.nbMuscleTotal() + 1]
    joint_torque[2] = u[model.nbMuscleTotal() + 2]

    # COMPUTE THE ANGULAR ACCELERATION BY FORWARD DYNAMICS
    Forward_Dynamics_Contact = external('libforward_dynamics_contact', 'libforward_dynamics_contact.so',{'enable_fd': True})
    d_xQ                     = Forward_Dynamics_Contact(Q, dQ, joint_torque)[:2*model.nbQ()]                # q et dq

    # STATE DERIVATIVE
    dX = vertcat(d_xQ)
    return dX