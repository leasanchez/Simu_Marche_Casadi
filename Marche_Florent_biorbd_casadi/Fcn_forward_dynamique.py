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
    muscularJointTorque       = Function('muscular_joint_torque', [activations, Q, dQ], model.muscularJointTorque(activations, Q, dQ)).expand()
    u[model.nbMuscleTotal():] = muscularJointTorque(activations, Q, dQ)

    # COMPUTE THE ANGULAR ACCELERATION BY FORWARD DYNAMICS
    Forward_Dynamics_SansContact = Function('ffcn_no_contact', [Q, dQ, u[model.nbMuscleTotal():]], vertcat(dQ, model.ForwardDynamics(Q, dQ, u[model.nbMuscleTotal():]))).expand()
    d_xQ                         = Forward_Dynamics_SansContact(Q, dQ, u[model.nbMuscleTotal():])

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
    muscularJointTorque = Function('muscular_joint_torque', [activations, Q, dQ], model.muscularJointTorque(activations, Q, dQ)).expand()
    u[model.nbMuscleTotal():] = muscularJointTorque(activations, Q, dQ)

    # COMPUTE THE ANGULAR ACCELERATION BY FORWARD DYNAMICS
    Forward_Dynamics_Contact = Function('ffcn_no_contact', [Q, dQ, u[model.nbMuscleTotal():]], vertcat(dQ, model.ForwardDynamicsConstraintsDirect(Q, dQ, u[model.nbMuscleTotal():]))).expand()
    d_xQ                     = Forward_Dynamics_Contact(Q, dQ, u[model.nbMuscleTotal():])

    # STATE DERIVATIVE
    dX = vertcat(d_xQ)
    return dX