import biorbd
from casadi import *

def Get_initial_forceiso(model):
    FISO0 = np.zeros(model.nbMuscleTotal())
    n_muscle = 0
    for nGrp in range(model.nbMuscleGroups()):
        for nMus in range(model.muscleGroup(nGrp).nbMuscles()):
            FISO0[n_muscle] = model.muscleGroup(nGrp).muscle(nMus).characteristics().forceIsoMax()
            n_muscle += 1
    return FISO0

def Set_parameter_forceiso(model, FISO0, p):
    n_muscle = 0
    for nGrp in range(model.nbMuscleGroups()):
        for nMus in range(model.muscleGroup(nGrp).nbMuscles()):
            model.muscleGroup(nGrp).muscle(nMus).characteristics().setforceisomax(p[0] * p[n_muscle] * FISO0)
            n_muscle += 1
    return [0]

def articular_torque(model, activations, Q, dQ):
    states = biorbd.VecBiorbdMuscleStateDynamics(model.nbMuscleTotal())
    for n_muscle in range(model.nbMuscleTotal()):
        states[n_muscle].setActivation(activations[n_muscle])
    joint_torque = model.muscularJointTorque(activations, Q, dQ)
    return joint_torque

def ffcn_no_contact(x, u, p):
    # SET MODEL
    model = biorbd.Model('/home/leasanchez/programmation/Simu_Marche_Casadi/ModelesS2M/ANsWER_Rleg_6dof_17muscle_0contact.bioMod')

    # EXTRACT INITIAL MAXIMAL ISOMETRIC FORCES FROM THE MODEL
    FISO0 = Get_initial_forceiso(model)

    # FIND THE PARAMETERS P OPTIMISING THE MAXIMUM ISOMETRIC FORCES -- MODEL SWING
    Set_parameter_forceiso(model, FISO0, p)
    # Set_forceISO_max_swing = external('libforce_iso_max', 'libforce_iso_max.so', {'enable_fd': True})
    # forceISO = p[0] * p[1:] * FISO0
    # Set_forceISO_max_swing(forceISO)

    # INPUT
    Q              = x[:model.nbQ()]                            # states
    dQ             = x[model.nbQ():2 * model.nbQ()]
    activations    = u[: model.nbMuscleTotal()]                 # controls

    # COMPUTE MOTOR JOINT TORQUES
    muscularJointTorque = Function('muscular_joint_torque', [activations, Q, dQ], articular_torque(model, activations, Q, dQ)).expand()
    # muscularJointTorque = external('libmuscular_joint_torque', 'libmuscular_joint_torque.so',{'enable_fd': True})

    joint_torque  = muscularJointTorque(activations, Q, dQ)
    joint_torque += u[model.nbMuscleTotal():]                # add residual torques
    # joint_torque[0] = u[model.nbMuscleTotal() + 0]           # add force pelvis
    # joint_torque[1] = u[model.nbMuscleTotal() + 1]
    # joint_torque[2] = u[model.nbMuscleTotal() + 2]

    # COMPUTE THE ANGULAR ACCELERATION BY FORWARD DYNAMICS
    Forward_Dynamics_SansContact = Function('Forward_Dynamics_SansContact', [Q, dQ, joint_torque], vertcat(dQ, model.ForwardDynamics(Q, dQ, joint_torque))).expand()
    # Forward_Dynamics_SansContact = external('libforward_dynamics_no_contact', 'libforward_dynamics_no_contact.so', {'enable_fd': True})
    d_xQ                         = Forward_Dynamics_SansContact(Q, dQ, joint_torque)    # q et dq

    # STATE DERIVATIVE
    dX = vertcat(d_xQ)
    return dX



def ffcn_contact(x, u, p):
    # SET MODEL
    model = biorbd.Model('/home/leasanchez/programmation/Simu_Marche_Casadi/ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact.bioMod')

    # EXTRACT INITIAL MAXIMAL ISOMETRIC FORCES FROM THE MODEL
    FISO0 = Get_initial_forceiso(model)

    # FIND THE PARAMETERS P OPTIMISING THE MAXIMUM ISOMETRIC FORCES -- MODEL STANCE
    Set_parameter_forceiso(model, FISO0, p)
    # Set_forceISO_max = external('libforce_iso_max_stance', 'libforce_iso_max_stance.so', {'enable_fd': True})
    # forceISO         = p[0] * p[1:] * FISO0
    # Set_forceISO_max(forceISO)

    # INPUT
    Q           = x[:model.nbQ()]                            # states
    dQ          = x[model.nbQ():2*model.nbQ()]
    activations = u[: model.nbMuscleTotal()]                 # controls

    # COMPUTE MOTOR JOINT TORQUES
    muscularJointTorque = Function('muscularJointTorque', [activations, Q, dQ], articular_torque(model, activations, Q, dQ)).expand()
    # muscularJointTorque = external('libmuscular_joint_torque_stance', 'libmuscular_joint_torque_stance.so', {'enable_fd': True})

    joint_torque  = muscularJointTorque(activations, Q, dQ)
    joint_torque += u[model.nbMuscleTotal():]                  # add residual torques
    # joint_torque[0] = u[model.nbMuscleTotal() + 0]           # add pelvis forces
    # joint_torque[1] = u[model.nbMuscleTotal() + 1]
    # joint_torque[2] = u[model.nbMuscleTotal() + 2]

    # COMPUTE THE ANGULAR ACCELERATION BY FORWARD DYNAMICS
    # Forward_Dynamics_Contact = external('libforward_dynamics_contact', 'libforward_dynamics_contact.so',{'enable_fd': True})
    Forward_Dynamics_Contact = Function('Forward_Dynamics_Contact', [Q, dQ, joint_torque], vertcat(dQ, model.ForwardDynamicsConstraintsDirect(Q, dQ, joint_torque))).expand()
    d_xQ                     = Forward_Dynamics_Contact(Q, dQ, joint_torque)

    # STATE DERIVATIVE
    dX = vertcat(d_xQ)
    return dX