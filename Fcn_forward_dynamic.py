import biorbd
from casadi import *

def ffcn_no_contact(x, u, p):
    # SET MODEL
    model = biorbd.Model('/home/leasanchez/programmation/Simu_Marche_Casadi/ModelesS2M/ANsWER_Rleg_6dof_17muscle_0contact.bioMod')

    # EXTRACT INITIAL MAXIMAL ISOMETRIC FORCES FROM THE MODEL
    FISO0 = np.zeros(model.nbMuscleTotal())
    n_muscle = 0
    for nGrp in range(model.nbMuscleGroups()):
        for nMus in range(model.muscleGroup(nGrp).nbMuscles()):
            FISO0[n_muscle] = model.muscleGroup(nGrp).muscle(nMus).characteristics().forceIsoMax()
            n_muscle += 1

    # FIND THE PARAMETERS P OPTIMISING THE MAXIMUM ISOMETRIC FORCES -- MODEL SWING
    Set_forceISO_max_swing = external('libforce_iso_max', 'libforce_iso_max.so', {'enable_fd': True})
    forceISO = p[0] * p[1:] * FISO0
    Set_forceISO_max_swing(forceISO)

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
    d_xQ                         = Forward_Dynamics_SansContact(Q, dQ, joint_torque)    # q et dq

    # STATE DERIVATIVE
    dX = vertcat(d_xQ)
    return dX



def ffcn_contact(x, u, p):
    # SET MODEL
    model = biorbd.Model('/home/leasanchez/programmation/Simu_Marche_Casadi/ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact.bioMod')

    # EXTRACT INITIAL MAXIMAL ISOMETRIC FORCES FROM THE MODEL
    FISO0 = np.zeros(model.nbMuscleTotal())
    n_muscle = 0
    for nGrp in range(model.nbMuscleGroups()):
        for nMus in range(model.muscleGroup(nGrp).nbMuscles()):
            FISO0[n_muscle] = model.muscleGroup(nGrp).muscle(nMus).characteristics().forceIsoMax()
            n_muscle += 1

    # FIND THE PARAMETERS P OPTIMISING THE MAXIMUM ISOMETRIC FORCES -- MODEL STANCE
    Set_forceISO_max = external('libforce_iso_max_stance', 'libforce_iso_max_stance.so', {'enable_fd': True})
    forceISO         = p[0] * p[1:] * FISO0
    Set_forceISO_max(forceISO)

    # INPUT
    Q           = x[:model.nbQ()]                            # states
    dQ          = x[model.nbQ():2*model.nbQ()]
    activations = u[: model.nbMuscleTotal()]                 # controls

    # COMPUTE MOTOR JOINT TORQUES
    muscularJointTorque = external('libmuscular_joint_torque_stance', 'libmuscular_joint_torque_stance.so', {'enable_fd': True})

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