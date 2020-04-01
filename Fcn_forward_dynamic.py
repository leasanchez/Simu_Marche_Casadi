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
            model.muscleGroup(nGrp).muscle(nMus).characteristics().setforceisomax(p[n_muscle] * FISO0[n_muscle])
            n_muscle += 1
    return [0]

def articular_torque(model, activations, Q, dQ):
    states = biorbd.VecBiorbdMuscleStateDynamics(model.nbMuscleTotal())
    for n_muscle in range(model.nbMuscleTotal()):
        states[n_muscle].setActivation(activations[n_muscle])
    joint_torque = model.muscularJointTorque(states, Q, dQ).to_mx()
    return joint_torque

def ffcn_no_contact(x, u, p):
    # SET MODEL
    model = biorbd.Model('/home/leasanchez/programmation/Simu_Marche_Casadi/ModelesS2M/ANsWER_Rleg_6dof_17muscle_0contact.bioMod')

    # EXTRACT INITIAL MAXIMAL ISOMETRIC FORCES FROM THE MODEL
    # FISO0 = Get_initial_forceiso(model)

    # FIND THE PARAMETERS P OPTIMISING THE MAXIMUM ISOMETRIC FORCES -- MODEL SWING
    # Set_parameter_forceiso(model, FISO0, p)

    # INPUT
    Q              = x[:model.nbQ()]                            # states
    dQ             = x[model.nbQ():2 * model.nbQ()]
    activations    = u[: model.nbMuscleTotal()]                 # controls
    torque         = u[model.nbMuscleTotal():]

    # COMPUTE MOTOR JOINT TORQUES
    joint_torque = articular_torque(model, activations, Q, dQ)
    joint_torque += torque                  # add residual torques

    # COMPUTE THE ANGULAR ACCELERATION BY FORWARD DYNAMICS
    ddQ = model.ForwardDynamics(Q, dQ, joint_torque).to_mx()

    return vertcat(dQ, ddQ)



def ffcn_contact(x, u, p):
    # SET MODEL
    model = biorbd.Model('/home/leasanchez/programmation/Simu_Marche_Casadi/ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact.bioMod')

    # EXTRACT INITIAL MAXIMAL ISOMETRIC FORCES FROM THE MODEL
    # FISO0 = Get_initial_forceiso(model)

    # FIND THE PARAMETERS P OPTIMISING THE MAXIMUM ISOMETRIC FORCES -- MODEL STANCE
    # Set_parameter_forceiso(model, p)

    # INPUT
    Q           = x[:model.nbQ()]                            # states
    dQ          = x[model.nbQ():2*model.nbQ()]
    activations = u[: model.nbMuscleTotal()]                 # controls
    torque      = u[model.nbMuscleTotal():]

    # COMPUTE MOTOR JOINT TORQUES
    joint_torque = articular_torque(model, activations, Q, dQ)
    joint_torque += torque                  # add residual torques

    # COMPUTE THE ANGULAR ACCELERATION BY FORWARD DYNAMICS
    ddQ = model.ForwardDynamicsConstraintsDirect(Q, dQ, joint_torque).to_mx()

    return vertcat(dQ, ddQ)