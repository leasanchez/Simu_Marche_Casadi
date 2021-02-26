import numpy as np
import biorbd
from Utils_start import utils
from casadi import MX, Function, vertcat




def muscles_tau(model, q, qdot, activation):
    muscles_states = biorbd.VecBiorbdMuscleState(model.nbMuscles())
    for k in range(model.nbMuscles()):
        muscles_states[k].setActivation(activation[k])
    muscles_tau = model.muscularJointTorque(muscles_states, q, qdot).to_mx()
    return muscles_tau

def muscular_torque(model):
    qMX = MX.sym("qMX", model.nbQ(), 1)
    dqMX = MX.sym("dqMX", model.nbQ(), 1)
    aMX = MX.sym("aMX", model.nbMuscles(), 1)
    return Function("MuscleTorque", [qMX, dqMX, aMX],
                    [muscles_tau(model, qMX, dqMX, aMX)],
                    ["qMX", "dqMX", "aMX"], ["Torque"]).expand()

def muscles_forces(model, q, qdot, activation):
    muscles_states = biorbd.VecBiorbdMuscleState(model.nbMuscles())
    for k in range(model.nbMuscles()):
        muscles_states[k].setActivation(activation[k])
    muscles_forces = model.muscleForces(muscles_states, q, qdot).to_mx()
    return muscles_forces

def muscular_force(model):
    qMX = MX.sym("qMX", model.nbQ(), 1)
    dqMX = MX.sym("dqMX", model.nbQ(), 1)
    aMX = MX.sym("aMX", model.nbMuscles(), 1)
    return Function("MuscleTorque", [qMX, dqMX, aMX],
                    [muscles_forces(model, qMX, dqMX, aMX)],
                    ["qMX", "dqMX", "aMX"], ["Forces"]).expand()

def muscles_jac(model):
    qMX = MX.sym("qMX", model.nbQ(), 1)
    return Function("MuscleJac", [qMX], [model.musclesLengthJacobian(qMX).to_mx()],["qMX"], ["momentarm"]).expand()

def force_isometric(model):
    fiso = []
    for nGrp in range(model.nbMuscleGroups()):
        for nMus in range(model.muscleGroup(nGrp).nbMuscles()):
            fiso.append(model.muscleGroup(nGrp).muscle(nMus).characteristics().forceIsoMax().to_mx())
    return vertcat(*fiso)

def get_force_iso(model):
    return Function("Fiso", [], [force_isometric(model)], [], ["fiso"]).expand()



class muscle:
    def __init__(self, ocp, sol):
        self.ocp = ocp
        self.sol = sol
        self.model = ocp.nlp[0].model

        # results data
        self.q = sol.states["q"]
        self.q_dot = sol.states["qdot"]
        self.tau = sol.controls["tau"]
        self.activations = sol.controls["muscles"]

        # muscles
        self.f_iso = self.set_f_iso()

    def set_f_iso(self):
        get_forceIso = get_force_iso(self.model)
        self.f_iso = get_forceIso()['fiso']


