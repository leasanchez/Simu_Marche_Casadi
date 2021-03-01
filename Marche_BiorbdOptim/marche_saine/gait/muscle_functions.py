import numpy as np
import biorbd
from casadi import MX, Function, vertcat




def muscles_tau(model, q, qdot, activation):
    muscles_states = biorbd.VecBiorbdMuscleState(model.nbMuscles())
    for k in range(model.nbMuscles()):
        muscles_states[k].setActivation(activation[k])
    muscles_tau = model.muscularJointTorque(muscles_states, q, qdot).to_mx()
    return muscles_tau

def get_muscular_torque(model):
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

def get_muscular_force(model):
    qMX = MX.sym("qMX", model.nbQ(), 1)
    dqMX = MX.sym("dqMX", model.nbQ(), 1)
    aMX = MX.sym("aMX", model.nbMuscles(), 1)
    return Function("MuscleTorque", [qMX, dqMX, aMX],
                    [muscles_forces(model, qMX, dqMX, aMX)],
                    ["qMX", "dqMX", "aMX"], ["Forces"]).expand()

def get_muscles_jac(model):
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

def get_muscle_length(model):
    qMX = MX.sym("qMX", model.nbQ(), 1)
    muscle_len = []
    for m in range(model.nbMuscleTotal()):
        muscle_len.append(Function(f"MuscleLen{m}", [qMX], [model.muscle(m).length(model, qMX).to_mx()], ["qMX"], [f"muscle_len{m}"]).expand())
    return muscle_len


class muscle:
    def __init__(self, ocp, sol):
        self.ocp = ocp
        self.sol = sol.merge_phases()

        # results data
        self.model=self.ocp.nlp[0].model
        self.number_shooting_points=self.ocp.nlp.ns
        self.q=sol.states["q"]
        self.qdot=sol.states["qdot"]
        self.tau=sol.controls["tau"]
        self.activations=sol.controls["muscles"]
        self.n_shooting = self.q.shape[1] - 1

        # model params
        self.n_q = self.model.nbQ()
        self.n_mus = self.model.nbMuscleTotal()

        # muscles
        self.muscle_name = self.get_muscle_name()
        self.f_iso = self.set_f_iso()
        self.muscle_force = self.compute_muscle_force()
        self.muscle_length = self.compute_muscle_length()
        self.muscular_torque = self.compute_muscular_torque()
        self.muscle_jacobian = self.compute_muscle_jacobian()
        self.individual_muscle_torque = self.compute_individual_muscle_torque()
        self.individual_muscle_power = self.compute_individual_muscle_power()


    def get_muscle_name(self):
        muscle_name = []
        for m in range(self.n_mus):
            muscle_name.append(self.model.muscle(m).name().to_string())
        return muscle_name

    def set_f_iso(self):
        f_iso_func = get_force_iso(self.model)
        return f_iso_func['fiso']

    def compute_muscular_torque(self):
        muscular_torque = np.zeros((self.n_q, self.n_shooting + 1))
        muscle_tau_func = get_muscular_torque(self.model)
        for n in range(self.n_shooting + 1):
            muscular_torque[:, n] = muscle_tau_func(self.q[:, n], self.qdot[:, n], self.activations[:, n])
        return muscular_torque

    def compute_muscle_force(self):
        muscle_force = np.zeros((self.n_mus, self.n_shooting + 1))
        muscle_force_func = get_muscular_force(self.model)
        for n in range(self.n_shooting + 1):
            muscle_force[:, n]=muscle_force_func(self.q[:, n], self.qdot[:, n], self.activations[:, n])
        return muscle_force

    def compute_muscle_length(self):
        muscle_length = np.zeros((self.n_mus, self.n_shooting + 1))
        muscle_length_func = get_muscle_length(self.model)
        for n in range(self.n_shooting + 1):
            for m in range(self.n_mus):
                muscle_length[m, n]=muscle_length_func[m](self.q[:, n])
        return muscle_length

    def compute_muscle_jacobian(self):
        muscle_jacobian = np.zeros((self.n_mus, self.n_q, self.n_shooting + 1))
        muscle_jacobian_func = get_muscles_jac(self.model)
        for n in range(self.n_shooting + 1):
            muscle_jacobian[:, :, n] = muscle_jacobian_func(self.q[:, n])
        return muscle_jacobian

    def compute_individual_muscle_torque(self):
        individual_muscle_torque = np.zeros((self.n_mus, self.n_q, self.n_shooting + 1))
        for m in range(self.n_mus):
            for i in range(self.n_q):
                individual_muscle_torque[m, i, :] = -self.muscle_jacobian[m, i, :] * self.muscle_force[m, :]
        return individual_muscle_torque

    def compute_individual_muscle_power(self):
        individual_muscle_power = np.zeros((self.n_mus, self.n_q, self.n_shooting + 1))
        for m in range(self.n_mus):
            for i in range(self.n_q):
                individual_muscle_power[m, i, :] = -self.muscle_jacobian[m, i, :] * self.muscle_force[m, :] * self.qdot[i, :]
        return individual_muscle_power


