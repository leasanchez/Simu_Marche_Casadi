import biorbd
from casadi import *

def forward_dynamics_torque_driven(states, controls, model):
    q = states[:model.nbQ()]
    qdot = states[model.nbQ():]
    tau = controls

    CS = model.getConstraints()
    qddot = biorbd.Model.ForwardDynamicsConstraintsDirect(model, q, qdot, tau, CS).to_mx()
    return vertcat(qdot, qddot, CS.getForce().to_mx())

model_stance = biorbd.Model('/home/leasanchez/programmation/Simu_Marche_Casadi/ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact.bioMod')
states   = MX.sym("x", model_stance.nbQ() * 2, 1)
controls = MX.sym("u", model_stance.nbQ(), 1)
dynamics = casadi.Function("ForwardDyn",
                           [states, controls],
                           [forward_dynamics_torque_driven(states, controls, self.model_stance)],
                           ["states", "controls"],
                           ["statesdot"]).expand()
a = dynamics(states=DM.ones(12, 1), controls=DM.ones(6, 1))
print(a["statesdot"][:6])
print(a["statesdot"][6:12])
print(a["statesdot"][12:])