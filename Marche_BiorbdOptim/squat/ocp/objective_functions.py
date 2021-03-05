from bioptim import ObjectiveFcn, Node, PenaltyNodes
from casadi import MX, vertcat

def sym_forces(pn: PenaltyNodes) -> MX:
    ns = pn.nlp.ns # number of shooting points
    nc = pn.nlp.model.nbContacts() # number of contact forces
    val = []  # init

    # --- compute forces ---
    for n in range(ns):
        force = pn.nlp.contact_forces_func(pn.x[n], pn.u[n], pn.p)  # compute force
        for c in range(int(nc/2)):
            val = vertcat(val, (force[c]**2 - force[c+int(nc/2)]**2))
    return val


class objective:
    @staticmethod
    def set_objectif_function(objective_functions):
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE,
                                quadratic=True,
                                node=Node.ALL,
                                weight=1)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_MUSCLES_CONTROL,
                                quadratic=True,
                                node=Node.ALL,
                                weight=10)
        objective_functions.add(sym_forces,
                                custom_type=ObjectiveFcn.Lagrange,
                                node=Node.ALL,
                                weight=0.1)
        return objective_functions

    @staticmethod
    def set_objectif_function_torque_driven(objective_functions):
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE,
                                quadratic=True,
                                node=Node.ALL,
                                index=(0, 1, 2, 5, 8, 9, 11, 14, 15, 17),
                                weight=0.001)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE,
                                quadratic=True,
                                node=Node.ALL,
                                index=(3, 4, 6, 7, 10, 12, 13, 16),
                                weight=0.01)
        return objective_functions