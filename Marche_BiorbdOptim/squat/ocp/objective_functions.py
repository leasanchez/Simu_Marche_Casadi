from bioptim import ObjectiveFcn, Node, PenaltyNodes
from casadi import MX, vertcat

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
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_MUSCLES_CONTROL,
                                quadratic=True,
                                node=Node.ALL,
                                weight=10)
        return objective_functions