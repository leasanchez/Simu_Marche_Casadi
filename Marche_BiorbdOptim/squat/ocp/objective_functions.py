from bioptim import ObjectiveFcn, Node, PenaltyNode, Axis
from casadi import MX, vertcat
import numpy as np
import biorbd_casadi as biorbd

def sym_forces(pn: PenaltyNode) -> MX:
    ns = pn.nlp.ns # number of shooting points
    nc = pn.nlp.model.nbContacts() # number of contact forces
    val = []  # init

    # --- compute forces ---
    for n in range(ns):
        force = pn.nlp.contact_forces_func(pn.x[n], pn.u[n], pn.p)  # compute force
        for c in range(int(nc/2)):
            val = vertcat(val, (force[c]**2 - force[c+int(nc/2)]**2))
    return val

def custom_CoM_position(pn: PenaltyNode) -> MX:
    compute_CoM = pn.nlp.add_casadi_func("CoM", pn.nlp.model.CoM, pn.nlp.states["q"].mx)
    com = compute_CoM(pn["q"])
    return com[2]

class objective:
    @staticmethod
    def set_objectif_function_fall(objective_functions, muscles, phase=0):
        # --- control minimize --- #
        if muscles:
            objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, # residual torque
                                    quadratic=True,
                                    key="tau",
                                    node=Node.ALL,
                                    weight=1,
                                    expand=False,
                                    phase=phase)
            objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, # muscles
                                    quadratic=True,
                                    key="muscles",
                                    node=Node.ALL,
                                    weight=10,
                                    expand=False,
    @staticmethod
    def set_objectif_function_fall(objective_functions, muscles, phase=0):
        # --- control minimize --- #
        if muscles:
            objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, # residual torque
                                    quadratic=True,
                                    key="tau",
                                    weight=1/100,
                                    expand=False,
                                    phase=phase)
            objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,  # muscles
                                    quadratic=True,
                                    key="muscles",
                                    weight=1/100,
                                    expand=False,
                                    phase=phase)
        else:
            objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,  # residual torque
                                    quadratic=True,
                                    key="tau",
                                    weight=1/100,
                                    expand=False,
                                    phase=phase)
        # --- com displacement --- #
        objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION,
                                node=Node.END,
                                axes=Axis.Z,
                                quadratic=False,
                                weight=10,
                                expand=False,
                                phase=phase)

    @staticmethod
    def set_objectif_function_climb(objective_functions, position_high, muscles, phase=0):
        # --- control minimize --- #
        if muscles:
            objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,  # residual torque
                                    quadratic=True,
                                    key="tau",
                                    weight=1,
                                    expand=False,
                                    phase=phase)
            objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,  # muscles
                                    quadratic=True,
                                    key="muscles",
                                    weight=1,
                                    expand=False,
                                    phase=phase)
        else:
            objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,  # residual torque
                                    quadratic=True,
                                    key="tau",
                                    weight=1/100,
                                    expand=False,
                                    phase=phase)

        # --- final position --- #
        objective_functions.add(ObjectiveFcn.Mayer.TRACK_STATE,
                                key="q",
                                target=position_high,
                                node=Node.END,
                                expand=False,
                                quadratic=True,
                                weight=100,
                                phase=phase)


    @staticmethod
    def set_objectif_function_multiphase(objective_functions, position_high, muscles=False):
        objective.set_objectif_function_fall(objective_functions, muscles, phase=0)
        objective.set_objectif_function_climb(objective_functions, position_high, muscles, phase=1)