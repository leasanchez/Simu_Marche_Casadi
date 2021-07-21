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
                                    phase=phase)
        else:
            objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
                                    quadratic=True,
                                    weight=0.01,
                                    phase=phase)
        # --- com displacement --- #
        objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_PREDICTED_COM_HEIGHT,
                                node=Node.END,
                                quadratic=True,
                                weight=1000,
                                phase=phase)

    @staticmethod
    def set_objectif_function_climb(objective_functions, muscles, phase=0):
        # --- control minimize --- #
        if muscles:
            objective.set_minimize_muscle_driven_torque(objective_functions, phase)
        else:
            objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE,
                                    quadratic=True,
                                    node=Node.ALL,
                                    weight=0.01,
                                    phase=phase)


    @staticmethod
    def set_objectif_function(objective_functions, position_high):
        # --- control minimize --- #
        objective.set_minimize_muscle_driven_torque(objective_functions)

        # # --- initial position --- #
        # objective_functions.add(ObjectiveFcn.Mayer.TRACK_STATE,
        #                         quadratic=True,
        #                         node=Node.START,
        #                         index=range(len(position_high)),
        #                         target=np.array(position_high),
        #                         weight=1000)

        # --- com displacement --- #
        objective_functions.add(custom_CoM_position,
                                custom_type=ObjectiveFcn.Mayer,
                                value=-0.2,
                                node=Node.MID,
                                quadratic=True,
                                weight=10000)

        # --- final position --- #
        objective_functions.add(ObjectiveFcn.Mayer.TRACK_STATE,
                                quadratic=True,
                                node=Node.END,
                                index=range(len(position_high)),
                                target=np.array(position_high),
                                weight=1)
        objective_functions.add(ObjectiveFcn.Mayer.TRACK_STATE,
                                quadratic=True,
                                node=Node.END,
                                index=range(len(position_high), 2*len(position_high)),
                                weight=1)
        return objective_functions

    @staticmethod
    def set_objectif_function_multiphase(objective_functions, muscles=False):
        objective.set_objectif_function_fall(objective_functions, muscles, phase=0)
        objective.set_objectif_function_climb(objective_functions, muscles, phase=1)
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