from bioptim import ObjectiveFcn, Node, PenaltyNodes, Axis
from casadi import MX, vertcat
import numpy as np
import biorbd

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

def custom_CoM_position(pn: PenaltyNodes, value: float) -> MX:
    nq = pn.nlp.shape["q"]
    compute_CoM = biorbd.to_casadi_func("CoM", pn.nlp.model.CoM, pn.nlp.q)
    com = compute_CoM(pn.x[0][:nq])
    return com[2] - value

class objective:
    @staticmethod
    def set_minimize_muscle_driven_torque(objective_functions, phase=0):
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE,
                                quadratic=True,
                                node=Node.ALL,
                                weight=1,
                                phase=phase)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_MUSCLES_CONTROL,
                                quadratic=True,
                                node=Node.ALL,
                                weight=10,
                                phase=phase)
    @staticmethod
    def set_objectif_function_fall(objective_functions, position_high, phase=0):
        # --- control minimize --- #
        objective.set_minimize_muscle_driven_torque(objective_functions, phase)

        # --- initial position --- #
        objective_functions.add(ObjectiveFcn.Mayer.TRACK_STATE,
                                quadratic=True,
                                node=Node.START,
                                index=range(len(position_high)),
                                target=np.array(position_high),
                                weight=1000,
                                phase=0)

        # --- com displacement --- #
        objective_functions.add(custom_CoM_position,
                                custom_type=ObjectiveFcn.Mayer,
                                value=-0.3,
                                node=Node.END,
                                quadratic=True,
                                weight=1000,
                                phase=0)

    @staticmethod
    def set_objectif_function_climb(objective_functions, position_high, phase=0):
        # --- control minimize --- #
        objective.set_minimize_muscle_driven_torque(objective_functions, phase)

        # --- initial position --- #
        objective_functions.add(custom_CoM_position,
                                custom_type=ObjectiveFcn.Mayer,
                                value=-0.3,
                                node=Node.START,
                                quadratic=True,
                                weight=1000,
                                phase=phase)

        # --- final position --- #
        objective_functions.add(ObjectiveFcn.Mayer.TRACK_STATE,
                                quadratic=True,
                                node=Node.END,
                                index=range(len(position_high)),
                                target=np.array(position_high),
                                weight=1000,
                                phase=phase)


    @staticmethod
    def set_objectif_function(objective_functions, position_high):
        # --- control minimize --- #
        objective.set_minimize_muscle_driven_torque(objective_functions)

        # --- initial position --- #
        objective_functions.add(ObjectiveFcn.Mayer.TRACK_STATE,
                                quadratic=True,
                                node=Node.START,
                                index=range(len(position_high)),
                                target=np.array(position_high),
                                weight=1000)

        # --- com displacement --- #
        objective_functions.add(custom_CoM_position,
                                custom_type=ObjectiveFcn.Mayer,
                                value=-0.3,
                                node=Node.MID,
                                quadratic=True,
                                weight=1000)

        # --- final position --- #
        objective_functions.add(ObjectiveFcn.Mayer.TRACK_STATE,
                                quadratic=True,
                                node=Node.END,
                                index=range(len(position_high)),
                                target=np.array(position_high),
                                weight=1000)
        return objective_functions

    @staticmethod
    def set_objectif_function_multiphase(objective_functions, position_high):
        # --- fall --- #
        objective.set_objectif_function_fall(objective_functions, position_high, phase=0)
        # --- climb --- #
        objective.set_objectif_function_climb(objective_functions, position_high, phase=1)
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

    @staticmethod
    def set_objectif_function_position_basse_torque_driven(objective_functions, position_high, time_max, time_min):
        nb_q = len(position_high)
        # --- control minimize --- #
        # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE,
        #                         quadratic=True,
        #                         node=Node.ALL,
        #                         weight=0.001)

        # --- initial position --- #
        # objective_functions.add(ObjectiveFcn.Mayer.TRACK_STATE,
        #                         quadratic=True,
        #                         node=Node.START,
        #                         index=range(nb_q),
        #                         target=np.array(position_high).reshape(nb_q, 1),
        #                         weight=1000)
        # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE,
        #                         quadratic=True,
        #                         node=Node.START,
        #                         index=range(nb_q, (2*nb_q)),
        #                         weight=1000)

        # # --- com displacement --- #
        objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE,
                                quadratic=True,
                                node=Node.END,
                                weight=1000)

        # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_PREDICTED_COM_HEIGHT,
        #                         node=Node.END,
        #                         weight=-100)
        # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE,
        #                         quadratic=True,
        #                         node=Node.END,
        #                         index=range(nb_q, (2*nb_q)),
        #                         weight=1000)

        # # --- minimize time --- #
        # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME,
        #                         weight=0.1,
        #                         min_bound=time_min,
        #                         max_bound=time_max,)

        return objective_functions