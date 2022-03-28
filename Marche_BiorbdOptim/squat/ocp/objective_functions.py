from bioptim import ObjectiveFcn, Node, Axis
import numpy as np

def track_kinematic(objective_functions, q_ref, weight, phase):
    objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE,  # track q
                            quadratic=True,
                            key="q",
                            target=q_ref,
                            node=Node.ALL,
                            weight=weight,
                            expand=False,
                            phase=phase)

def track_muscle(objective_functions, activation_ref, weight, phase):
    objective_functions.add(ObjectiveFcn.Lagrange.TRACK_CONTROL,  # track muscle
                            quadratic=True,
                            key="muscles",
                            target=activation_ref[:, :-1],
                            node=Node.ALL,
                            weight=weight,
                            expand=False,
                            phase=phase)

def minimize_tau(objective_functions, weight, phase):
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,  # residual torque
                            quadratic=True,
                            key="tau",
                            weight=weight,
                            node=Node.ALL,
                            expand=False,
                            phase=phase)

def target_com(objective_functions, target, weight, phase):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION,
                            node=Node.END,
                            axes=Axis.Z,
                            quadratic=True,
                            target=target,
                            weight=weight,
                            expand=False,
                            phase=phase)


class objective:
    @staticmethod
    def set_objectif_function(objective_functions, position_high, nb_shooting, muscles=True):
        # --- control minimize --- #
        if muscles:
            minimize_tau(objective_functions, 1, phase=0)
            act = np.zeros((38, nb_shooting + 1))
            track_muscle(objective_functions, act, 10, phase=0)
        else:
            minimize_tau(objective_functions, 0.01, phase=0)

        # --- com displacement --- #
        objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION,
                                node=Node.MID,
                                axes=Axis.Z,
                                weight=1000,
                                expand=False)
        # --- final standing position --- #
        objective_functions.add(ObjectiveFcn.Mayer.TRACK_STATE,
                                key="q",
                                target=position_high,
                                node=Node.END,
                                expand=False,
                                quadratic=True,
                                weight=100)

    @staticmethod
    def set_objectif_function_fall(objective_functions, q_ref, activation_ref=None, muscles=False, phase=0):
        track_kinematic(objective_functions, q_ref, 100, phase)
        minimize_tau(objective_functions, 1, phase)
        if muscles:
            act = activation_ref if activation_ref is not None else np.zeros((38, q_ref.shape[1]))
            track_muscle(objective_functions, act, 10, phase)


    @staticmethod
    def set_objectif_function_climb(objective_functions, q_ref, activation_ref, muscles, phase=1):
        track_kinematic(objective_functions, q_ref, 100, phase)
        minimize_tau(objective_functions, 0.001, phase)
        if muscles:
            act = activation_ref if activation_ref is not None else np.zeros((38, q_ref.shape[1]))
            track_muscle(objective_functions, act, 10, phase)


    @staticmethod
    def set_objectif_function_multiphase(objective_functions, q_ref=None, activation_ref=None, muscles=False):
        objective.set_objectif_function_fall(objective_functions, q_ref[0], activation_ref, muscles, phase=0)
        objective.set_objectif_function_climb(objective_functions, q_ref[1], activation_ref, muscles, phase=1)