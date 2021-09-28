from bioptim import ObjectiveFcn, Node, Axis

class objective:
    @staticmethod
    def set_objectif_function_exp(objective_functions, q_ref, markers_ref=None, activations_ref=None):
        # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", indices=range(6), weight=0.001, expand=False)
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_CONTROL, key='muscles', target=activations_ref[:14, :-2], weight=1, expand=True)
        # objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="qdot", weight=0.001)
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="q", target=q_ref[:, :-1], node=Node.ALL, weight=1, expand=True)
        # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, axes=Axis.Z, node=Node.MID, expand=True)

    @staticmethod
    def set_objectif_function(objective_functions, position_high, position_low, muscles=True):
        # --- control minimize --- #
        if muscles:
            objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,  # residual torque
                                    quadratic=True,
                                    key="tau",
                                    node=Node.ALL,
                                    weight=1,
                                    expand=False)
            objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,  # muscles
                                    quadratic=True,
                                    key="muscles",
                                    node=Node.ALL,
                                    weight=10,
                                    expand=False)
        else:
            objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,  # residual torque
                                    quadratic=True,
                                    key="tau",
                                    node=Node.ALL,
                                    weight=0.1,
                                    expand=False)

        # --- com displacement --- #
        objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION,
                                node=Node.MID,
                                axes=Axis.Z,
                                weight=1000,
                                expand=False)
        # objective_functions.add(ObjectiveFcn.Mayer.TRACK_STATE,
        #                         key="q",
        #                         target=position_low,
        #                         node=Node.MID,
        #                         expand=False,
        #                         quadratic=True,
        #                         weight=100)

        objective_functions.add(ObjectiveFcn.Mayer.TRACK_STATE,
                                key="q",
                                target=position_high,
                                node=Node.END,
                                expand=False,
                                quadratic=True,
                                weight=100)

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