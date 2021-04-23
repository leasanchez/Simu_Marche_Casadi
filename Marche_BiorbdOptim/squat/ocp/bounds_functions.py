from bioptim import QAndQDotBounds, BiMapping
import numpy as np

class bounds:
    @staticmethod
    def set_bounds(model, x_bounds, u_bounds, mapping=False):
        torque_min, torque_max = -1000, 1000
        activation_min, activation_max = 1e-3, 0.99

        x_bounds.add(bounds=QAndQDotBounds(model))
        if mapping:
            u_bounds.add(
                [torque_min] * (model.nbGeneralizedTorque() - model.nbRoot()) + [activation_min] * model.nbMuscleTotal(),
                [torque_max] * (model.nbGeneralizedTorque() - model.nbRoot()) + [activation_max] * model.nbMuscleTotal(),
            )
        else:
            u_bounds.add(
                [torque_min] * model.nbGeneralizedTorque() + [activation_min] * model.nbMuscleTotal(),
                [torque_max] * model.nbGeneralizedTorque() + [activation_max] * model.nbMuscleTotal(),
            )
        return x_bounds, u_bounds

    @staticmethod
    def set_bounds_torque_driven(model, position_init, x_bounds, u_bounds):
        torque_min, torque_max = -1000, 1000
        n_phases = 2
        for i in range(n_phases):
            x_bounds.add(bounds=QAndQDotBounds(model))

            u_bounds.add(
                [torque_min] * model.nbGeneralizedTorque(),
                [torque_max] * model.nbGeneralizedTorque(),
            )
        x_bounds[0][:model.nbQ(), 0] = position_init
        x_bounds[1][:model.nbQ(), -1] = position_init
        return x_bounds, u_bounds

    @staticmethod
    def set_mapping():
        u_mapping = BiMapping([None, None, None, None, None, None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                                         [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
        return u_mapping