from bioptim import QAndQDotBounds, BiMapping
import numpy as np

class bounds:
    @staticmethod
    def set_bounds(model, x_bounds, u_bounds, position_high, mapping=False):
        torque_min, torque_max = -1000, 1000
        activation_min, activation_max = 1e-3, 0.99

        x_bounds.add(bounds=QAndQDotBounds(model))
        x_bounds[0][:len(position_high), 0] = np.array(position_high)
        x_bounds[0][:len(position_high), -1] = np.array(position_high)
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
    def set_bounds_position(model, x_bounds, u_bounds, mapping=False):
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
    def set_bounds_torque_driven(model, x_bounds, u_bounds, position_init):
        torque_min, torque_max = -1000, 1000

        x_bounds.add(bounds=QAndQDotBounds(model))
        x_bounds[0].min[:model.nbQ(), 0] = np.array(position_init).squeeze()
        x_bounds[0].max[:model.nbQ(), 0] = np.array(position_init).squeeze()

        x_bounds[0].min[:model.nbQ(), -1] = np.array(position_init).squeeze()
        x_bounds[0].max[:model.nbQ(), -1] = np.array(position_init).squeeze()

        u_bounds.add(
            [torque_min] * model.nbGeneralizedTorque(),
            [torque_max] * model.nbGeneralizedTorque(),
        )
        return x_bounds, u_bounds

    @staticmethod
    def set_mapping():
        u_mapping = BiMapping([None, None, None, None, None, None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                                         [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
        return u_mapping