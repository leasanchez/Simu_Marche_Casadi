from bioptim import QAndQDotBounds
import numpy as np

class bounds:
    @staticmethod
    def set_bounds(model, x_bounds, u_bounds, position_init):
        torque_min, torque_max = -1000, 1000
        activation_min, activation_max = 1e-3, 1.0

        x_bounds.add(bounds=QAndQDotBounds(model))
        x_bounds[0].min[:model.nbQ(), 0] = np.array(position_init).squeeze()
        x_bounds[0].max[:model.nbQ(), 0] = np.array(position_init).squeeze()

        x_bounds[0].min[:model.nbQ(), -1] = np.array(position_init).squeeze()
        x_bounds[0].max[:model.nbQ(), -1] = np.array(position_init).squeeze()

        u_bounds.add(
            [torque_min] * model.nbGeneralizedTorque() + [activation_min] * model.nbMuscleTotal(),
            [torque_max] * model.nbGeneralizedTorque() + [activation_max] * model.nbMuscleTotal(),
        )
        return x_bounds, u_bounds