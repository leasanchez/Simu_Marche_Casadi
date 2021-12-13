from bioptim import QAndQDotBounds, BiMapping

class bounds:
    @staticmethod
    def set_bounds(model, x_bounds, u_bounds, muscles=False, mapping=False):
        torque_min, torque_max = -1000, 1000
        activation_min, activation_max = 1e-3, 0.99

        nb_mus = model.nbMuscleTotal() if muscles else 0
        nb_tau = (model.nbGeneralizedTorque() - model.nbRoot()) if mapping else model.nbGeneralizedTorque()

        x_bounds.add(bounds=QAndQDotBounds(model))
        u_bounds.add([torque_min] * nb_tau + [activation_min] * nb_mus,
                     [torque_max] * nb_tau + [activation_max] * nb_mus,)


    @staticmethod
    def set_mapping():
        u_mapping = BiMapping([None, None, None, None, None, None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                                         [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
        return u_mapping