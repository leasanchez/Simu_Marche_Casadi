import numpy as np
from bioptim import InterpolationType, InitialGuessList

class initial_guess:
    @staticmethod
    def set_initial_guess_fall(model, x_init, u_init, position_high, position_low, nb_shooting, mapping=False):
        init_x = np.zeros((model.nbQ() + model.nbQdot(), nb_shooting + 1))
        for i in range(model.nbQ()):
            init_x[i, :] = np.concatenate(np.linspace(position_high[i], position_low[i], nb_shooting + 1)).squeeze()
        init_x[model.nbQ():, :] = np.gradient(init_x[:model.nbQ(), :])[0]
        x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)

        if mapping:
            u_init.add([0] * (model.nbGeneralizedTorque() - model.nbRoot()) + [0.1] * model.nbMuscleTotal())
        else:
            u_init.add([0] * model.nbGeneralizedTorque() + [0.1] * model.nbMuscleTotal())

    @staticmethod
    def set_initial_guess_climb(model, x_init, u_init, position_high, position_low, nb_shooting, mapping=False):
        init_x = np.zeros((model.nbQ() + model.nbQdot(), nb_shooting + 1))
        for i in range(model.nbQ()):
            init_x[i, :] = np.concatenate(np.linspace(position_low[i], position_high[i], nb_shooting + 1)).squeeze()
        init_x[model.nbQ():, :] = np.gradient(init_x[:model.nbQ(), :])[0]
        x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)

        if mapping:
            u_init.add([0] * (model.nbGeneralizedTorque() - model.nbRoot()) + [0.1] * model.nbMuscleTotal())
        else:
            u_init.add([0] * model.nbGeneralizedTorque() + [0.1] * model.nbMuscleTotal())

    @staticmethod
    def set_initial_guess_multiphase(model, x_init, u_init, position_high, position_low, nb_shooting, mapping=False):
        initial_guess.set_initial_guess_fall(model[0], x_init, u_init, position_high, position_low, nb_shooting[0], mapping)
        initial_guess.set_initial_guess_climb(model[1], x_init, u_init, position_high, position_low, nb_shooting[1], mapping)
        return x_init, u_init

    @staticmethod
    def set_initial_guess(model, x_init, u_init, position_high, position_low, nb_shooting, mapping=False):
        init_x = np.zeros((model.nbQ() + model.nbQdot(), nb_shooting + 1))
        for i in range(model.nbQ()):
            init_x[i, :] = np.concatenate((np.linspace(position_high[i], position_low[i], int(nb_shooting / 2)),
                                           np.linspace(position_low[i], position_high[i],
                                                       int(nb_shooting / 2) + 1))).squeeze()
        init_x[model.nbQ():, :] = np.gradient(init_x[:model.nbQ(), :])[0]
        x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)

        if mapping:
            u_init.add([0] * (model.nbGeneralizedTorque() - model.nbRoot()) + [0.1] * model.nbMuscleTotal())
        else:
            u_init.add([0] * model.nbGeneralizedTorque() + [0.1] * model.nbMuscleTotal())
        return x_init, u_init

    @staticmethod
    def set_initial_guess_position_basse(model, x_init, u_init, position_high, position_low, nb_shooting, mapping=False):
        init_x = np.zeros((model.nbQ() + model.nbQdot(), nb_shooting + 1))
        for i in range(model.nbQ()):
            init_x[i, :] = np.concatenate(np.linspace(position_high[i], position_low[i], nb_shooting + 1)).squeeze()
        init_x[model.nbQ():, :] = np.gradient(init_x[:model.nbQ(), :])[0]
        x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)

        if mapping:
            u_init.add([0] * (model.nbGeneralizedTorque() - model.nbRoot()) + [0.1] * model.nbMuscleTotal())
        else:
            u_init.add([0] * model.nbGeneralizedTorque() + [0.1] * model.nbMuscleTotal())
        return x_init, u_init

    @staticmethod
    def set_initial_guess_position_basse_torque_driven(model, x_init, u_init, position_high, position_low, nb_shooting, mapping=False):
        # init_x = np.zeros((model.nbQ() + model.nbQdot(), nb_shooting + 1))
        # for i in range(model.nbQ()):
        #     init_x[i, :] = np.linspace(position_high[i], position_low[i], nb_shooting + 1).squeeze()
        # init_x[model.nbQ():, :] = np.gradient(init_x[:model.nbQ(), :])[0]
        # x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)
        n_phases = 2
        for i in range(n_phases):
            x_init.add(np.zeros(model.nbQ() + model.nbQ()))
            if mapping:
                u_init.add([0] * (model.nbGeneralizedTorque() - model.nbRoot()))
            else:
                u_init.add([0] * model.nbGeneralizedTorque())
        return x_init, u_init


    @staticmethod
    def set_initial_guess_torque_driven(model, x_init, u_init, position_high, position_low, nb_shooting):
        init_x = np.zeros((model.nbQ() + model.nbQdot(), nb_shooting + 1))
        for i in range(model.nbQ()):
            init_x[i, :] = np.concatenate((np.linspace(position_high[i], position_low[i], int(nb_shooting / 2)),
                                           np.linspace(position_low[i], position_high[i],
                                                       int(nb_shooting / 2) + 1))).squeeze()
        init_x[model.nbQ():, :] = np.gradient(init_x[:model.nbQ(), :])[0]
        x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)

        u_init.add([0] * model.nbGeneralizedTorque())
        return x_init, u_init

    @staticmethod
    def set_initial_guess_from_previous_solution(model, x_init, u_init, save_path, nb_shooting, mapping=False):
        init_x = np.zeros((model.nbQ() + model.nbQdot(), nb_shooting + 1))
        init_x[:model.nbQ(), :] = np.load(save_path + "q.npy")
        init_x[model.nbQ():, :] = np.load(save_path + "qdot.npy")
        x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)

        init_u = np.zeros((model.nbGeneralizedTorque() + model.nbMuscleTotal(), nb_shooting))
        if mapping:
            init_u[:(model.nbGeneralizedTorque() - model.nbRoot()), :] = np.load(save_path + "tau.npy")[6:, :-1]
        else:
            init_u[:model.nbGeneralizedTorque(), :] = np.load(save_path + "tau.npy")[:, :-1]
        init_u[model.nbGeneralizedTorque():, :] = np.load(save_path + "muscle.npy")[:, :-1]
        u_init.add(init_u, interpolation=InterpolationType.EACH_FRAME)
        return x_init, u_init

    @staticmethod
    def set_initial_guess_from_previous_solution_torque_driven(model, x_init, u_init, save_path, nb_shooting):
        init_x = np.zeros((model.nbQ() + model.nbQdot(), nb_shooting + 1))
        init_x[:model.nbQ(), :] = np.load(save_path + "q.npy")
        init_x[model.nbQ():, :] = np.load(save_path + "qdot.npy")
        x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)

        init_u = np.load(save_path + "tau.npy")[:, :-1]
        u_init.add(init_u, interpolation=InterpolationType.EACH_FRAME)
        return x_init, u_init

    @staticmethod
    def set_initial_guess_from_kalman_torque_driven(model, q_ref, nb_shooting):
        qdot_ref = np.gradient(q_ref)[0]
        x_init = InitialGuessList()
        u_init = InitialGuessList()

        x_init.add(np.vstack([q_ref[:, :nb_shooting[0]], qdot_ref[:, :nb_shooting[0]]]),
                   interpolation=InterpolationType.EACH_FRAME)
        x_init.add(np.vstack([q_ref[:, nb_shooting[0]:], qdot_ref[:, nb_shooting[0]:]]),
                   interpolation=InterpolationType.EACH_FRAME)

        init_u = [0] * model.nbQ()
        u_init.add(init_u)
        u_init.add(init_u)
        return x_init, u_init