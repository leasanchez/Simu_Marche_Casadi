import numpy as np
from bioptim import InterpolationType, InitialGuessList, OptimalControlProgram

class initial_guess:
    @staticmethod
    def set_initial_guess(model, x_init, u_init, q_ref, muscles=False, activation_ref=None, mapping=False):
        nb_mus = model.nbMuscleTotal() if muscles else 0
        nb_tau = (model.nbGeneralizedTorque() - model.nbRoot()) if mapping else model.nbGeneralizedTorque()

        qdot_ref = np.gradient(q_ref)[0]
        # x_init.add([0]*model.nbQ() + [0]*model.nbQ())
        x_init.add(np.vstack([q_ref, qdot_ref]), interpolation=InterpolationType.EACH_FRAME)
        if (activation_ref is not None) and (nb_mus != 0):
            tau_ref = np.zeros((nb_tau, q_ref.shape[1]))
            u_init.add(np.vstack([tau_ref[:, :-1], activation_ref[:, :-1]]), interpolation=InterpolationType.EACH_FRAME)
        else:
            u_init.add([0] * nb_tau + [0.5] * nb_mus)


    @staticmethod
    def set_initial_guess_multiphase(model, x_init, u_init, q_ref, muscles=True, activation_ref=[None, None], mapping=False):
        initial_guess.set_initial_guess(model[0], x_init, u_init, q_ref[0], muscles, activation_ref[0], mapping)
        initial_guess.set_initial_guess(model[1], x_init, u_init,  q_ref[1], muscles, activation_ref[1], mapping)

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