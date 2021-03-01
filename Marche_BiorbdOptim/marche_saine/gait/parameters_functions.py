import biorbd
import numpy as np
from casadi import vertcat, Function
from bioptim import InitialGuess, Bounds, InterpolationType


def force_isometric(model):
    fiso = []
    for nGrp in range(model.nbMuscleGroups()):
        for nMus in range(model.muscleGroup(nGrp).nbMuscles()):
            fiso.append(model.muscleGroup(nGrp).muscle(nMus).characteristics().forceIsoMax().to_mx())
    return vertcat(*fiso)

def get_force_iso_init(model):
    f_iso_func= Function("Fiso", [], [force_isometric(model)], [], ["fiso"]).expand()
    return f_iso_func['fiso']

def modify_isometric_force(model: biorbd.Model, fiso_init: list, value: np.ndarray):
    n_muscle = 0
    for nGrp in range(model.nbMuscleGroups()):
        for nMus in range(model.muscleGroup(nGrp).nbMuscles()):
            model.muscleGroup(nGrp).muscle(nMus).characteristics().setForceIsoMax(value[n_muscle] * fiso_init)
            n_muscle += 1


class parameter:
    
    @staticmethod
    def set_parameters_f_iso(parameters, model):
        bound_f_iso = Bounds(min_bound=np.repeat(0.2, model.nbMuscleTotal()),
                             max_bound=np.repeat(5, model.nbMuscleTotal()),
                             interpolation_type=InterpolationType.CONSTANT)

        initial_fiso= InitialGuess(np.repeat(1, model.nbMuscleTotal()))
        fiso_init = get_force_iso_init(model)
        parameters.add(
            "force_isometrique",  # The name of the parameter
            modify_isometric_force,  # The function that modifies the biorbd model
            initial_fiso,  # The initial guess
            bound_f_iso,  # The bounds
            size=model.nbMuscleTotal(),
            fiso_init=fiso_init,
        )