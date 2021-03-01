import biorbd
import numpy as np
from casadi import vertcat, MX
from bioptim import InitialGuess, Bounds, InterpolationType, Objective, ObjectiveFcn


def force_isometric(model):
    fiso = []
    for nGrp in range(model.nbMuscleGroups()):
        for nMus in range(model.muscleGroup(nGrp).nbMuscles()):
            fiso.append(model.muscleGroup(nGrp).muscle(nMus).characteristics().forceIsoMax().to_mx())
    return vertcat(*fiso)

def modify_isometric_force(model: biorbd.Model, value: MX, fiso_init: list):
    n_muscle = 0
    for nGrp in range(model.nbMuscleGroups()):
        for nMus in range(model.muscleGroup(nGrp).nbMuscles()):
            model.muscleGroup(nGrp).muscle(nMus).characteristics().setForceIsoMax(value[n_muscle] * fiso_init[n_muscle])
            n_muscle += 1


class parameter:
    @staticmethod
    def set_parameters_f_iso(parameters, model):
        bound_f_iso = Bounds([0.2]*model.nbMuscleTotal(),
                             [5]*model.nbMuscleTotal(),
                             interpolation=InterpolationType.CONSTANT)
        initial_fiso= InitialGuess(np.repeat(1, model.nbMuscleTotal()))
        fiso_init = force_isometric(model)
        parameters.add(
            "force_isometrique",  # The name of the parameter
            modify_isometric_force,  # The function that modifies the biorbd model
            initial_fiso,  # The initial guess
            bound_f_iso,  # The bounds
            size=model.nbMuscleTotal(),
            fiso_init=fiso_init,
        )