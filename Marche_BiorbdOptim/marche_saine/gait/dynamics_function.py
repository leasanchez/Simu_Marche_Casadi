import numpy as np
from bioptim import DynamicsFcn

class dynamics:

    @staticmethod
    def set_torque_driven_dynamics(dynamics):
        dynamics.add(DynamicsFcn.TORQUE_DRIVEN_WITH_CONTACT)
        dynamics.add(DynamicsFcn.TORQUE_DRIVEN_WITH_CONTACT)
        dynamics.add(DynamicsFcn.TORQUE_DRIVEN_WITH_CONTACT)
        dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    @staticmethod
    def set_muscle_driven_dynamics(dynamics):
        dynamics.add(DynamicsFcn.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN_WITH_CONTACT)
        dynamics.add(DynamicsFcn.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN_WITH_CONTACT)
        dynamics.add(DynamicsFcn.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN_WITH_CONTACT)
        dynamics.add(DynamicsFcn.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN)