from bioptim import ConstraintFcn, Node
import numpy as np


def get_contact_index(model):
    force_names = [s.to_string() for s in model.contactNames()]
    return [i for i, t in enumerate([s[-1] == "Z" for s in force_names]) if t]

def get_marker_index(model):
    marker_names = [s.to_string() for s in model.markerNames()]
    return [i for i, t in enumerate([s[-3:] == "HEE" for s in marker_names]) if t]

def force_inequality(constraints, model, phase=0):
    contact_z_axes = get_contact_index(model)
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=0,
        max_bound=np.inf,
        node=Node.ALL,
        contact_index=contact_z_axes,
        expand=False,
        phase=phase,
    )

def optimize_time(constraints, phase=0):
    constraints.add(ConstraintFcn.TIME_CONSTRAINT,
                    node=Node.END,
                    min_bound=0.3,
                    max_bound=1.2,
                    phase=phase,
                    expand=False)

def sliding_marker(constraints, model, phase=0):
    marker_idx = get_marker_index(model)
    constraints.add(ConstraintFcn.TRACK_MARKERS_VELOCITY,
                    node=Node.START,
                    marker_index=marker_idx,
                    phase=phase,
                    expand=False)


class constraint:
    @staticmethod
    def set_constraints_fall(constraints, model, phase=0):
        force_inequality(constraints, model, phase) # contact forces
        optimize_time(constraints, phase) # time
        sliding_marker(constraints, model, phase) # sliding marker


    @staticmethod
    def set_constraints_climb(constraints, model, phase=1):
        force_inequality(constraints, model, phase) # contact forces
        optimize_time(constraints, phase) # time


    @staticmethod
    def set_constraints_multiphase(constraints, model):
        constraint.set_constraints_fall(constraints, model, phase=0)
        constraint.set_constraints_climb(constraints, model, phase=1)

    @staticmethod
    def set_constraints(constraints, model):
        force_inequality(constraints, model, phase=0) # contact forces
        optimize_time(constraints, phase=0) # time
        sliding_marker(constraints, model, phase=0) # sliding marker