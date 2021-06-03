import numpy as np
from bioptim import ConstraintFcn, Node, PenaltyNode
from casadi import vertcat, MX


# --- force nul at last point ---
def get_last_contact_force_null(pn: PenaltyNode, idx_forces: np.ndarray) -> MX:
    """
    Adds the constraint that the force at the specific contact point should be null
    at the last phase point.
    All contact forces can be set at 0 at the last node by using 'all' at contact_name.

    Parameters
    ----------
    pn: PenaltyNodes
        The penalty node elements
    idx_forces: array
        indices of the force to set at 0

    Returns
    -------
    The value that should be constrained in the MX format

    """

    force = pn.nlp.contact_forces_func(pn.x, pn.u, pn.p)
    val = force[np.array(idx_forces)]
    return val

class constraint:
    @staticmethod
    def set_constraint_heel_strike(constraints, p):
        constraints.add(  # null speed for the first phase --> non sliding contact point
            ConstraintFcn.TRACK_MARKERS_VELOCITY,
            node=Node.START,
            index=26,
            phase=p,
        )

    @staticmethod
    def set_constraint_flatfoot(constraints, p):
        constraints.add(  # positive vertical forces
            ConstraintFcn.CONTACT_FORCE,
            min_bound=0,
            max_bound=np.inf,
            node=Node.ALL,
            contact_force_idx=(1, 2, 5),
            phase=p,
        )
        constraints.add(  # non slipping y
            ConstraintFcn.NON_SLIPPING,
            node=Node.ALL,
            normal_component_idx=(1, 2, 5),
            tangential_component_idx=4,
            static_friction_coefficient=0.2,
            phase=p,
        )
        constraints.add(  # non slipping x m5
            ConstraintFcn.NON_SLIPPING,
            node=Node.ALL,
            normal_component_idx=(2, 5),
            tangential_component_idx=3,
            static_friction_coefficient=0.2,
            phase=p,
        )
        constraints.add(  # non slipping x heel
            ConstraintFcn.NON_SLIPPING,
            node=Node.ALL,
            normal_component_idx=1,
            tangential_component_idx=0,
            static_friction_coefficient=0.2,
            phase=p,
        )

        constraints.add(  # forces heel at zeros at the end of the phase
            get_last_contact_force_null,
            node=Node.PENULTIMATE,
            idx_forces=(0, 1),
            phase=p,
        )

    @staticmethod
    def set_constraint_forefoot(constraints, p):
        constraints.add(  # positive vertical forces
            ConstraintFcn.CONTACT_FORCE,
            min_bound=0,
            max_bound=np.inf,
            node=Node.ALL,
            contact_force_idx=(2, 4, 5),
            phase=p,
        )
        constraints.add(
            ConstraintFcn.NON_SLIPPING,
            node=Node.ALL,
            normal_component_idx=(2, 4, 5),
            tangential_component_idx=1,
            static_friction_coefficient=0.2,
            phase=p,
        )
        constraints.add(  # non slipping x m1
            ConstraintFcn.NON_SLIPPING,
            node=Node.ALL,
            normal_component_idx=2,
            tangential_component_idx=0,
            static_friction_coefficient=0.2,
            phase=p,
        )
        constraints.add(
            get_last_contact_force_null,
            node=Node.PENULTIMATE,
            idx_forces=range(6),
            phase=p,
        )

    @staticmethod
    def set_constraint_forefoot_four_phases(constraints, p):
        constraints.add(  # positive vertical forces
            ConstraintFcn.CONTACT_FORCE,
            min_bound=0,
            max_bound=np.inf,
            node=Node.ALL,
            contact_force_idx=(0, 2, 5),
            phase=p,
        )
        constraints.add(
            ConstraintFcn.NON_SLIPPING,
            node=Node.ALL,
            normal_component_idx=(0, 2, 5),
            tangential_component_idx=4,
            static_friction_coefficient=0.2,
            phase=p,
        )
        # constraints.add(  # non slipping x m1
        #     ConstraintFcn.NON_SLIPPING,
        #     node=Node.ALL,
        #     normal_component_idx=2,
        #     tangential_component_idx=0,
        #     static_friction_coefficient=0.2,
        #     phase=p,
        # )
        # constraints.add(
        #     get_last_contact_force_null,
        #     node=Node.ALL,
        #     idx_forces=(0, 1, 2),
        #     phase=p,
        # )

    @staticmethod
    def set_constraint_toe(constraints, p):
        constraints.add(
            get_last_contact_force_null,
            node=Node.ALL,
            idx_forces=(0, 1, 2),
            phase=p,
        )