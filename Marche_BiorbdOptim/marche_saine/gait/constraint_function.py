import numpy as np
from bioptim import ConstraintFcn, Node, PenaltyNodes
from casadi import vertcat, MX


# --- force nul at last point ---
def get_last_contact_force_null(pn: PenaltyNodes, contact_name: str) -> MX:
    """
    Adds the constraint that the force at the specific contact point should be null
    at the last phase point.
    All contact forces can be set at 0 at the last node by using 'all' at contact_name.

    Parameters
    ----------
    pn: PenaltyNodes
        The penalty node elements
    contact_name: str
        Name of the contacts that sould be null at the last node

    Returns
    -------
    The value that should be constrained in the MX format

    """

    force = pn.nlp.contact_forces_func(pn.x[-1], pn.u[-1], pn.p)
    if contact_name == "all":
        val = force
    else:
        cn = pn.nlp.model.contactNames()
        val = []
        for i, c in enumerate(cn):
            if isinstance(contact_name, tuple):
                for name in contact_name:
                    if name in c.to_string():
                        val = vertcat(val, force[i])
            else:
                if contact_name in c.to_string():
                    val = vertcat(val, force[i])
    return val

class constraint:
    @staticmethod
    def set_constraint_heel_strike(constraints):
        constraints.add(  # null speed for the first phase --> non sliding contact point
            ConstraintFcn.TRACK_MARKERS_VELOCITY,
            node=Node.START,
            index=26,
            phase=0,
        )

    @staticmethod
    def set_constraint_flatfoot(constraints):
        constraints.add(  # positive vertical forces
            ConstraintFcn.CONTACT_FORCE,
            min_bound=0,
            max_bound=np.inf,
            node=Node.ALL,
            contact_force_idx=(1, 2, 5),
            phase=1,
        )
        constraints.add(  # non slipping y
            ConstraintFcn.NON_SLIPPING,
            node=Node.ALL,
            normal_component_idx=(1, 2, 5),
            tangential_component_idx=4,
            static_friction_coefficient=0.2,
            phase=1,
        )
        constraints.add(  # non slipping x m5
            ConstraintFcn.NON_SLIPPING,
            node=Node.ALL,
            normal_component_idx=(2, 5),
            tangential_component_idx=3,
            static_friction_coefficient=0.2,
            phase=1,
        )
        constraints.add(  # non slipping x heel
            ConstraintFcn.NON_SLIPPING,
            node=Node.ALL,
            normal_component_idx=1,
            tangential_component_idx=0,
            static_friction_coefficient=0.2,
            phase=1,
        )

        constraints.add(  # forces heel at zeros at the end of the phase
            get_last_contact_force_null,
            node=Node.ALL,
            contact_name='Heel_r',
            phase=1,
        )

    @staticmethod
    def set_constraint_forefoot(constraints):
        constraints.add(  # positive vertical forces
            ConstraintFcn.CONTACT_FORCE,
            min_bound=0,
            max_bound=np.inf,
            node=Node.ALL,
            contact_force_idx=(2, 4, 5),
            phase=2,
        )
        constraints.add(
            ConstraintFcn.NON_SLIPPING,
            node=Node.ALL,
            normal_component_idx=(2, 4, 5),
            tangential_component_idx=1,
            static_friction_coefficient=0.2,
            phase=2,
        )
        constraints.add(  # non slipping x m1
            ConstraintFcn.NON_SLIPPING,
            node=Node.ALL,
            normal_component_idx=2,
            tangential_component_idx=0,
            static_friction_coefficient=0.2,
            phase=2,
        )
        constraints.add(
            get_last_contact_force_null,
            node=Node.ALL,
            contact_name='all',
            phase=2,
        )