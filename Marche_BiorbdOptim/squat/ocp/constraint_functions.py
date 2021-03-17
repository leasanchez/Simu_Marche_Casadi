from bioptim import ConstraintFcn, Node, PenaltyNodes
import numpy as np
import biorbd
from casadi import MX

def custom_CoM_position(pn: PenaltyNodes) -> MX:
    nq = pn.nlp.shape["q"]
    compute_CoM = biorbd.to_casadi_func("CoM", pn.nlp.model.CoM, pn.nlp.q)
    com = compute_CoM(pn.x[0][:nq])
    return com[2]

def get_last_contact_force(pn: PenaltyNodes, contact_force_idx) -> MX:
    force = pn.nlp.contact_forces_func(pn.x[-1], pn.u[-1], pn.p)
    return force[contact_force_idx]

class constraint:
    @staticmethod
    def set_constraints(constraints):
        # --- contact forces --- #
        contact_z_axes = (2, 3, 5, 8, 9, 11)
        for c in contact_z_axes:
            constraints.add(  # positive vertical forces
                ConstraintFcn.CONTACT_FORCE,
                min_bound=0,
                max_bound=np.inf,
                node=Node.ALL,
                contact_force_idx=c,
            )

        constraints.add(
            get_last_contact_force,
            contact_force_idx=contact_z_axes,
            min_bound=0,
            max_bound=np.inf,
            node=Node.ALL,
        )

        # --- CoM initial and final --- #
        constraints.add(
            custom_CoM_position,
            node=Node.START,
            max_bound=0.0,
            min_bound=-0.02,
        )
        constraints.add(
            custom_CoM_position,
            node=Node.END,
            max_bound=0.0,
            min_bound=-0.02,
        )

        return constraints