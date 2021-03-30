from bioptim import ConstraintFcn, Node, PenaltyNodes
import numpy as np
import biorbd
from casadi import MX, vertcat

def custom_CoM_position(pn: PenaltyNodes) -> MX:
    nq = pn.nlp.shape["q"]
    compute_CoM = biorbd.to_casadi_func("CoM", pn.nlp.model.CoM, pn.nlp.q)
    com = compute_CoM(pn.x[0][:nq])
    return com[2]

def get_last_contact_force(pn: PenaltyNodes, contact_force_idx) -> MX:
    force = pn.nlp.contact_forces_func(pn.x[-1], pn.u[-1], pn.p)
    return force[contact_force_idx]

def custom_foot_inequality(pn: PenaltyNodes, inequality_value: float) -> MX:
    # pied droit au dessus
    nq = pn.nlp.shape["q"]
    val = []
    markers = biorbd.to_casadi_func("markers", pn.nlp.model.markers, pn.nlp.q)
    val.append((markers(pn.x[0][:nq])[2, 31] + inequality_value) - markers(pn.x[0][:nq])[2, 55])  # heel
    val.append((markers(pn.x[0][:nq])[2, 32] + inequality_value) - markers(pn.x[0][:nq])[2, 56])  # meta1
    val.append((markers(pn.x[0][:nq])[2, 33] + inequality_value) - markers(pn.x[0][:nq])[2, 57])  # meta5
    return vertcat(*val)

def custom_foot_position(pn: PenaltyNodes) -> MX:
    nq = pn.nlp.shape["q"]
    val=[]
    markers = biorbd.to_casadi_func("markers", pn.nlp.model.markers, pn.nlp.q)

    # --- toe --- #
    val.append(markers(pn.x[0][:nq])[0, 31] - markers(pn.x[0][:nq])[0, 55])
    # val.append((-markers(pn.x[0][:nq])[1, 31]) - markers(pn.x[0][:nq])[1, 55])
    # --- meta 1 --- #
    val.append(markers(pn.x[0][:nq])[0, 32] - markers(pn.x[0][:nq])[0, 56])
    # val.append((-markers(pn.x[0][:nq])[1, 32]) - markers(pn.x[0][:nq])[1, 56])
    # --- meta 5 --- #
    val.append(markers(pn.x[0][:nq])[0, 33] - markers(pn.x[0][:nq])[0, 57])
    # val.append((-markers(pn.x[0][:nq])[1, 33]) - markers(pn.x[0][:nq])[1, 57])
    return vertcat(*val)


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
            custom_foot_position,
            node=Node.START,
        )

        constraints.add(
            custom_CoM_position,
            node=Node.END,
            max_bound=0.0,
            min_bound=-0.02,
        )
        constraints.add(  # non sliding contact point
            ConstraintFcn.TRACK_MARKERS_VELOCITY,
            node=Node.START,
            index=(31, 55),
        )

        # --- Inequality foot --- #
        constraints.add(
            custom_foot_inequality,
            inequality_value=0.03,
            node=Node.START,
        )
        return constraints

    @staticmethod
    def set_constraints_position(constraints, inequality_value=0.0):
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

        # # --- CoM initial and final --- #
        # constraints.add(
        #     custom_CoM_position,
        #     node=Node.START,
        #     max_bound=0.02,
        #     min_bound=-0.02,
        # )
        #
        # constraints.add(
        #     custom_foot_position,
        #     node=Node.START,
        # )
        #
        # constraints.add(  # non sliding contact point
        #     ConstraintFcn.TRACK_MARKERS_VELOCITY,
        #     node=Node.START,
        #     index=(31, 32, 55, 56),
        # )

        # --- Inequality foot --- #
        constraints.add(
            custom_foot_inequality,
            inequality_value=inequality_value,
            node=Node.START,
        )
        return constraints