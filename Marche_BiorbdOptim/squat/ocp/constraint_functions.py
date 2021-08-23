from bioptim import ConstraintFcn, Node, PenaltyNode
import numpy as np
import biorbd_casadi as biorbd
from casadi import MX, vertcat

def custom_CoM_position(pn: PenaltyNode) -> MX:
    nq = pn.nlp.shape["q"]
    compute_CoM = biorbd.to_casadi_func("CoM", pn.nlp.model.CoM, pn.nlp.q)
    com = compute_CoM(pn.x[0][:nq])
    return com[2]

def custom_CoM_velocity(pn: PenaltyNode) -> MX:
    nq = pn.nlp.shape["q"]
    compute_CoM_dot = biorbd.to_casadi_func("CoM_dot", pn.nlp.model.CoMdot, pn.nlp.q, pn.nlp.qdot)
    com_dot = compute_CoM_dot(pn.x[-1][:nq], pn.x[-1][nq:])
    return com_dot[2]

def get_last_contact_force(pn: PenaltyNode, contact_force_idx) -> MX:
    force = pn.nlp.contact_forces_func(pn.x[-1], pn.u[-1], pn.p)
    return force[contact_force_idx]

def custom_foot_inequality(pn: PenaltyNode, inequality_value: float) -> MX:
    # pied droit au dessus
    nq = pn.nlp.shape["q"]
    val = []
    markers = biorbd.to_casadi_func("markers", pn.nlp.model.markers, pn.nlp.q)
    # all points same level for 1 foot
    val.append((markers(pn.x[0][:nq])[2, 31]) - markers(pn.x[0][:nq])[2, 32])
    val.append((markers(pn.x[0][:nq])[2, 31]) - markers(pn.x[0][:nq])[2, 33])

    val.append((markers(pn.x[0][:nq])[2, 31] + inequality_value) - markers(pn.x[0][:nq])[2, 55])  # heel
    val.append((markers(pn.x[0][:nq])[2, 32] + inequality_value) - markers(pn.x[0][:nq])[2, 56])  # meta1
    val.append((markers(pn.x[0][:nq])[2, 33] + inequality_value) - markers(pn.x[0][:nq])[2, 57])  # meta5
    return vertcat(*val)

def custom_foot_position(pn: PenaltyNode) -> MX:
    nq = pn.nlp.shape["q"]
    val=[]
    markers = biorbd.to_casadi_func("markers", pn.nlp.model.markers, pn.nlp.q)

    # --- toe --- #
    val.append(markers(pn.x[0][:nq])[0, 31] - markers(pn.x[0][:nq])[0, 55])
    # --- meta 1 --- #
    # val.append(markers(pn.x[0][:nq])[0, 32] - markers(pn.x[0][:nq])[0, 56])
    # --- meta 5 --- #
    # val.append(markers(pn.x[0][:nq])[0, 33] - markers(pn.x[0][:nq])[0, 57])

    return vertcat(*val)


class constraint:
    @staticmethod
    def set_constraints(constraints, inequality_value=0.0):
        # --- contact forces --- #
        contact_z_axes = (2, 3, 5, 8, 9, 11)
        for c in contact_z_axes:
            constraints.add(
                ConstraintFcn.TRACK_CONTACT_FORCES,
                min_bound=0,
                max_bound=np.inf,
                node=Node.ALL,
                contact_index=c,
                expand=False,
            )
        constraints.add(ConstraintFcn.TIME_CONSTRAINT,
                        node=Node.END,
                        min_bound=0.8,
                        max_bound=2.0,
                        expand=False)
        constraints.add(  # non sliding contact point
            ConstraintFcn.TRACK_MARKERS_VELOCITY,
            node=Node.START,
            marker_index=(31, 55),
        )

    @staticmethod
    def set_constraints_fall(constraints, inequality_value=0.0, phase=0):
        # --- contact forces --- #
        contact_z_axes = (2, 3, 5, 8, 9, 11)
        for c in contact_z_axes:
            constraints.add(
                ConstraintFcn.TRACK_CONTACT_FORCES,
                min_bound=0,
                max_bound=np.inf,
                node=Node.ALL,
                contact_index=c,
                phase=phase,
                expand=False,
            )
        constraints.add(ConstraintFcn.TIME_CONSTRAINT,
                        node=Node.END,
                        min_bound=0.3,
                        max_bound=1.2,
                        phase=phase,
                        expand=False)

        # --- Foot --- #
        # constraints.add(
        #     custom_foot_position,
        #     node=Node.START,
        # )
        # constraints.add(
        #     custom_foot_inequality,
        #     inequality_value=inequality_value,
        #     node=Node.START,
        #     phase=phase,
        # )
        constraints.add(  # non sliding contact point
            ConstraintFcn.TRACK_MARKERS_VELOCITY,
            node=Node.START,
            marker_index=(31, 55),
            phase=phase,
        )


    @staticmethod
    def set_constraints_climb(constraints, inequality_value=0.0, phase=1):
        # --- contact forces --- #
        contact_z_axes = (2, 3, 5, 8, 9, 11)
        for c in contact_z_axes:
            constraints.add(
                ConstraintFcn.TRACK_CONTACT_FORCES,
                min_bound=0,
                max_bound=np.inf,
                node=Node.ALL,
                contact_index=c,
                phase=phase,
                expand=False,
            )
        constraints.add(ConstraintFcn.TIME_CONSTRAINT,
                        node=Node.END,
                        min_bound=0.3,
                        max_bound=1.2,
                        phase=phase,
                        expand=False)


    @staticmethod
    def set_constraints_multiphase(constraints, inequality_value=0.0):
        constraint.set_constraints_fall(constraints, inequality_value, phase=0)
        constraint.set_constraints_climb(constraints, inequality_value, phase=1)