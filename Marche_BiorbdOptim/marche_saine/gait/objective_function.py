import numpy as np
import biorbd
from bioptim import ObjectiveFcn, Node, PenaltyNodes
from casadi import vertcat, MX


def track_sum_contact_forces(pn: PenaltyNodes, grf: np.ndarray) -> MX:
    """
    Adds the objective that the mismatch between the
    sum of the contact forces and the reference ground reaction forces should be minimized.

    Parameters
    ----------
    pn: PenaltyNodes
        The penalty node elements
    grf: np.ndarray
        Array of the measured ground reaction forces

    Returns
    -------
    The cost that should be minimize in the MX format.
    """

    ns = pn.nlp.ns  # number of shooting points for the phase
    val = []  # init
    cn = []
    for c in pn.nlp.model.contactNames():
        cn.append(c.to_string()) # contact name for the model

    # --- compute forces ---
    forces = {}  # define dictionnary with all the contact point possible
    labels_forces = [
        "Heel_r_X",
        "Heel_r_Y",
        "Heel_r_Z",
        "Meta_1_r_X",
        "Meta_1_r_Y",
        "Meta_1_r_Z",
        "Meta_5_r_X",
        "Meta_5_r_Y",
        "Meta_5_r_Z",
        "Toe_r_X",
        "Toe_r_Y",
        "Toe_r_Z",
    ]

    for n in range(ns):
        # --- compute forces ---
        force_sim = pn.nlp.contact_forces_func(pn.x[n], pn.u[n], pn.p)
        for f_name in labels_forces:
            forces[f_name] = 0.0
        for c in cn:
            forces[c] = force_sim[cn.index(c)]

        # --- tracking forces ---
        val = vertcat(
            val,
            grf[0, pn.t[n]]
            - (forces["Heel_r_X"] + forces["Meta_1_r_X"] + forces["Meta_5_r_X"] + forces["Toe_r_X"]),
        )
        val = vertcat(
            val,
            grf[1, pn.t[n]]
            - (forces["Heel_r_Y"] + forces["Meta_1_r_Y"] + forces["Meta_5_r_Y"] + forces["Toe_r_Y"]),
        )
        val = vertcat(
            val,
            grf[2, pn.t[n]]
            - (forces["Heel_r_Z"] + forces["Meta_1_r_Z"] + forces["Meta_5_r_Z"] + forces["Toe_r_Z"]),
        )
    return val

def track_sum_contact_moments(pn: PenaltyNodes, CoP: np.ndarray, M_ref: np.ndarray) -> MX:
    """
    Adds the objective that the mismatch between the
    sum of the contact moments and the reference ground reaction moments should be minimized.

    Parameters
    ----------
    pn: PenaltyNodes
        The penalty node elements
    CoP: np.ndarray
        Array of the measured center of pressure trajectory
    M_ref: np.ndarray
        Array of the measured ground reaction moments

    Returns
    -------
    The cost that should be minimize in the MX format.

    """

    # --- aliases ---
    ns = pn.nlp.ns  # number of shooting points for the phase
    nq = pn.nlp.shape["q"] # number of dof
    markers = biorbd.to_casadi_func("markers", pn.nlp.model.markers, pn.nlp.q)
    cn = []
    for c in pn.nlp.model.contactNames() :
        cn.append(c.to_string())           # contact name for the model
    val = []  # init

    # --- init forces ---
    forces = {}  # define dictionnary with all the contact point possible
    labels_forces = [
        "Heel_r_X",
        "Heel_r_Y",
        "Heel_r_Z",
        "Meta_1_r_X",
        "Meta_1_r_Y",
        "Meta_1_r_Z",
        "Meta_5_r_X",
        "Meta_5_r_Y",
        "Meta_5_r_Z",
        "Toe_r_X",
        "Toe_r_Y",
        "Toe_r_Z",
    ]

    for n in range(ns):
        # --- compute contact point position ---
        q = pn.x[n][:nq]
        heel = markers(q)[:, 26] - CoP[:, pn.t[n]]
        meta1 = markers(q)[:, 27] - CoP[:, pn.t[n]]
        meta5 = markers(q)[:, 28] - CoP[:, pn.t[n]]
        toe = markers(q)[:, 29] - CoP[:, pn.t[n]]

        # --- compute forces ---
        force_sim = pn.nlp.contact_forces_func(pn.x[n], pn.u[n], pn.p)
        for f_name in labels_forces:
            forces[f_name] = 0.0
        for c in cn:
            forces[c]=force_sim[cn.index(c)]

        # --- tracking moments ---
        Mx = (
            heel[1] * forces["Heel_r_Z"]
            + meta1[1] * forces["Meta_1_r_Z"]
            + meta5[1] * forces["Meta_5_r_Z"]
            + toe[1] * forces["Toe_r_Z"]
        )
        My = (
            -heel[0] * forces["Heel_r_Z"]
            - meta1[0] * forces["Meta_1_r_Z"]
            - meta5[0] * forces["Meta_5_r_Z"]
            - toe[0] * forces["Toe_r_Z"]
        )
        Mz = (heel[0] * forces["Heel_r_Y"] - heel[1] * forces["Heel_r_X"]
              + meta1[0] * forces["Meta_1_r_Y"] - meta1[1] * forces["Meta_1_r_X"]
              + meta5[0] * forces["Meta_5_r_Y"] - meta5[1] * forces["Meta_5_r_X"]
              + toe[0] * forces["Toe_r_Y"] - toe[1] * forces["Toe_r_X"])

        val = vertcat(val, M_ref[0, pn.t[n]] - Mx)
        val = vertcat(val, M_ref[1, pn.t[n]] - My)
        val = vertcat(val, M_ref[2, pn.t[n]] - Mz)
    return val


class objective:
    @staticmethod
    def set_objective_function_markers(objective_functions, markers_ref,p):
        # --- markers_idx ---
        markers_pelvis = [0, 1, 2, 3]
        markers_anat = [4, 9, 10, 11, 12, 17, 18]
        markers_tissus = [5, 6, 7, 8, 13, 14, 15, 16]
        markers_pied = [19, 20, 21, 22, 23, 24, 25]
        markers_idx = (markers_anat, markers_pelvis, markers_pied, markers_tissus)
        weigth_markers = (1000, 100000, 100000, 100)
        for (i, m_idx) in enumerate(markers_idx):
            objective_functions.add(ObjectiveFcn.Lagrange.TRACK_MARKERS,
                                    weight=weigth_markers[i],
                                    index=m_idx,
                                    target=markers_ref[:, m_idx, :],
                                    phase=p,
                                    quadratic=True)

    @staticmethod
    def set_objective_function_muscle_controls(objective_functions, p):
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=1e-2, index=(10), phase=p, quadratic=True)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=1e1, index=(6, 7, 8, 9, 11), phase=p, quadratic=True)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=1e2, phase=p, quadratic=True)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE_DERIVATIVE, weight=1e-1, phase=p, quadratic=True)

    @staticmethod
    def set_objective_function_forces(objective_functions, grf_ref, p):
        objective_functions.add(track_sum_contact_forces,
                                grf=grf_ref,
                                custom_type=ObjectiveFcn.Lagrange,
                                node=Node.ALL,
                                weight=0.1,
                                quadratic=True,
                                phase=p)

    @staticmethod
    def set_objective_function_moments(objective_functions, moment_ref, cop_ref, p):
        objective_functions.add(track_sum_contact_moments,
                                 CoP=cop_ref,
                                 M_ref=moment_ref,
                                 custom_type=ObjectiveFcn.Lagrange,
                                 node=Node.ALL,
                                 weight=0.01,
                                 quadratic=True,
                                 phase=p)

    @staticmethod
    def set_objective_function_heel_strike(objective_functions, markers_ref, grf_ref, moment_ref, cop_ref, p):
        objective.set_objective_function_markers(objective_functions, markers_ref, p)
        objective.set_objective_function_muscle_controls(objective_functions, p)
        objective.set_objective_function_forces(objective_functions, grf_ref, p)

    @staticmethod
    def set_objective_function_flatfoot(objective_functions, markers_ref, grf_ref, moment_ref, cop_ref, p):
        objective.set_objective_function_markers(objective_functions, markers_ref, p)
        objective.set_objective_function_muscle_controls(objective_functions, p)
        objective.set_objective_function_forces(objective_functions, grf_ref, p)
        objective.set_objective_function_moments(objective_functions, moment_ref, cop_ref, p)

    @staticmethod
    def set_objective_function_forefoot(objective_functions, markers_ref, grf_ref, moment_ref, cop_ref, p):
        objective.set_objective_function_markers(objective_functions, markers_ref, p)
        objective.set_objective_function_muscle_controls(objective_functions, p)
        objective.set_objective_function_forces(objective_functions, grf_ref, p)
        objective.set_objective_function_moments(objective_functions, moment_ref, cop_ref, p)

    @staticmethod
    def set_objective_function_toe(objective_functions, markers_ref, grf_ref, moment_ref, cop_ref, p):
        objective.set_objective_function_markers(objective_functions, markers_ref, p)
        objective.set_objective_function_muscle_controls(objective_functions, p)
        objective.set_objective_function_forces(objective_functions, grf_ref, p)

    @staticmethod
    def set_objective_function_swing(objective_functions, markers_ref, grf_ref, moment_ref, cop_ref, p):
        objective.set_objective_function_markers(objective_functions, markers_ref, p)
        objective.set_objective_function_muscle_controls(objective_functions, p)





