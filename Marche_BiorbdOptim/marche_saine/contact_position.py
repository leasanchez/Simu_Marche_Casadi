from time import time

import numpy as np
import biorbd
from casadi import Function, MX, vertcat, nlpsol

from gait.load_experimental_data import LoadData



def get_phase_time_shooting_numbers(data, dt):
    phase_time = data.c3d_data.get_time()
    number_shooting_points = []
    for time in phase_time:
        number_shooting_points.append(int(time / dt) - 1)
    return phase_time, number_shooting_points

def forces_from_forward_dynamics(model, q, qdot, residual_tau, activation):
    muscles_states = biorbd.VecBiorbdMuscleState(model.nbMuscleTotal())
    for k in range(model.nbMuscleTotal()):
        muscles_states[k].setActivation(activation[k])
    muscles_tau = model.muscularJointTorque(muscles_states, q, qdot).to_mx()

    tau = muscles_tau + residual_tau

    cs = model.getConstraints()
    biorbd.Model.ForwardDynamicsConstraintsDirect(model, q, qdot, tau, cs)
    return cs.getForce().to_mx()

def contact_forces_casadi(model):
    q_sym = MX.sym("q_sym", model.nbQ(), 1)
    qdot_sym = MX.sym("qdot_sym", model.nbQ(), 1)
    residual_tau_sym = MX.sym("residual_tau_sym", model.nbQ(), 1)
    activation_sym = MX.sym("activation_sym", model.nbMuscleTotal())
    return Function("contact_forces",
                              [q_sym, qdot_sym, residual_tau_sym, activation_sym],
                              [forces_from_forward_dynamics(model, q_sym, qdot_sym, residual_tau_sym,
                                                            activation_sym)],
                              ["q_sym", "qdot_sym", "residual_tau_sym", "activation_sym"],
                              ["forces"]).expand()

def markers_func_casadi(model):
    symbolic_q = MX.sym("q", model.nbQ(), 1)
    markers_func = []
    for m in range(model.nbMarkers()):
        markers_func.append(Function(
            "ForwardKin",
            [symbolic_q], [model.marker(symbolic_q, m).to_mx()],
            ["q"],
            ["markers"],
        ).expand())
    return markers_func

# Define the problem -- model path
biorbd_model = (
    biorbd.Model("models/Gait_1leg_12dof_heel.bioMod"),
    biorbd.Model("models/Gait_1leg_12dof_flatfoot.bioMod"),
    biorbd.Model("models/Gait_1leg_12dof_forefoot.bioMod"),
    biorbd.Model("models/Gait_1leg_12dof_0contact.bioMod"),
)

# Problem parameters
nb_q = biorbd_model[0].nbQ()
nb_qdot = biorbd_model[0].nbQdot()
nb_tau = biorbd_model[0].nbGeneralizedTorque()
nb_phases = len(biorbd_model)
nb_markers = biorbd_model[0].nbMarkers()
nb_mus = biorbd_model[0].nbMuscleTotal()

# --- files path ---
c3d_file = "../../DonneesMouvement/normal01_out.c3d"
q_kalman_filter_file = "../../DonneesMouvement/normal01_q_KalmanFilter.txt"
qdot_kalman_filter_file = "../../DonneesMouvement/normal01_qdot_KalmanFilter.txt"
data = LoadData(biorbd_model[0], c3d_file, q_kalman_filter_file, qdot_kalman_filter_file, 0.01, interpolation=True)

# --- phase time and number of shooting ---
phase_time=data.phase_time
number_shooting_points = data.number_shooting_points

# --- get experimental data ---
q_ref = data.q_ref
qdot_ref = data.qdot_ref
markers_ref = data.markers_ref
grf_ref = data.grf_ref
moments_ref = data.moments_ref
cop_ref = data.cop_ref

res_tau = np.load("./RES/1leg/cycle/muscles/4_contacts/markers_tracking/hip/tau_hip.npy")
activation = np.load("./RES/1leg/cycle/muscles/4_contacts/markers_tracking/hip/activation_hip.npy")

forces = {}
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

cn = []
for c in biorbd_model[1].contactNames():
    cn.append(c.to_string())

# --- flatfoot - get position ---
pos_Heel = MX.sym("pos_Heel", 2, 1)
pos_Meta1 = MX.sym("pos_Meta1", 2, 1)
pos_Meta5 = MX.sym("pos_Meta5", 2, 1)
pos_Toe = MX.sym("pos_Toe", 2, 1)
objective = 0
lbg = []
ubg = []
constraint = []

# q_sym = MX.sym("q_sym", biorbd_model[1].nbQ(), 1)
# qdot_sym = MX.sym("qdot_sym", biorbd_model[1].nbQ(), 1)
# residual_tau_sym = MX.sym("residual_tau_sym", biorbd_model[1].nbQ(), 1)
# activation_sym = MX.sym("activation_sym", biorbd_model[1].nbMuscleTotal())
# contact_forces= Function( "contact_forces",
#                 [q_sym, qdot_sym, residual_tau_sym, activation_sym],
#                 [forces_from_forward_dynamics(biorbd_model[1], q_sym, qdot_sym, residual_tau_sym,
#                                               activation_sym)],
#                 ["q_sym", "qdot_sym", "residual_tau_sym", "activation_sym"],
#                 ["forces"]).expand()

markers_func = np.array(markers_func_casadi(biorbd_model[0]))
init_heel = np.array(markers_func[26](q_ref[1][:, 0]))
init_meta1 = np.array(markers_func[27](q_ref[1][:, 0]))
init_meta5 = np.array(markers_func[28](q_ref[1][:, 0]))
init_toe = np.array(markers_func[29](q_ref[2][:, 0]))

contact_forces=contact_forces_casadi(biorbd_model[1])
for i in range(number_shooting_points[1] + 1):
    # Aliases
    heel = cop_ref[1][:-1, i] - pos_Heel
    meta1 = cop_ref[1][:-1, i] - pos_Meta1
    meta5 = cop_ref[1][:-1, i] - pos_Meta5

    # compute forces
    force_sim = contact_forces(q_ref[1][:, i], qdot_ref[1][:, i], res_tau[:, 4 + i], activation[:, 4 + i])
    for f_name in labels_forces:
        forces[f_name] = 0.0
    for c in cn:
        forces[c] = force_sim[cn.index(c)]

    # tracking moments
    Mx = (heel[1] * forces["Heel_r_Z"]
            + meta1[1] * forces["Meta_1_r_Z"]
            + meta5[1] * forces["Meta_5_r_Z"])
    My = (-heel[0] * forces["Heel_r_Z"]
            - meta1[0] * forces["Meta_1_r_Z"]
            - meta5[0] * forces["Meta_5_r_Z"])
    Mz = (heel[0] * forces["Heel_r_Y"] - heel[1] * forces["Heel_r_X"]
          + meta1[0] * forces["Meta_1_r_Y"] - meta1[1] * forces["Meta_1_r_X"]
          + meta5[0] * forces["Meta_5_r_Y"] - meta5[1] * forces["Meta_5_r_X"])

    jm0 = Mx - moments_ref[1][0, i]
    jm1 = moments_ref[1][1, i] - My
    jm2 = moments_ref[1][2, i] - Mz
    objective += jm0 * jm0 + jm1 * jm1 + jm2 * jm2

contact_forces_2 = contact_forces_casadi(biorbd_model[2])
for i in range(number_shooting_points[2] + 1):
    # Aliases
    meta1 = cop_ref[2][:-1, i] - pos_Meta1
    meta5 = cop_ref[2][:-1, i] - pos_Meta5
    toe = cop_ref[2][:-1, i] - pos_Toe

    # compute forces
    force_sim = contact_forces_2(q_ref[2][:, i], qdot_ref[2][:, i], res_tau[:, 38+i], activation[:, 38+i])
    for f_name in labels_forces:
        forces[f_name] = 0.0
    for c in cn:
        forces[c] = force_sim[cn.index(c)]

    # tracking moments
    Mx = ( meta1[1] * forces["Meta_1_r_Z"] + meta5[1] * forces["Meta_5_r_Z"] + toe[1] * forces["Toe_r_Z"])
    My = (- meta1[0] * forces["Meta_1_r_Z"] - meta5[0] * forces["Meta_5_r_Z"] - toe[0] * forces["Toe_r_Z"])
    Mz = (meta1[0] * forces["Meta_1_r_Y"] - meta1[1] * forces["Meta_1_r_X"]
          + meta5[0] * forces["Meta_5_r_Y"] - meta5[1] * forces["Meta_5_r_X"]
          + toe[0] * forces["Toe_r_Y"] - toe[1] * forces["Toe_r_X"])

    jm0 = Mx - moments_ref[2][0, i]
    jm1 = moments_ref[2][1, i] - My
    jm2 = moments_ref[2][2, i] - Mz
    objective += jm0 * jm0 + jm1 * jm1 + jm2 * jm2

# x0_pos = [init_heel[0], init_heel[1], init_meta1[0], init_meta1[1], init_meta5[0], init_meta5[1]]
x0_pos = [init_heel[0], init_heel[1], init_meta1[0], init_meta1[1], init_meta5[0], init_meta5[1], init_toe[0], init_toe[1]]
lbx = []
ubx = []
for x0 in x0_pos:
    lbx.append(x0-0.01)
    ubx.append(x0+0.01)

# w = [pos_Heel, pos_Meta1, pos_Meta5]
w = [pos_Heel, pos_Meta1, pos_Meta5, pos_Toe]
nlp = {"x": vertcat(*w), "f": objective, "g": vertcat(*constraint)}
opts = {"ipopt.tol": 1e-8, "ipopt.hessian_approximation": "exact"}
solver = nlpsol("solver", "ipopt", nlp, opts)
res = solver(x0=x0_pos, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

a=2