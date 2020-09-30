import biorbd
import numpy as np
import casadi
from BiorbdViz import BiorbdViz
import Marche_BiorbdOptim.moco.Load_OpenSim_data as Moco
from matplotlib import pyplot as plt

# problems parameters
model = biorbd.Model("../../ModelesS2M/Open_Sim/subject_walk_armless_test.bioMod")
t_init = 0.81
t_end = 1.65
final_time = 0.84
nb_shooting = 84
nb_q = model.nbQ()
nb_markers = model.nbMarkers()
node_t = np.linspace(0, final_time, nb_shooting + 1)
q_name = []
for s in range(model.nbSegment()):
    seg_name = model.segment(s).name().to_string()
    for d in range(model.segment(s).nbDof()):
        dof_name = model.segment(s).nameDof(d).to_string()
        q_name.append(seg_name + "_" + dof_name)

# Generate data from file OpenSim
[Q_ref, Qdot_ref, Qddot_ref] = Moco.get_state_tracked(t_init, t_end, final_time, nb_q, nb_shooting)
[Q_MI, Qdot_MI, Activation_MI] = Moco.get_state_from_solution_MI(t_init, t_end, final_time, nb_q, nb_shooting)
[Tau_MI, Excitation_MI] = Moco.get_control_from_solution_MI(t_init, t_end, final_time, nb_q, nb_shooting)
[Force_ref, Moment_ref] = Moco.get_grf(t_init, t_end, final_time, nb_shooting)
position = Moco.get_position(t_init, t_end, final_time, nb_shooting)
Tau_ID = Moco.get_tau_from_inverse_dynamics("inverse_dynamics_fext.xlsx", t_init, t_end, final_time, nb_q, nb_shooting)

# Compute ankle moments
markers_pos = np.zeros((3, nb_markers, nb_shooting + 1))
for i in range(nb_shooting + 1):
    for m in range(nb_markers):
        # --- define casadi function ---
        q_iv = casadi.MX.sym("Q", nb_q, 1)
        marker_func = biorbd.to_casadi_func("marker", model.marker, q_iv, m)
        markers_pos[:, m, i:i+1] = marker_func(Q_ref[:, i])

Mext = []
for leg in range(len(Force_ref)):
    pos = position[leg]
    force = Force_ref[leg]
    moment = Moment_ref[leg]
    marker = markers_pos[:, leg, :]
    M = np.zeros((3, nb_shooting + 1))
    for i in range(nb_shooting + 1):
        p = marker[:, i] - pos[:, i]
        M[:, i]=np.cross(p, force[:, i]) + moment[:, i]
    Mext.append(M)

# plot forces and moments
figure, axes = plt.subplots(1,3)
axes = axes.flatten()
forces_coord = ['Fx', 'Fy', 'Fz']
for i in range(3):
    axes[i].plot(node_t, Force_ref[0][i, :], color="tab:blue", linestyle="-")
    axes[i].plot(node_t, Force_ref[1][i, :], color="tab:red", linestyle="--")
    axes[i].set_title(forces_coord[i])
plt.legend(['Right', 'Left'])

figure, axes = plt.subplots(1,3)
axes = axes.flatten()
moments_coord = ['Mx', 'My', 'Mz']
for i in range(3):
    axes[i].plot(node_t, Mext[0][i, :], color="tab:blue", linestyle="-")
    axes[i].plot(node_t, Mext[1][i, :], color="tab:red", linestyle="--")
    axes[i].set_title(moments_coord[i])
plt.legend(['Right', 'Left'])
plt.show()

# compute inverse dynamics
tau_fext = np.zeros((nb_q, nb_shooting + 1))
for n in range(nb_shooting + 1):
    # --- define external forces ---
    forces = biorbd.VecBiorbdSpatialVector()
    forces.append(biorbd.SpatialVector(casadi.MX((Mext[0][0, n], Mext[0][1, n], Mext[0][2, n], Force_ref[0][0, n], Force_ref[0][1, n], Force_ref[0][2, n]))))
    forces.append(biorbd.SpatialVector(casadi.MX((Mext[1][0, n], Mext[1][1, n], Mext[1][2, n], Force_ref[1][0, n], Force_ref[1][1, n], Force_ref[1][2, n]))))

    # --- define casadi function ---
    q_iv = casadi.MX.sym("Q", nb_q, 1)
    dq_iv = casadi.MX.sym("Qdot", nb_q, 1)
    ddq_iv = casadi.MX.sym("Qddot", nb_q, 1)
    func = biorbd.to_casadi_func("ID", model.InverseDynamics, q_iv, dq_iv, ddq_iv, forces)

    # --- compute torques from inverse dynamics ---
    q_iv = Q_ref[:, n]
    dq_iv = Qdot_ref[:, n]
    ddq_iv = Qddot_ref[:, n]
    tau_fext[:, n:n+1] = func(q_iv, dq_iv, ddq_iv)

figure, axes = plt.subplots(5, 6)
axes = axes.flatten()
for i in range(nb_q):
    axes[i].plot(node_t, Tau_ID[i, :], color="tab:blue", linestyle="-")
    axes[i].plot(node_t, tau_fext[i, :], color="tab:red", linestyle="--")
    axes[i].set_title(q_name[i])
plt.legend(["OpenSim", "Biorbd"])
plt.show()