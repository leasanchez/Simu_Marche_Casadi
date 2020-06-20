import numpy as np
from casadi import vertcat, MX, nlpsol, mtimes, dot, Function
from matplotlib import pyplot as plt
import biorbd
from Marche_BiorbdOptim.LoadData import Data_to_track


def get_forces(biorbd_model, states, controls):
    nb_q = biorbd_model.nbQ()
    nb_qdot = biorbd_model.nbQdot()
    nb_tau = biorbd_model.nbGeneralizedTorque()
    nb_mus = biorbd_model.nbMuscleTotal()

    muscles_states = biorbd.VecBiorbdMuscleState(nb_mus)
    muscles_excitation = controls[nb_tau:]
    muscles_activations = states[nb_q + nb_qdot :]

    for k in range(nb_mus):
        muscles_states[k].setExcitation(muscles_excitation[k])
        muscles_states[k].setActivation(muscles_activations[k])

    muscles_tau = biorbd_model.muscularJointTorque(muscles_states, states[:nb_q], states[nb_q : nb_q + nb_qdot]).to_mx()
    tau = muscles_tau + controls[:nb_tau]
    cs = biorbd_model.getConstraints()
    biorbd.Model.ForwardDynamicsConstraintsDirect(biorbd_model, states[:nb_q], states[nb_q : nb_q + nb_qdot], tau, cs)
    return cs.getForce().to_mx()


biorbd_model = (
    biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_1contact_deGroote_3d_Heel.bioMod"),
    biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_3contacts_deGroote_3d.bioMod"),
    biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_contact_deGroote_3d_Forefoot.bioMod"),
)
# Problem parameters
number_shooting_points = [5, 10, 15]

# Generate data from file
Data_to_track = Data_to_track(name_subject="normal01", multiple_contact=True)
[T, T_stance, T_swing] = Data_to_track.GetTime()
phase_time = T_stance
grf_ref = Data_to_track.load_data_GRF(biorbd_model[0], T_stance, number_shooting_points)
M_ref = Data_to_track.load_data_Moment(biorbd_model[0], T_stance, number_shooting_points)
markers_ref = Data_to_track.load_data_markers(biorbd_model[0], T_stance, number_shooting_points, "stance")
q_ref = Data_to_track.load_q_kalman(biorbd_model[0], T_stance, number_shooting_points, "stance")
qdot_ref = Data_to_track.load_qdot_kalman(biorbd_model[0], T_stance, number_shooting_points, "stance")
emg_ref = Data_to_track.load_data_emg(biorbd_model[0], T_stance, number_shooting_points, "stance")
excitation_ref = []
for i in range(len(phase_time)):
    excitation_ref.append(Data_to_track.load_muscularExcitation(emg_ref[i]))

# # foot markers position
# plt.figure("foot position")
# plt.plot(markers_ref[1][0, 19, :] + 0.04, markers_ref[1][1, 19, :], "b+")
# plt.text(np.mean(markers_ref[1][0, 19, :] + 0.04), np.mean(markers_ref[1][1, 19, :]) + 0.0025, "talon")
# plt.plot(markers_ref[1][0, 21, :], markers_ref[1][1, 21, :], "r+")
# plt.text(np.mean(markers_ref[1][0, 21, :]), np.mean(markers_ref[1][1, 21, :]) + 0.0025, "FMP1")
# plt.plot(markers_ref[1][0, 24, :], markers_ref[1][1, 24, :], "g+")
# plt.text(np.mean(markers_ref[1][0, 24, :]), np.mean(markers_ref[1][1, 24, :]) + 0.0025, "FM5")
# plt.xticks(np.arange(np.min(markers_ref[1][0, 19, :] + 0.04) - 0.01, np.max(markers_ref[1][0, 21, :]) + 0.01, step = 0.02))
# plt.show()

# contact positions
Heel = np.array([np.mean(markers_ref[1][0, 19, :] + 0.04), np.mean(markers_ref[1][1, 19, :]), 0])
Meta1 = np.array([np.mean(markers_ref[1][0, 21, :]), np.mean(markers_ref[1][1, 21, :]), 0])
Meta5 = np.array([np.mean(markers_ref[1][0, 24, :]), np.mean(markers_ref[1][1, 24, :]), 0])

# Problem parameters
nb_q = biorbd_model[0].nbQ()
nb_qdot = biorbd_model[0].nbQdot()
nb_tau = biorbd_model[0].nbGeneralizedTorque()
nb_mus = biorbd_model[0].nbMuscleTotal()
nb_contact = biorbd_model[0].nbContacts()

# --- compute initial contact forces ---
symbolic_states = MX.sym("x", nb_q + nb_qdot + nb_mus, 1)
symbolic_controls = MX.sym("u", nb_tau + nb_mus, 1)
computeGRF = Function(
    "computeGRF",
    [symbolic_states, symbolic_controls],
    [get_forces(biorbd_model[1], symbolic_states, symbolic_controls)],
    ["x", "u"],
    ["GRF"],
).expand()

cf = np.zeros((nb_contact * 3, number_shooting_points[1] + 1))
for n in range(number_shooting_points[1] + 1):
    state = np.concatenate((q_ref[1][:, n], qdot_ref[1][:, n], excitation_ref[1][:, n]))
    control = np.concatenate((np.zeros(nb_tau), excitation_ref[1][:, n]))
    cf[:, n] = np.array(computeGRF(state, control)).squeeze()

# label = ["x", "y", "z"]
# for i in range(3):
#     plt.figure("contact forces" + label[i])
#     plt.plot(grf_ref[1][i, :], "k")
#     plt.plot(cf[i, :], "g")
#     plt.plot(cf[i + 3, :], "r")
#     plt.plot(cf[i + 6, :], "b")
#     plt.legend(("plateforme", "heel", "Meta 1", "Meta 5"))
# plt.show()

# --- 2 contacts forefoot ---
p = 0.5  # repartition entre les 2 points

F_Meta1 = MX.sym("F_Meta1", 3 * (number_shooting_points[2] + 1), 1)
F_Meta5 = MX.sym("F_Meta1", 3 * (number_shooting_points[2] + 1), 1)

objective = 0
lbg = []
ubg = []
constraint = []
for i in range(number_shooting_points[2] + 1):
    # Aliases
    fm1 = F_Meta1[3 * i : 3 * (i + 1)]
    fm5 = F_Meta5[3 * i : 3 * (i + 1)]

    # sum forces = 0 --> Fp1 + Fp2 = Ftrack
    sf = fm1 + fm5
    jf = sf - grf_ref[2][:, i]
    constraint += (jf[0], jf[1], jf[2])
    lbg += [0] * 3
    ubg += [0] * 3
    # objective += 1000 * mtimes(jf.T, jf)
    #
    # # sum moments = 0 --> CP1xFp1 + CP2xFp2 = Mtrack
    # sm = dot(Meta1, fm1) + dot(Meta5, fm5)
    # jm = sm - M_ref[2][:, i]
    # constraint += (jm[0], jm[1], jm[2])
    # lbg += [0] * 3
    # ubg += [0] * 3
    # objective += 100 * mtimes(jm.T, jm)

    # use of p to dispatch forces --> p*Fp1 - (1-p)*Fp2 = 0
    jf2 = p * fm1 - (1 - p) * fm5
    objective += mtimes(jf2.T, jf2)

    # positive vertical force
    constraint += (fm1[2], fm5[2])
    lbg += [0] * 2
    ubg += [1000] * 2

    # non slipping --> -0.4*Fz < Fx < 0.4*Fz
    constraint += ((-0.4 * fm1[2] - fm1[0]), (-0.4 * fm5[2] - fm5[0]))
    lbg += [-1000] * 2
    ubg += [0] * 2

    constraint += ((0.4 * fm1[2] - fm1[0]), (0.4 * fm5[2] - fm5[0]))
    lbg += [0] * 2
    ubg += [1000] * 2

w = [F_Meta1, F_Meta5]
nlp = {"x": vertcat(*w), "f": objective, "g": vertcat(*constraint)}
opts = {"ipopt.tol": 1e-8, "ipopt.hessian_approximation": "exact"}
solver = nlpsol("solver", "ipopt", nlp, opts)
res = solver(x0=np.zeros(6 * (number_shooting_points[2] + 1)), lbx=-1000, ubx=1000, lbg=lbg, ubg=ubg)
F1 = res["x"][: 3 * (number_shooting_points[2] + 1)]
F5 = res["x"][3 * (number_shooting_points[2] + 1) : 6 * (number_shooting_points[2] + 1)]

force_meta1 = np.zeros((3, (number_shooting_points[2] + 1)))
force_meta5 = np.zeros((3, (number_shooting_points[2] + 1)))
for i in range(3):
    force_meta1[i, :] = np.array(F1[i::3]).squeeze()
    force_meta5[i, :] = np.array(F5[i::3]).squeeze()

for i in range(3):
    plt.figure(f"contact forces {i + 1}")
    plt.plot(grf_ref[2][i, :], "k")
    plt.plot(force_meta1[i, :], "r")
    plt.plot(force_meta5[i, :], "b")
    plt.legend(("plateforme", "Meta 1", "Meta 5"))
plt.show()


# --- 3 contact points ---
p_heel = np.linspace(0, 1, number_shooting_points[1] + 1)
x = np.linspace(-number_shooting_points[1], number_shooting_points[1], number_shooting_points[1] + 1, dtype=int)
p_heel_sig = 1 / (1 + np.exp(-x))
p = 0.5
# Forces
F_Heel = MX.sym("F_Heel", 3 * (number_shooting_points[1] + 1), 1)
F_Meta1_3 = MX.sym("F_Meta1_3", 3 * (number_shooting_points[1] + 1), 1)
F_Meta5_3 = MX.sym("F_Meta5_3", 3 * (number_shooting_points[1] + 1), 1)

objective = 0
lbg = []
ubg = []
constraint = []

for i in range(number_shooting_points[1] + 1):
    # Aliases
    fh = F_Heel[3 * i : 3 * (i + 1)]
    fm1 = F_Meta1_3[3 * i : 3 * (i + 1)]
    fm5 = F_Meta5_3[3 * i : 3 * (i + 1)]

    # --- Torseur equilibre ---
    # sum forces = 0 --> Fp1 + Fp2 + Fh = Ftrack
    sf = fm1 + fm5 + fh
    jf = sf - grf_ref[1][:, i]
    constraint += (jf[0], jf[1], jf[2])
    lbg += [0] * 3
    ubg += [0] * 3

    # objective += 100 * mtimes(jf.T, jf)
    # sum moments = 0 --> Mp1_P1 + CP1xFp1 + Mp2_P2 + CP2xFp2 = Mtrack
    sm = dot(Meta1, fm1) + dot(Meta5, fm5) + dot(Heel, fh)
    jm = sm - M_ref[1][:, i]
    constraint += (jm[0], jm[1], jm[2])
    lbg += [0] * 3
    ubg += [0] * 3
    # objective += 100 * mtimes(jm.T, jm)

    # --- Dispatch on different contact points ---
    # use of p to dispatch forces --> p_heel*Fh - (1-p_heel)*Fm = 0
    jf2 = p_heel_sig[i] * fh - ((1 - p_heel_sig[i]) * (p * fm1 + (1 - p) * fm5))
    objective += mtimes(jf2.T, jf2)

    # --- Forces constraints ---
    # positive vertical force
    constraint+=(fh[2], fm1[2], fm5[2])
    lbg += [0] * 3
    ubg += [1000] * 3
    # non slipping --> -0.4*Fz < Fx < 0.4*Fz
    constraint += ((-0.4 * fh[2] - fh[0]), (-0.4 * fm1[2] - fm1[0]), (-0.4 * fm5[2] - fm5[0]))
    lbg += [-1000] * 3
    ubg += [0] * 3

    constraint += ((0.4 * fh[2] - fh[0]), (0.4 * fm1[2] - fm1[0]), (0.4 * fm5[2] - fm5[0]))
    lbg += [0] * 3
    ubg += [1000] * 3

w = [F_Heel, F_Meta1_3, F_Meta5_3]
nlp = {"x": vertcat(*w), "f": objective, "g": vertcat(*constraint)}
opts = {"ipopt.tol": 1e-8, "ipopt.hessian_approximation": "exact"}
solver = nlpsol("solver", "ipopt", nlp, opts)
res = solver(x0=np.zeros(9 * (number_shooting_points[1] + 1)), lbx=-1000, ubx=1000, lbg=lbg, ubg=ubg)

FH = res["x"][: 3 * (number_shooting_points[1] + 1)]
FM1 = res["x"][3 * (number_shooting_points[1] + 1) : 6 * (number_shooting_points[1] + 1)]
FM5 = res["x"][6 * (number_shooting_points[1] + 1) :]

force_heel = np.zeros((3, (number_shooting_points[1] + 1)))
force_meta1_3 = np.zeros((3, (number_shooting_points[1] + 1)))
force_meta5_3 = np.zeros((3, (number_shooting_points[1] + 1)))
for i in range(3):
    force_heel[i, :] = np.array(FH[i::3]).squeeze()
    force_meta1_3[i, :] = np.array(FM1[i::3]).squeeze()
    force_meta5_3[i, :] = np.array(FM5[i::3]).squeeze()

for i in range(3):
    plt.figure(f"contact forces {i + 1}")
    plt.plot(grf_ref[1][i, :], "k")
    plt.plot(force_heel[i, :], "g")
    plt.plot(force_meta1_3[i, :], "r")
    plt.plot(force_meta5_3[i, :], "b")
    plt.legend(("plateforme", "Heel", "Meta 1", "Meta 5"))
plt.show()


# --- enchainement des phases ---
F_heel = np.zeros((3, (number_shooting_points[0] + number_shooting_points[1] + number_shooting_points[2] + 1)))
F_meta1 = np.zeros((3, (number_shooting_points[0] + number_shooting_points[1] + number_shooting_points[2] + 1)))
F_meta5 = np.zeros((3, (number_shooting_points[0] + number_shooting_points[1] + number_shooting_points[2] + 1)))

# phase 1 = heel strike : 1 point de contact talon
F_heel[:, :number_shooting_points[0] + 1] = grf_ref[0]

# phase 2 = flatfoot : 3 points de contact
F_heel[:, number_shooting_points[0]: number_shooting_points[0] + number_shooting_points[1] + 1] = force_heel
F_meta1[:, number_shooting_points[0]: number_shooting_points[0] + number_shooting_points[1] + 1] = force_meta1_3
F_meta5[:, number_shooting_points[0]: number_shooting_points[0] + number_shooting_points[1] + 1] = force_meta5_3

# phase 3 = forefoot : 2 points de contact
F_meta1[:, number_shooting_points[0] + number_shooting_points[1]:] = force_meta1
F_meta5[:, number_shooting_points[0] + number_shooting_points[1]:] = force_meta5

t1 = np.linspace(0, phase_time[0], number_shooting_points[0] + 1)
t2 = t1[-1] + np.linspace(0, phase_time[1], number_shooting_points[1] + 1)
t3 = t2[-1] + np.linspace(0, phase_time[2], number_shooting_points[2] + 1)
t = np.concatenate((t1[:-1], t2[:-1], t3))
for i in range(3):
    G = np.concatenate((grf_ref[0][i, :], grf_ref[1][i, 1:], grf_ref[2][i, 1:]))
    plt.figure(f"contact forces {i + 1}")
    plt.plot(t, G, "k")
    plt.plot(t, F_heel[i, :], "g")
    plt.plot(t, F_meta1[i, :], "r")
    plt.plot(t, F_meta5[i, :], "b")
    plt.plot([phase_time[0], phase_time[0]], [np.min(G), np.max(G)], color="k", linestyle="--", linewidth=1)
    plt.plot([phase_time[0] + phase_time[1], phase_time[0] + phase_time[1]], [np.min(G), np.max(G)], color="k", linestyle="--", linewidth=1)
    plt.grid(color="k", linestyle="--", linewidth=0.5)
    plt.legend(("plateforme", "Heel", "Meta 1", "Meta 5"))
plt.show()