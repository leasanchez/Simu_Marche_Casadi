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

# foot markers position
plt.figure("foot position")
plt.plot(markers_ref[1][0, 19, :] + 0.04, markers_ref[1][1, 19, :], "b+")
plt.text(0.575, 0.153, "talon")
plt.plot(markers_ref[1][0, 21, :], markers_ref[1][1, 21, :], "r+")
plt.text(0.68, 0.165, "FMP1")
plt.plot(markers_ref[1][0, 24, :], markers_ref[1][1, 24, :], "g+")
plt.text(0.68, 0.07, "FM5")
plt.show()

# contact positions
Heel = np.array([np.mean(markers_ref[0][0, 19, :] + 0.04), np.mean(markers_ref[0][1, 19, :]), 0])
Meta1 = np.array([np.mean(markers_ref[0][0, 21, :]), np.mean(markers_ref[0][1, 21, :]), 0])
Meta5 = np.array([np.mean(markers_ref[0][0, 24, :]), np.mean(markers_ref[0][1, 24, :]), 0])

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

label = ["x", "y", "z"]
for i in range(3):
    plt.figure("contact forces" + label[i])
    plt.plot(grf_ref[1][i, :], "k")
    plt.plot(cf[i, :], "g")
    plt.plot(cf[i + 3, :], "r")
    plt.plot(cf[i + 6, :], "b")
    plt.legend(("plateforme", "heel", "Meta 1", "Meta 5"))
plt.show()

# --- 2 contacts forefoot ---
p = 0.5  # repartition entre les 2 points
px = np.repeat(0.5, number_shooting_points[2] + 1)
px[:5] = np.linspace(0, 0.5, 5)


F_Meta1 = MX.sym("F_Meta1", 3 * (number_shooting_points[2] + 1), 1)
F_Meta5 = MX.sym("F_Meta1", 3 * (number_shooting_points[2] + 1), 1)
M_Meta1 = MX.sym("M_Meta1", 3 * (number_shooting_points[2] + 1), 1)
M_Meta5 = MX.sym("M_Meta1", 3 * (number_shooting_points[2] + 1), 1)

objective = 0
lbg = []
ubg = []
constraint = []
for i in range(number_shooting_points[2] + 1):
    # Aliases
    fm1 = F_Meta1[3 * i : 3 * (i + 1)]
    fm5 = F_Meta5[3 * i : 3 * (i + 1)]
    mm1 = M_Meta1[3 * i : 3 * (i + 1)]
    mm5 = M_Meta5[3 * i : 3 * (i + 1)]

    # sum forces = 0 --> Fp1 + Fp2 = Ftrack
    sf = fm1 + fm5
    jf = sf - grf_ref[2][:, i]
    objective += 100 * mtimes(jf.T, jf)

    # sum moments = 0 --> Mp1_P1 + CP1xFp1 + Mp2_P2 + CP2xFp2 = Mtrack
    sm = mm1 + dot(Meta1, fm1) + mm5 + dot(Meta5, fm5)
    jm = sm - M_ref[2][:, i]
    objective += 100 * mtimes(jm.T, jm)

    # use of p to dispatch forces --> p*Fp1 - (1-p)*Fp2 = 0
    jf2 = p * fm1 - (1 - p) * fm5
    objective += mtimes(jf2.T, jf2)

    # # use of p to dispatch forces --> p*Fp1 - (1-p)*Fp2 = 0
    # jf2 = p*fm1[2] - (1-p)*fm5[2]
    # jf3 = px[i]*p * fm1[0] - (1-px[i])*(1 - p) * fm5[0]
    # jf4 = p * fm1[1] - (1 - p) * fm5[1]
    # objective += dot(jf2, jf2) + dot(jf3, jf3) + dot(jf4, jf4)

    # use of p to dispatch moments
    jm2 = p * (mm1 + dot(Meta1, fm1)) - (1 - p) * (mm5 + dot(Meta5, fm5))
    objective += mtimes(jm2.T, jm2)

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

w = [F_Meta1, F_Meta5, M_Meta1, M_Meta5]
nlp = {"x": vertcat(*w), "f": objective, "g": vertcat(*constraint)}
opts = {"ipopt.tol": 1e-8, "ipopt.hessian_approximation": "exact"}
solver = nlpsol("solver", "ipopt", nlp, opts)
res = solver(x0=np.zeros(6 * 2 * (number_shooting_points[2] + 1)), lbx=-1000, ubx=1000, lbg=lbg, ubg=ubg)
F1 = res["x"][: 3 * (number_shooting_points[2] + 1)]
F5 = res["x"][3 * (number_shooting_points[2] + 1) : 6 * (number_shooting_points[2] + 1)]
M1 = res["x"][6 * (number_shooting_points[2] + 1) : 9 * (number_shooting_points[2] + 1)]
M5 = res["x"][9 * (number_shooting_points[2] + 1) :]

force_meta1 = np.zeros((3, (number_shooting_points[2] + 1)))
force_meta5 = np.zeros((3, (number_shooting_points[2] + 1)))
moment_meta1 = np.zeros((3, (number_shooting_points[2] + 1)))
moment_meta5 = np.zeros((3, (number_shooting_points[2] + 1)))
for i in range(3):
    force_meta1[i, :] = np.array(F1[i::3]).squeeze()
    force_meta5[i, :] = np.array(F5[i::3]).squeeze()
    moment_meta1[i, :] = np.array(M1[i::3]).squeeze()
    moment_meta5[i, :] = np.array(M5[i::3]).squeeze()

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
F_Meta1 = MX.sym("F_Meta1", 3 * (number_shooting_points[1] + 1), 1)
F_Meta5 = MX.sym("F_Meta5", 3 * (number_shooting_points[1] + 1), 1)
# Moments
M_Heel = MX.sym("M_Heel", 3 * (number_shooting_points[1] + 1), 1)
M_Meta1 = MX.sym("M_Meta1", 3 * (number_shooting_points[1] + 1), 1)
M_Meta5 = MX.sym("M_Meta5", 3 * (number_shooting_points[1] + 1), 1)

objective = 0
lbg = []
ubg = []
constraint = []
for i in range(number_shooting_points[1] + 1):
    # Aliases
    fh = F_Heel[3 * i : 3 * (i + 1)]
    fm1 = F_Meta1[3 * i : 3 * (i + 1)]
    fm5 = F_Meta5[3 * i : 3 * (i + 1)]
    mh = M_Heel[3 * i : 3 * (i + 1)]
    mm1 = M_Meta1[3 * i : 3 * (i + 1)]
    mm5 = M_Meta5[3 * i : 3 * (i + 1)]

    # --- Torseur equilibre ---
    # sum forces = 0 --> Fp1 + Fp2 + Fh = Ftrack
    sf = fm1 + fm5 + fh
    jf = sf - grf_ref[1][:, i]
    objective += 100 * mtimes(jf.T, jf)
    # sum moments = 0 --> Mp1_P1 + CP1xFp1 + Mp2_P2 + CP2xFp2 = Mtrack
    sm = mm1 + dot(Meta1, fm1) + mm5 + dot(Meta5, fm5) + mh + dot(Heel, fh)
    jm = sm - M_ref[1][:, i]
    objective += 100 * mtimes(jm.T, jm)

    # --- Dispatch on different contact points ---
    # use of p to dispatch forces --> p_heel*Fh - (1-p_heel)*Fm = 0
    jf2 = p_heel_sig[i] * fh - ((1 - p_heel_sig[i]) * (p * fm1 + (1 - p) * fm5))
    objective += mtimes(jf2.T, jf2)
    # # use of p to dispatch forces --> p_heel*Fh - (1-p_heel)*Fm = 0
    # jf2 = p_heel[i] * fh[2] - ((1 - p_heel[i]) * (p * fm1[2] + (1 - p) * fm5[2]))
    # jf3 = p_heel[i] * fh[0] - ((1 - p_heel[i]) * (1 * fm1[0] + 0 * fm5[0]))
    # jf32 = fm5[0]
    # jf4 = p_heel[i] * fh[1] - ((1 - p_heel[i]) * (p * fm1[1] + (1 - p) * fm5[1]))
    # objective += dot(jf2, jf2) + dot(jf3, jf3) + dot(jf32, jf32) + dot(jf4, jf4)

    # --- Forces constraints ---
    # positive vertical force
    constraint += (fh[2], fm1[2], fm5[2])
    lbg += [0] * 3
    ubg += [1000] * 3
    # non slipping --> -0.4*Fz < Fx < 0.4*Fz
    constraint += ((-0.4 * fh[2] - fh[0]), (-0.4 * fm1[2] - fm1[0]), (-0.4 * fm5[2] - fm5[0]))
    lbg += [-1000] * 3
    ubg += [0] * 3

    constraint += ((0.4 * fh[2] - fh[0]), (0.4 * fm1[2] - fm1[0]), (0.4 * fm5[2] - fm5[0]))
    lbg += [0] * 3
    ubg += [1000] * 3

w = [F_Heel, F_Meta1, F_Meta5, M_Heel, M_Meta1, M_Meta5]
nlp = {"x": vertcat(*w), "f": objective, "g": vertcat(*constraint)}
opts = {"ipopt.tol": 1e-8, "ipopt.hessian_approximation": "exact"}
solver = nlpsol("solver", "ipopt", nlp, opts)
res = solver(x0=np.zeros(9 * 2 * (number_shooting_points[1] + 1)), lbx=-1000, ubx=1000, lbg=lbg, ubg=ubg)

FH = res["x"][: 3 * (number_shooting_points[1] + 1)]
FM1 = res["x"][3 * (number_shooting_points[1] + 1) : 6 * (number_shooting_points[1] + 1)]
FM5 = res["x"][6 * (number_shooting_points[1] + 1) : 9 * (number_shooting_points[1] + 1)]

force_heel = np.zeros((3, (number_shooting_points[1] + 1)))
force_meta1 = np.zeros((3, (number_shooting_points[1] + 1)))
force_meta5 = np.zeros((3, (number_shooting_points[1] + 1)))
for i in range(3):
    force_heel[i, :] = np.array(FH[i::3]).squeeze()
    force_meta1[i, :] = np.array(FM1[i::3]).squeeze()
    force_meta5[i, :] = np.array(FM5[i::3]).squeeze()

for i in range(3):
    plt.figure(f"contact forces {i + 1}")
    plt.plot(grf_ref[1][i, :], "k")
    plt.plot(force_heel[i, :], "g")
    plt.plot(force_meta1[i, :], "r")
    plt.plot(force_meta5[i, :], "b")
    plt.legend(("plateforme", "Heel", "Meta 1", "Meta 5"))
plt.show()
