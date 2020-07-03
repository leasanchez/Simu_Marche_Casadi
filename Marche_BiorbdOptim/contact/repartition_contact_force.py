import numpy as np
from casadi import vertcat, MX, nlpsol, mtimes, dot, Function, SX, rootfinder
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
M_CoP = Data_to_track.load_data_Moment_at_CoP(biorbd_model[0], T_stance, number_shooting_points)
CoP = Data_to_track.load_data_CoP(biorbd_model[0], T_stance, number_shooting_points)

# contact positions
Heel_pos = np.array([np.mean(markers_ref[1][0, 19, :] + 0.04), np.mean(markers_ref[1][1, 19, :]), 0])
Meta1_pos = np.array([np.mean(markers_ref[1][0, 20, :]), np.mean(markers_ref[1][1, 20, :]), 0])
Meta5_pos = np.array([np.mean(markers_ref[1][0, 24, :]), np.mean(markers_ref[1][1, 24, :]), 0])

# Problem parameters
nb_q = biorbd_model[0].nbQ()
nb_qdot = biorbd_model[0].nbQdot()
nb_tau = biorbd_model[0].nbGeneralizedTorque()
nb_mus = biorbd_model[0].nbMuscleTotal()
nb_contact = biorbd_model[0].nbContacts()

# --- 3 contacts flatfoot - solution analytique ---
sol = np.zeros((6, number_shooting_points[1]+1))
for i in range(number_shooting_points[1] + 1):
    Heel = CoP[1][:, i] - Heel_pos
    Meta1 = CoP[1][:, i] - Meta1_pos
    Meta5 = CoP[1][:, i] - Meta5_pos
    A = np.array([[1, 0, 0, 0, 1, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 1, 0, 1],
                  [0, -Heel[2], Heel[1], Meta1[1], 0, Meta5[1]],
                  [Heel[2], 0, -Heel[0], -Meta1[0], Meta5[2], -Meta5[0]],
                  [-Heel[1], Heel[0], 0, 0, -Meta5[1], 0]])
    sol[:, i] = np.linalg.solve(A, np.concatenate((grf_ref[1][:, i],M_CoP[1][:, i])))

Fx_heel = sol[0, :]
Fy_heel = sol[1, :]
Fz_heel = sol[2, :]
Fz_meta1 = sol[3, :]
Fx_meta5 = sol[4, :]
Fz_meta5 = sol[5, :]

# Forces
figure, axes = plt.subplots(1, 3)
axes = axes.flatten()
axes[0].set_title("contact forces in x")
axes[0].plot(grf_ref[1][0, :], "k--")
axes[0].plot(Fx_heel, "g")
axes[0].plot(Fx_meta5, "b")
axes[0].plot(Fx_heel + Fx_meta5, "m")
axes[0].legend(("plateforme", "Heel", "Meta 5", "sum"))

axes[1].set_title("contact forces in y")
axes[1].plot(grf_ref[1][1, :], "k--")
axes[1].plot(Fy_heel, "g")
axes[1].legend(("plateforme", "Heel"))

axes[2].set_title("contact forces in z")
axes[2].plot(grf_ref[1][2, :], "k--")
axes[2].plot(Fz_heel, "g")
axes[2].plot(Fz_meta5, "b")
axes[2].plot(Fz_meta1, "r")
axes[2].plot(Fz_heel + Fz_meta1 + Fz_meta5, "m")
axes[2].legend(("plateforme", "Heel", "Meta 5", "Meta 1", "sum"))

mx = np.zeros(number_shooting_points[1] + 1)
my = np.zeros(number_shooting_points[1] + 1)
mz = np.zeros(number_shooting_points[1] + 1)
for i in range(number_shooting_points[1] + 1):
    Heel = CoP[1][:, i] - Heel_pos
    Meta1 = CoP[1][:, i] - Meta1_pos
    Meta5 = CoP[1][:, i] - Meta5_pos
    mx[i] = -Heel[2] * Fy_heel[i] + Heel[1]*Fz_heel[i] + Meta1[1]*Fz_meta1[i] + Meta5[1]*Fz_meta5[i]
    my[i] = Heel[2] * Fx_heel[i] - Heel[0] * Fz_heel[i] - Meta1[0]*Fz_meta1[i] + Meta5[2]*Fx_meta5[i] - Meta5[0] * Fz_meta5[i]
    mz[i] = (Heel[0] * Fy_heel[i] - Heel[1] * Fx_heel[i] - Meta5[1] * Fx_meta5[i])

# Moments
figure, axes = plt.subplots(1, 3)
axes = axes.flatten()
axes[0].set_title("moment in x")
axes[0].plot(M_CoP[1][0, :], "k--")
axes[0].plot(mx, "r")
axes[0].set_ylim([-1, 1])
axes[0].legend(("plateforme", "simulation"))

axes[1].set_title("moment in y")
axes[1].plot(M_CoP[1][1, :], "k--")
axes[1].plot(my, "r")
axes[1].set_ylim([-1, 1])
axes[1].legend(("plateforme", "simulation"))

axes[2].set_title("moment in z")
axes[2].plot(M_CoP[1][2, :], "k--")
axes[2].plot(mz, "r")
axes[2].legend(("plateforme", "simulation"))
plt.show()

# --- 3 contact points - optimisation ---
x = np.linspace(-number_shooting_points[1], number_shooting_points[1], number_shooting_points[1] + 1, dtype=int)
p_heel_sig = 1 / (1 + np.exp(-x))
p = 0.5

# Forces
F_Heel = MX.sym("F_Heel", 3 * (number_shooting_points[1] + 1), 1) #xyz
F_Meta1_3 = MX.sym("F_Meta5_3", 1 * (number_shooting_points[1] + 1), 1) #z
F_Meta5_3 = MX.sym("F_Meta1_3", 2 * (number_shooting_points[1] + 1), 1) #xz

objective = 0
lbg = []
ubg = []
constraint = []

for i in range(number_shooting_points[1] + 1):
    # contact point position against CoP
    Heel = CoP[1][:, i] - Heel_pos
    Meta1 = CoP[1][:, i] - Meta1_pos
    Meta5 = CoP[1][:, i] - Meta5_pos

    # Aliases
    fh = F_Heel[3 * i : 3 * (i + 1)]
    fm1 = F_Meta1_3[i]
    fm5 = F_Meta5_3[2 * i : 2 * (i + 1)]

    # --- Torseur equilibre ---
    # sum forces = 0 --> Fp1 + Fp2 + Fh = Ftrack
    jf0 = (fm5[0] + fh[0]) - grf_ref[1][0, i]
    jf1 = fh[1] - grf_ref[1][1, i]
    jf2 = (fm1 + fm5[1] + fh[2]) - grf_ref[1][2, i]
    objective += 100*(jf0*jf0 + jf1*jf1 + jf2*jf2)

    # sum moments = 0 --> CP1xFp1 + CP2xFp2 = Mtrack
    jm0 = (Heel[1]*fh[2] - Heel[2]*fh[1] + Meta1[1]*fm1 + Meta5[1]*fm5[1]) - M_CoP[1][0, i]
    jm1 = (Heel[2]*fh[0] - Heel[0] * fh[2] - Meta1[0] * fm1 + Meta5[2]*fm5[0] - Meta5[0]*fm5[1]) - M_CoP[1][1, i]
    jm2 = (Heel[0] * fh[1] - Heel[1] * fh[0] - Meta5[1] * fm5[0]) - M_CoP[1][2, i]
    objective += 100*(jm0*jm0 + jm1*jm1 + jm2*jm2)

    # --- Dispatch on different contact points ---
    # # use of p to dispatch forces --> p_heel*Fh - (1-p_heel)*Fm = 0
    # jf = p_heel_sig[i] * fh[2] - ((1 - p_heel_sig[i]) * (p * fm1 + (1 - p) * fm5[1]))
    # jff = p_heel_sig[i] * fh[0] - ((1 - p_heel_sig[i]) * fm5[0])
    # objective += jf*jf + jff*jff

    # --- Forces constraints ---
    # positive vertical force
    constraint += (fh[2], fm1, fm5[1])
    lbg += [0] * 3
    ubg += [5000] * 3

    # # non slipping --> -0.4*Fz < Fx < 0.4*Fz
    # constraint += ((-0.6 * fh[2] - fh[0]), (-0.6 * fm5[1] - fm5[0]))
    # lbg += [-1000] * 2
    # ubg += [0] * 2
    #
    # constraint += ((0.6 * fh[2] - fh[0]), (0.6 * fh[2] - fh[1]), (0.6 * fm5[1] - fm5[0]))
    # lbg += [0] * 3
    # ubg += [1000] * 3
end_fh = F_Heel[-3:]
objective += end_fh[0]*end_fh[0] + end_fh[2]*end_fh[2]

init_fm5 = F_Meta5_3[:2]
objective += F_Meta1_3[0]*F_Meta1_3[0] + init_fm5[0]*init_fm5[0] + init_fm5[1]*init_fm5[1]


# x0 = np.zeros(6 * (number_shooting_points[1] + 1))
# x0 = np.random.randint(low=-100, high=1000, size=6 * (number_shooting_points[1] + 1))
x0 = np.concatenate((grf_ref[1][0, :]/2, grf_ref[1][1, :], grf_ref[1][2, :]/3, grf_ref[1][2, :]/3, grf_ref[1][0, :]/2, grf_ref[1][2, :]/3))

w = [F_Heel, F_Meta1_3, F_Meta5_3]
nlp = {"x": vertcat(*w), "f": objective, "g": vertcat(*constraint)}
opts = {"ipopt.tol": 1e-8, "ipopt.hessian_approximation": "exact"}
solver = nlpsol("solver", "ipopt", nlp, opts)
res = solver(x0=x0, lbx=-5000, ubx=5000, lbg=lbg, ubg=ubg)

FH = res["x"][: 3 * (number_shooting_points[1] + 1)]
FM1 = res["x"][3 * (number_shooting_points[1] + 1) : 4 * (number_shooting_points[1] + 1)]
FM5 = res["x"][4 * (number_shooting_points[1] + 1) :]

force_heel = np.zeros((3, (number_shooting_points[1] + 1)))
force_meta5_3 = np.zeros((2, (number_shooting_points[1] + 1)))

Fx_heel = np.array(FH[0::3]).squeeze()
Fy_heel = np.array(FH[1::3]).squeeze()
Fz_heel = np.array(FH[2::3]).squeeze()
Fz_meta1 = np.array(FM1).squeeze()
Fx_meta5 = np.array(FM5[0::2]).squeeze()
Fz_meta5 = np.array(FM5[1::2]).squeeze()

# Forces
figure, axes = plt.subplots(1, 3)
axes = axes.flatten()
axes[0].set_title("contact forces in x")
axes[0].plot(grf_ref[1][0, :], "k--")
axes[0].plot(Fx_heel, "g")
axes[0].plot(Fx_meta5, "b")
axes[0].plot(Fx_heel + Fx_meta5, "m")
axes[0].legend(("plateforme", "Heel", "Meta 5", "sum"))

axes[1].set_title("contact forces in y")
axes[1].plot(grf_ref[1][1, :], "k--")
axes[1].plot(Fy_heel, "g")
axes[1].legend(("plateforme", "Heel"))

axes[2].set_title("contact forces in z")
axes[2].plot(grf_ref[1][2, :], "k--")
axes[2].plot(Fz_heel, "g")
axes[2].plot(Fz_meta5, "b")
axes[2].plot(Fz_meta1, "r")
axes[2].plot(Fz_heel + Fz_meta1 + Fz_meta5, "m")
axes[2].legend(("plateforme", "Heel", "Meta 5", "Meta 1", "sum"))

mx = np.zeros(number_shooting_points[1] + 1)
my = np.zeros(number_shooting_points[1] + 1)
mz = np.zeros(number_shooting_points[1] + 1)
for i in range(number_shooting_points[1] + 1):
    Heel = CoP[1][:, i] - Heel_pos
    Meta1 = CoP[1][:, i] - Meta1_pos
    Meta5 = CoP[1][:, i] - Meta5_pos
    mx[i] = -Heel[2] * Fy_heel[i] + Heel[1]*Fz_heel[i] + Meta1[1]*Fz_meta1[i] + Meta5[1]*Fz_meta5[i]
    my[i] = Heel[2] * Fx_heel[i] - Heel[0] * Fz_heel[i] - Meta1[0]*Fz_meta1[i] + Meta5[2]*Fx_meta5[i] - Meta5[0] * Fz_meta5[i]
    mz[i] = (Heel[0] * Fy_heel[i] - Heel[1] * Fx_heel[i] - Meta5[1] * Fx_meta5[i])

# Moments
figure, axes = plt.subplots(1, 3)
axes = axes.flatten()
axes[0].set_title("moment in x")
axes[0].plot(M_CoP[1][0, :], "k--")
axes[0].plot(mx, "r")
axes[0].legend(("plateforme", "simulation"))

axes[1].set_title("moment in y")
axes[1].plot(M_CoP[1][1, :], "k--")
axes[1].plot(my, "r")
axes[1].legend(("plateforme", "simulation"))

axes[2].set_title("moment in z")
axes[2].plot(M_CoP[1][2, :], "k--")
axes[2].plot(mz, "r")
axes[2].legend(("plateforme", "simulation"))
plt.show()


# --- 2 contacts forefoot ---
p = 0.5  # repartition entre les 2 points

F_Meta1 = MX.sym("F_Meta1", 2 * (number_shooting_points[2] + 1), 1)
F_Meta5 = MX.sym("F_Meta1", 3 * (number_shooting_points[2] + 1), 1)

objective = 0
lbg = []
ubg = []
constraint = []
for i in range(number_shooting_points[2] + 1):
    # Aliases
    fm1 = F_Meta1[2 * i : 2 * (i + 1)]
    fm5 = F_Meta5[3 * i : 3 * (i + 1)]

    # sum forces = 0 --> Fp1 + Fp2 = Ftrack
    jf0 = (fm1[0] + fm5[0]) - grf_ref[2][0, i]
    jf1 = fm5[1] - grf_ref[2][1, i]
    jf2 = (fm1[1] + fm5[2]) - grf_ref[2][2, i]
    objective += jf0*jf0 + jf1*jf1 + jf2*jf2

    # # # sum moments = 0 --> CP1xFp1 + CP2xFp2 = Mtrack
    # sm = dot(Meta1, fm1) + dot(Meta5, fm5)
    # jm = sm - M_ref[2][:, i]
    # constraint += (jm[0], jm[1], jm[2])
    # lbg += [0] * 3
    # ubg += [0] * 3
    # objective += 100 * mtimes(jm.T, jm)

    # use of p to dispatch forces --> p*Fp1 - (1-p)*Fp2 = 0
    jf = p * fm1[1] - (1 - p) * fm5[2]
    objective += jf*jf

    # positive vertical force
    constraint += (fm1[1], fm5[2])
    lbg += [0] * 2
    ubg += [1000] * 2

    # non slipping --> -0.4*Fz < Fx < 0.4*Fz
    constraint += ((-0.4 * fm1[1] - fm1[0]), (-0.4 * fm5[2] - fm5[0]))
    lbg += [-1000] * 2
    ubg += [0] * 2

    constraint += ((0.4 * fm1[1] - fm1[0]), (0.4 * fm5[2] - fm5[0]))
    lbg += [0] * 2
    ubg += [1000] * 2


w = [F_Meta1, F_Meta5]
nlp = {"x": vertcat(*w), "f": objective, "g": vertcat(*constraint)}
opts = {"ipopt.tol": 1e-8, "ipopt.hessian_approximation": "exact"}
solver = nlpsol("solver", "ipopt", nlp, opts)
res = solver(x0=np.zeros(5 * (number_shooting_points[2] + 1)), lbx=-1000, ubx=1000, lbg=lbg, ubg=ubg)
F1 = res["x"][: 2 * (number_shooting_points[2] + 1)]
F5 = res["x"][2 * (number_shooting_points[2] + 1) :]

force_meta1 = np.zeros((2, (number_shooting_points[2] + 1)))
force_meta5 = np.zeros((3, (number_shooting_points[2] + 1)))
for i in range(3):
    if (i == 2):
        force_meta5[i, :] = np.array(F5[i::3]).squeeze()
    else:
        force_meta5[i, :] = np.array(F5[i::3]).squeeze()
        force_meta1[i, :] = np.array(F1[i::2]).squeeze()


# --- 3 contact points ---
p_heel = np.linspace(0, 1, number_shooting_points[1] + 1)
x = np.linspace(-number_shooting_points[1], number_shooting_points[1], number_shooting_points[1] + 1, dtype=int)
p_heel_sig = 1 / (1 + np.exp(-x))
p = 0.5

# contact positions
Heel = np.array([np.mean(markers_ref[1][0, 19, :] + 0.04), np.mean(markers_ref[1][1, 19, :]), 0])
Meta1 = np.array([np.mean(markers_ref[1][0, 20, :]), np.mean(markers_ref[1][1, 20, :]), 0])
Meta5 = np.array([np.mean(markers_ref[1][0, 24, :]), np.mean(markers_ref[1][1, 24, :]), 0])

# Forces
F_Heel = MX.sym("F_Heel", 3 * (number_shooting_points[1] + 1), 1) #xyz
F_Meta1_3 = MX.sym("F_Meta5_3", 1 * (number_shooting_points[1] + 1), 1) #z
F_Meta5_3 = MX.sym("F_Meta1_3", 2 * (number_shooting_points[1] + 1), 1) #xz

objective = 0
lbg = []
ubg = []
constraint = []

for i in range(number_shooting_points[1] + 1):
    # Aliases
    fh = F_Heel[3 * i : 3 * (i + 1)]
    fm1 = F_Meta1_3[i]
    fm5 = F_Meta5_3[2 * i : 2 * (i + 1)]

    # --- Torseur equilibre ---
    # sum forces = 0 --> Fp1 + Fp2 + Fh = Ftrack
    jf0 = (fm5[0] + fh[0]) - grf_ref[1][0, i]
    jf1 = fh[1] - grf_ref[1][1, i]
    jf2 = (fm1 + fm5[1] + fh[2]) - grf_ref[1][2, i]
    objective += jf0*jf0 + jf1*jf1 + jf2*jf2

    # sum moments = 0 --> CP1xFp1 + CP2xFp2 = Mtrack
    jm0 = (Heel[1]*fh[2] - Heel[2]*fh[1] + Meta1[1]*fm1 + Meta5[1]*fm5[1]) - M_ref[1][0, i]
    jm1 = (Heel[2]*fh[0] - Heel[0] * fh[2] - Meta1[0] * fm1 + Meta5[2]*fm5[0] - Meta5[0]*fm5[1]) - M_ref[1][1, i]
    jm2 = (Heel[0] * fh[1] - Heel[1] * fh[0] - Meta5[1] * fm5[0]) - M_ref[1][2, i]
    objective += jm0*jm0 + jm1*jm1 + jm2*jm2

    # --- Dispatch on different contact points ---
    # use of p to dispatch forces --> p_heel*Fh - (1-p_heel)*Fm = 0
    jf = p_heel_sig[i] * fh[2] - ((1 - p_heel_sig[i]) * (p * fm1 + (1 - p) * fm5[1]))
    jff = p_heel_sig[i] * fh[0] - ((1 - p_heel_sig[i]) * fm5[0])
    objective += jf*jf + jff*jff

    # --- Forces constraints ---
    # positive vertical force
    constraint += (fh[2], fm1, fm5[1])
    lbg += [0] * 3
    ubg += [5000] * 3

    # # non slipping --> -0.4*Fz < Fx < 0.4*Fz
    # constraint += ((-0.6 * fh[2] - fh[0]), (-0.6 * fm5[1] - fm5[0]))
    # lbg += [-1000] * 2
    # ubg += [0] * 2
    #
    # constraint += ((0.6 * fh[2] - fh[0]), (0.6 * fh[2] - fh[1]), (0.6 * fm5[1] - fm5[0]))
    # lbg += [0] * 3
    # ubg += [1000] * 3

# x0 = np.zeros(6 * (number_shooting_points[1] + 1))
# x0 = np.random.randint(low=-100, high=1000, size=6 * (number_shooting_points[1] + 1))
x0 = np.concatenate((grf_ref[1][0, :]/2, grf_ref[1][1, :], grf_ref[1][2, :]/3, grf_ref[1][2, :]/3, grf_ref[1][0, :]/2, grf_ref[1][2, :]/3))

w = [F_Heel, F_Meta1_3, F_Meta5_3]
nlp = {"x": vertcat(*w), "f": objective, "g": vertcat(*constraint)}
opts = {"ipopt.tol": 1e-8, "ipopt.hessian_approximation": "exact"}
solver = nlpsol("solver", "ipopt", nlp, opts)
res = solver(x0=x0, lbx=-5000, ubx=5000, lbg=lbg, ubg=ubg)

FH = res["x"][: 3 * (number_shooting_points[1] + 1)]
FM1 = res["x"][3 * (number_shooting_points[1] + 1) : 4 * (number_shooting_points[1] + 1)]
FM5 = res["x"][4 * (number_shooting_points[1] + 1) :]

force_heel = np.zeros((3, (number_shooting_points[1] + 1)))
force_meta5_3 = np.zeros((2, (number_shooting_points[1] + 1)))

force_heel[0, :] = np.array(FH[0::3]).squeeze()
force_heel[1, :] = np.array(FH[1::3]).squeeze()
force_heel[2, :] = np.array(FH[2::3]).squeeze()
force_meta1_3 = np.array(FM1).squeeze()
force_meta5_3[0, :] = np.array(FM5[0::2]).squeeze()
force_meta5_3[1, :] = np.array(FM5[1::2]).squeeze()

m = np.zeros((3, number_shooting_points[1] + 1))
for i in range(number_shooting_points[1] + 1):
    fh = force_heel[:, i]
    fm1 = force_meta1_3[i]
    fm5 = force_meta5_3[:, i]

    m[0, i] = (Heel[1]*fh[2] + Meta1[1]*fm1 + Meta5[1]*fm5[1])
    m[1, i] = (- Heel[0] * fh[2] - Meta1[0] * fm1 - Meta5[0]*fm5[1])
    m[2, i] = (Heel[0] * fh[1] - Heel[1] * fh[0] - Meta5[1] * fm5[0])


# Forces
figure, axes = plt.subplots(1, 3)
axes = axes.flatten()
axes[0].set_title("contact forces in x")
axes[0].plot(grf_ref[1][0, :], "k--")
axes[0].plot(force_heel[0, :], "g")
axes[0].plot(force_meta5_3[0, :], "b")
axes[0].plot(force_meta5_3[0, :] + force_heel[0, :], "m")
axes[0].legend(("plateforme", "Heel", "Meta 5", "sum"))

axes[1].set_title("contact forces in y")
axes[1].plot(grf_ref[1][1, :], "k--")
axes[1].plot(force_heel[1, :], "g")
axes[1].legend(("plateforme", "Heel"))

axes[2].set_title("contact forces in z")
axes[2].plot(grf_ref[1][2, :], "k--")
axes[2].plot(force_heel[2, :], "g")
axes[2].plot(force_meta1_3, "r")
axes[2].plot(force_meta5_3[1, :], "b")
axes[2].plot(force_meta1_3 + force_meta5_3[1, :] + force_heel[2, :], "m")
axes[2].legend(("plateforme", "Heel", "Meta 1", "Meta 5", "sum"))

# Moments
figure, axes = plt.subplots(1, 3)
axes = axes.flatten()
axes[0].set_title("moment in x")
axes[0].plot(M_ref[1][0, :], "k--")
axes[0].plot(m[0, :], "r")
axes[0].legend(("plateforme", "simulation"))

axes[1].set_title("moment in y")
axes[1].plot(M_ref[1][1, :], "k--")
axes[1].plot(m[1, :], "r")
axes[1].legend(("plateforme", "simulation"))

axes[2].set_title("moment in z")
axes[2].plot(M_ref[1][2, :], "k--")
axes[2].plot(m[2, :], "r")
axes[2].legend(("plateforme", "simulation"))
plt.show()

# # Forces
# F_HeelX = SX.sym("F_HeelX", number_shooting_points[1] + 1, 1)
# F_HeelY = SX.sym("F_HeelY", number_shooting_points[1] + 1, 1)
# F_HeelZ = SX.sym("F_HeelZ", number_shooting_points[1] + 1, 1)
#
# F_Meta1Z = SX.sym("F_Meta1Z", number_shooting_points[1] + 1, 1)
#
# F_Meta5X = SX.sym("F_Meta5X", number_shooting_points[1] + 1, 1)
# F_Meta5Z = SX.sym("F_Meta5Z", number_shooting_points[1] + 1, 1)
#
# g = []
# for i in range(number_shooting_points[1] + 1):
#     g.append(F_HeelX[i] + F_Meta5X[i] - grf_ref[1][0, i])
#     g.append(F_HeelY[i] - grf_ref[1][1, i])
#     g.append(F_Meta1Z[i] + F_Meta5Z[i] + F_HeelZ[i] - grf_ref[1][2, i])
#
#     g.append((Heel[1] * F_HeelZ[i] - Heel[2] * F_HeelY[i] + Meta1[1] * F_Meta1Z[i] + Meta5[1] * F_Meta5Z[i]) - M_ref[1][0, i])
#     g.append((Heel[2] * F_HeelX[i] - Heel[0] * F_HeelZ[i] - Meta1[0] * F_Meta1Z[i] + Meta5[2] * F_Meta5X[i] - Meta5[0] * F_Meta5Z[i]) - M_ref[1][1, i])
#     g.append((Heel[0] * F_HeelY[i] - Heel[1] * F_HeelX[i] - Meta5[1] * F_Meta5X[i]) - M_ref[1][2, i])
#
#     g0 = (F_HeelX[i] + F_Meta5X[i] - grf_ref[1][0, i])
#     g1 = (F_HeelY[i] - grf_ref[1][1, i])
#     g2 = (F_Meta1Z[i] + F_Meta5Z[i] + F_HeelZ[i] - grf_ref[1][2, i])
#
#     g3 = ((Heel[1] * F_HeelZ[i] - Heel[2] * F_HeelY[i] + Meta1[1] * F_Meta1Z[i] + Meta5[1] * F_Meta5Z[i]) - M_ref[1][0, i])
#     g4 = ((Heel[2] * F_HeelX[i] - Heel[0] * F_HeelZ[i] - Meta1[0] * F_Meta1Z[i] + Meta5[2] * F_Meta5X[i] - Meta5[0] * F_Meta5Z[i]) - M_ref[1][1, i])
#     g5 = ((Heel[0] * F_HeelY[i] - Heel[1] * F_HeelX[i] - Meta5[1] * F_Meta5X[i]) - M_ref[1][2, i])
#     g = Function('g', [F_HeelX[i], F_HeelY[i], F_HeelZ[i], F_Meta1Z[i], F_Meta5X[i], F_Meta5Z[i]], [g0, g1, g2, g3, g4, g5])
#     G = rootfinder('G', 'newton', g)
#
#
# G = Function('g', [F_HeelX, F_HeelY, F_HeelZ, F_Meta1Z, F_Meta5X, F_Meta5Z], g)

# # --- enchainement des phases ---
# F_heel = np.zeros((3, (number_shooting_points[0] + number_shooting_points[1] + number_shooting_points[2] + 1)))
# F_meta1 = np.zeros((3, (number_shooting_points[0] + number_shooting_points[1] + number_shooting_points[2] + 1)))
# F_meta5 = np.zeros((3, (number_shooting_points[0] + number_shooting_points[1] + number_shooting_points[2] + 1)))
#
# # phase 1 = heel strike : 1 point de contact talon
# F_heel[:, :number_shooting_points[0] + 1] = grf_ref[0]
#
# # phase 2 = flatfoot : 3 points de contact
# F_heel[:, number_shooting_points[0]: number_shooting_points[0] + number_shooting_points[1] + 1] = force_heel
# F_meta1[:, number_shooting_points[0]: number_shooting_points[0] + number_shooting_points[1] + 1] = force_meta1_3
# F_meta5[:, number_shooting_points[0]: number_shooting_points[0] + number_shooting_points[1] + 1] = force_meta5_3

# # phase 3 = forefoot : 2 points de contact
# F_meta1[:, number_shooting_points[0] + number_shooting_points[1]:] = force_meta1
# F_meta5[:, number_shooting_points[0] + number_shooting_points[1]:] = force_meta5
#
# t1 = np.linspace(0, phase_time[0], number_shooting_points[0] + 1)
# t2 = t1[-1] + np.linspace(0, phase_time[1], number_shooting_points[1] + 1)
# t3 = t2[-1] + np.linspace(0, phase_time[2], number_shooting_points[2] + 1)
# t = np.concatenate((t1[:-1], t2[:-1], t3))
# for i in range(3):
#     G = np.concatenate((grf_ref[0][i, :], grf_ref[1][i, 1:], grf_ref[2][i, 1:]))
#     plt.figure(f"contact forces {i + 1}")
#     plt.plot(t, G, "k")
#     plt.plot(t, F_heel[i, :], "g")
#     plt.plot(t, F_meta1[i, :], "r")
#     plt.plot(t, F_meta5[i, :], "b")
#     plt.plot([phase_time[0], phase_time[0]], [np.min(G), np.max(G)], color="k", linestyle="--", linewidth=1)
#     plt.plot([phase_time[0] + phase_time[1], phase_time[0] + phase_time[1]], [np.min(G), np.max(G)], color="k", linestyle="--", linewidth=1)
#     plt.grid(color="k", linestyle="--", linewidth=0.5)
#     plt.legend(("plateforme", "Heel", "Meta 1", "Meta 5"))
# plt.show()