import numpy as np
from casadi import vertcat, MX, nlpsol, mtimes
from matplotlib import pyplot as plt
import biorbd
from Marche_BiorbdOptim.LoadData import Data_to_track

biorbd_model = (
    biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_1contact_deGroote_3d_Heel.bioMod"),
    biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_2contacts_deGroote_3d.bioMod"),
    biorbd.Model("../../ModelesS2M/Marche_saine/ANsWER_Rleg_6dof_17muscle_1contact_deGroote_3d_Forefoot.bioMod"),
)
# Problem parameters
number_shooting_points = [5, 10, 15]

# Generate data from file
Data_to_track = Data_to_track("equincocont01", multiple_contact=True)
[T, T_stance, T_swing] = Data_to_track.GetTime()
phase_time = T_stance
grf_ref = Data_to_track.load_data_GRF(biorbd_model, T_stance, number_shooting_points)
M_ref = Data_to_track.load_data_Moment(biorbd_model, T_stance, number_shooting_points)

# contact positions
Center_Foot = Data_to_track.GetFootCenterPosition()
Heel = [0.0054278, 0.00220395, 0]
Meta1 = [0.156485, -0.0323195, 0]
Meta5 = [ 0.156485, 0.0638694, 0]

# --- 2 contacts forefoot ---
p = 0.5 # repartition entre les 2 points

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
    # p = p_init[i]
    fm1 = F_Meta1[3*i : 3*(i+1)]
    fm5 = F_Meta5[3 * i: 3 * (i + 1)]
    sf = fm1 + fm5

    # sum forces = 0 --> Fp1 + Fp2 = Ftrack
    j = sf - grf_ref[2][:, i]
    objective += 100 * mtimes(j.T, j)

    # sum moments = 0 --> Mp1_P1 + CP1xFp1 + Mp2_P2 + CP2xFp2 = Mtrack

    # use of p to dispatch forces --> p*Fp1 - (1-p)*Fp2 = 0
    j2 = p*fm1 - (1-p)*fm5
    objective += mtimes(j2.T, j2)

    # positive vertical force
    constraint += (fm1[2], fm5[2])
    lbg += [0]*2
    ubg += [1000]*2

w = [F_Meta1, F_Meta5]
nlp = {'x': vertcat(*w), 'f': objective, 'g': vertcat(*constraint)}
opts = {"ipopt.tol": 1e-8, "ipopt.hessian_approximation": "exact"}
solver = nlpsol("solver", "ipopt", nlp, opts)
res = solver(x0=np.zeros(6*(number_shooting_points[2] + 1)),
             lbx=-1000,
             ubx=1000,
             lbg=lbg,
             ubg=ubg)
F1 = res['x'][:3 * (number_shooting_points[2] + 1)]
F5 = res['x'][3 * (number_shooting_points[2] + 1):]
force_meta1 = np.zeros((3, (number_shooting_points[2] + 1)))
force_meta5 = np.zeros((3, (number_shooting_points[2] + 1)))
for i in range(3):
    force_meta1[i, :] = np.array(F1[i::3]).squeeze()
    force_meta5[i, :] = np.array(F5[i::3]).squeeze()

plt.figure('contact forces')
plt.plot(grf_ref[2][2, :], 'k')
plt.plot(force_meta1[2, :], 'r')
plt.plot(force_meta5[2, :], 'b')
plt.legend(('plateforme', 'Meta 1', 'Meta 5'))
plt.show()


# --- 3 contact points ---
p_heel = np.linspace(0, 1, number_shooting_points[1] + 1)
# p_heel = 1 - p_m

F_Heel = MX.sym("F_Heel", 3 * (number_shooting_points[1] + 1), 1)
F_Meta1 = MX.sym("F_Meta1", 3 * (number_shooting_points[1] + 1), 1)
F_Meta5 = MX.sym("F_Meta5", 3 * (number_shooting_points[1] + 1), 1)

# M_Heel = MX.sym("M_Meta1", 3 * (number_shooting_points[1] + 1), 1)
# M_Meta1 = MX.sym("M_Meta1", 3 * (number_shooting_points[1] + 1), 1)
# M_Meta5 = MX.sym("M_Meta1", 3 * (number_shooting_points[1] + 1), 1)

objective = 0
lbg = []
ubg = []
constraint = []
for i in range(number_shooting_points[1] + 1):
    # Aliases
    fh = F_Heel[3 * i: 3 * (i + 1)]
    fm1 = F_Meta1[3 * i: 3 * (i + 1)]
    fm5 = F_Meta5[3 * i: 3 * (i + 1)]
    sf = fm1 + fm5 + fh

    # sum forces = 0 --> Fp1 + Fp2 + Fh = Ftrack
    j = sf - grf_ref[1][:, i]
    objective += 100 * mtimes(j.T, j)

    # sum moments = 0 --> Mp1_P1 + CP1xFp1 + Mp2_P2 + CP2xFp2 = Mtrack

    # use of p to dispatch forces --> p_heel*Fh - (1-p_heel)*Fm = 0
    j2 = p_heel[i] * fh - ((1 - p_heel[i]) * (p * fm1 + (1 - p) * fm5))
    objective += mtimes(j2.T, j2)

    # positive vertical force
    constraint += (fh[2], fm1[2], fm5[2])
    lbg += [0] * 3
    ubg += [1000] * 3

w = [F_Heel, F_Meta1, F_Meta5]
nlp = {'x': vertcat(*w), 'f': objective, 'g': vertcat(*constraint)}
opts = {"ipopt.tol": 1e-8, "ipopt.hessian_approximation": "exact"}
solver = nlpsol("solver", "ipopt", nlp, opts)
res = solver(x0=np.zeros(9 * (number_shooting_points[1] + 1)),
             lbx=-1000,
             ubx=1000,
             lbg=lbg,
             ubg=ubg)

FH = res['x'][:3 * (number_shooting_points[1] + 1)]
FM1 = res['x'][3 * (number_shooting_points[1] + 1): 6 * (number_shooting_points[1] + 1)]
FM5 = res['x'][6 * (number_shooting_points[1] + 1):]

force_heel = np.zeros((3, (number_shooting_points[1] + 1)))
force_meta1 = np.zeros((3, (number_shooting_points[1] + 1)))
force_meta5 = np.zeros((3, (number_shooting_points[1] + 1)))
for i in range(3):
    force_heel[i, :] = np.array(FH[i::3]).squeeze()
    force_meta1[i, :] = np.array(FM1[i::3]).squeeze()
    force_meta5[i, :] = np.array(FM5[i::3]).squeeze()

plt.figure('contact forces 3 points')
plt.plot(grf_ref[1][2, :], 'k')
plt.plot(force_heel[2, :], 'g')
plt.plot(force_meta1[2, :], 'r')
plt.plot(force_meta5[2, :], 'b')
plt.legend(('plateforme', 'Heel', 'Meta 1', 'Meta 5'))
plt.show()