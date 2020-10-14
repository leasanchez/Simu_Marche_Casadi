import numpy as np
from matplotlib import pyplot as plt
from bioptim import Data

def get_q_name(model):
    q_name = []
    for s in range(model.nbSegment()):
        seg_name = model.segment(s).name().to_string()
        for d in range(model.segment(s).nbDof()):
            dof_name = model.segment(s).nameDof(d).to_string()
            q_name.append(seg_name + "_" + dof_name)
    return q_name

def get_time_vector(phase_time, number_shooting_points):
    nb_phases = len(phase_time)
    if (nb_phases > 1):
        t = np.linspace(0, phase_time[0], number_shooting_points[0] + 1)
        for p in range(1, nb_phases):
            t = np.concatenate((t[:-1], t[-1] + np.linspace(0, phase_time[p], number_shooting_points[p] + 1)))
    else:
        t = np.linspace(0, phase_time, number_shooting_points + 1)
    return t

def compute_individual_forces(ocp, sol, two_leg=False, muscles=False):
    # --- total number of shooting points ---
    nb_phases = ocp.nb_phases
    if (nb_phases>1):
        nb_shooting = 0
        for p in range(nb_phases):
            nb_shooting += ocp.nlp[p].ns
    else:
        nb_shooting = ocp.nlp[0].ns

    if two_leg:
        labels_forces = ['Heel_r_X', 'Heel_r_Y', 'Heel_r_Z',
                         'Meta_1_r_X', 'Meta_1_r_Y', 'Meta_1_r_Z',
                         'Meta_5_r_X', 'Meta_5_r_Y', 'Meta_5_r_Z',
                         'Heel_l_X', 'Heel_l_Y', 'Heel_l_Z',
                         'Meta_1_l_X', 'Meta_1_l_Y', 'Meta_1_l_Z',
                         'Meta_5_l_X', 'Meta_5_l_Y', 'Meta_5_l_Z'
                         ]
    else:
        labels_forces = ['Heel_r_X', 'Heel_r_Y', 'Heel_r_Z',
                     'Meta_1_r_X', 'Meta_1_r_Y', 'Meta_1_r_Z',
                     'Meta_5_r_X', 'Meta_5_r_Y', 'Meta_5_r_Z',]
    # --- dictionary for forces ---
    forces = {}
    for label in labels_forces:
        forces[label] = np.zeros(nb_shooting + 1)

    # COMPUTE FORCES FOR EACH PHASE
    if (nb_phases>1):
        n_shoot = 0
        for p in range(nb_phases):
            cn = ocp.nlp[p].model.contactNames() # get contact names for each model
            if (len(cn) > 0):
                states_p, controls_p = Data.get_data(ocp, sol["x"], phase_idx=p)
                Q = states_p["q"]
                Qdot = states_p["q_dot"]
                Tau = controls_p["tau"]
                if muscles:
                    Activation = controls_p["muscles"]
                for n in range(ocp.nlp[p].ns + 1):
                    s = np.concatenate([Q[:, n], Qdot[:, n]])
                    if muscles:
                        c = np.concatenate([Tau[:, n], Activation[:, n]])
                    else:
                        c=Tau[:, n]
                    forces_sim = ocp.nlp[p].contact_forces_func(s, c, 0)
                    for i, c in enumerate(cn):
                        if c.to_string() in forces:
                            forces[c.to_string()][n_shoot + n] = forces_sim[i]  # put corresponding forces in dictionnary
                n_shoot += ocp.nlp[p].ns
    else:
        cn = ocp.nlp[0].model.contactNames()
        states_p, controls_p = Data.get_data(ocp, sol["x"])
        Q = states_p["q"]
        Qdot = states_p["q_dot"]
        Tau = controls_p["tau"]
        if muscles:
            Activation = controls_p["muscles"]
        for n in range(ocp.nlp[0].ns + 1):
            s = np.concatenate([Q[:, n], Qdot[:, n]])
            if muscles:
                c=np.concatenate([Tau[:, n], Activation[:, n]])
            else:
                c = Tau[:, n]
            forces_sim = ocp.nlp[0].contact_forces_func(s, c, 0)
            for i, c in enumerate(cn):
                if c.to_string() in forces:
                    forces[c.to_string()][n] = forces_sim[i]
    return forces

def compute_mean_difference(x, x_ref):
    nb_x = x.shape[0]
    nb_phases = len(x_ref)
    mean_diff = []
    for i in range(nb_x):
        if (nb_phases > 1):
            X_ref = x_ref[0][i, :]
            for p in range(1, nb_phases):
                X_ref = np.concatenate([X_ref[:-1], x_ref[p][i, :]])
        else:
            X_ref = x_ref
        mean_diff.append(np.mean(np.sqrt((x[i, :] - X_ref) ** 2)))
    return mean_diff

def compute_max_difference(x, x_ref):
    nb_x = x.shape[0]
    nb_phases = len(x_ref)
    max_diff = []
    idx_max = []
    for i in range(nb_x):
        if (nb_phases > 1):
            X_ref = x_ref[0][i, :]
            for p in range(1, nb_phases):
                X_ref = np.concatenate([X_ref[:-1], x_ref[p][i, :]])
        else:
            X_ref = x_ref
        diff = np.sqrt((x[i, :] - X_ref) ** 2)
        max_diff.append(np.max(diff))
        idx_max.append(np.where(diff == np.max(diff))[0][0])
    return idx_max, max_diff

def compute_R2(x, x_ref):
    nb_x = x.shape[0]
    nb_phases = len(x_ref)
    R2 = []
    for i in range(nb_x):
        mean_x = np.repeat(np.mean(x[i, :]), x.shape[1])
        if (nb_phases > 1):
            X_ref = x_ref[0][i, :]
            for p in range(1, nb_phases):
                X_ref = np.concatenate([X_ref[:-1], x_ref[p][i, :]])
        else:
            X_ref = x_ref
        diff_q_square = (x[i, :] - X_ref)**2
        diff_q_mean = (x[i, :] - mean_x)**2
        if (np.sum(diff_q_square) > np.sum(diff_q_mean)):
            s = np.sum(diff_q_mean) / np.sum(diff_q_square)
        else:
            s = np.sum(diff_q_square) / np.sum(diff_q_mean)
        R2.append(1-s)
    return R2


def plot_q(biorbd_model, phase_time, number_shooting_points, q, q_ref):
    # --- multiphase ---
    nb_phases = len(biorbd_model)
    if (nb_phases>1):
        model = biorbd_model[0]
    else:
        model = biorbd_model

    t = get_time_vector(phase_time, number_shooting_points) # time vector
    q_name=get_q_name(model) # dof names

    # --- plot q VS q_ref ---
    nb_q = model.nbQ()
    n_column = int(nb_q/3) + (nb_q % 3 > 0)
    figure, axes = plt.subplots(3,n_column)
    axes = axes.flatten()
    for i in range(nb_q):
        axes[i].plot(t, q[i, :], 'r-', alpha=0.5)
        # compute q_ref
        if (nb_phases>1):
            Q_ref = q_ref[0][i, :]
            for p in range(1, nb_phases):
                Q_ref = np.concatenate([Q_ref[:-1], q_ref[p][i, :]])
        else:
            Q_ref = q_ref[i, :]
        axes[i].plot(t, Q_ref, 'b-', alpha=0.5)
        axes[i].scatter(t, q[i, :], color='r', s=3)
        axes[i].scatter(t, Q_ref, color='b', s=3)

        # plot phase transition
        if (nb_phases>1):
            pt=0
            for p in range(nb_phases):
                pt += phase_time[p]
                axes[i].plot([pt, pt], [np.min(Q_ref), np.max(Q_ref)], 'k--')
        axes[i].set_title(q_name[i])
    plt.legend(['simulated', 'reference'])

def plot_tau(biorbd_model, phase_time, number_shooting_points, tau):
    # --- multiphase ---
    nb_phases = len(biorbd_model)
    if (nb_phases>1):
        model = biorbd_model[0]
    else:
        model = biorbd_model

    # --- time vector ---
    t = get_time_vector(phase_time, number_shooting_points)

    # --- dof names ---
    q_name=get_q_name(model)

    # --- plot q VS q_ref ---
    nb_q = model.nbQ()
    n_column = int(nb_q/3) + (nb_q % 3 > 0)
    figure, axes = plt.subplots(3,n_column)
    axes = axes.flatten()
    for i in range(nb_q):
        axes[i].plot(t, tau[i, :], 'r-')
        axes[i].plot([t[0], t[-1]], [0, 0], 'k--')
        # plot phase transition
        if (nb_phases>1):
            pt=0
            for p in range(nb_phases):
                pt += phase_time[p]
                axes[i].plot([pt, pt], [np.min(tau[i, :]), np.max(tau[i, :])], 'k--')
        axes[i].set_title(q_name[i])

def plot_individual_forces(ocp, sol, two_leg=False, muscles=False):
    forces=compute_individual_forces(ocp, sol, two_leg, muscles)
    nb_phases = ocp.nb_phases

    if (nb_phases>1):
        phase_time = []
        number_shooting_points = []
        for p in range(nb_phases):
            phase_time.append(ocp.nlp[p].tf)
            number_shooting_points.append(ocp.nlp[p].ns)
        t = get_time_vector(phase_time, number_shooting_points)
    else:
        phase_time = ocp.nlp[0].tf
        t = get_time_vector(ocp.nlp[0].tf, ocp.nlp[0].ns)

    if two_leg:
        figureR, axesR = plt.subplots(3, 3)
        axesR = axesR.flatten()
        figureL, axesL = plt.subplots(3, 3)
        axesL = axesL.flatten()

        for i, f in enumerate(forces):
            if (i<9):
                ax = axesR[i]
            else:
                ax = axesL[i-9]
            ax.scatter(t, forces[f], color='r', s=3)
            ax.plot(t, forces[f], 'r-', alpha=0.5)
            if (i == 2) or (i == 5) or (i == 8):
                ax.plot([t[0], t[-1]], [0, 0], 'k--')
            ax[i].set_title(f)
            if (nb_phases > 1):
                pt = 0
                for p in range(nb_phases):
                    pt += phase_time[p]
                    ax.plot([pt, pt], [np.min(forces[f]), np.max(forces[f])], 'k--')

    else:
        figure, axes = plt.subplots(3, 3)
        axes = axes.flatten()
        for i, f in enumerate(forces):
            axes[i].scatter(t, forces[f], color='r', s=3)
            axes[i].plot(t, forces[f], 'r-', alpha=0.5)
            if (i==2) or (i==5) or (i==8):
                axes[i].plot([t[0], t[-1]], [0, 0], 'k--')
            axes[i].set_title(f)
            if (nb_phases>1):
                pt = 0
                for p in range(nb_phases):
                    pt += phase_time[p]
                    axes[i].plot([pt, pt], [np.min(forces[f]), np.max(forces[f])], 'k--')

def plot_sum_forces(ocp, sol, grf_ref, two_leg=False, muscles=False):
    # PLOT SUM FORCES VS PLATEFORME FORCES
    forces=compute_individual_forces(ocp, sol, two_leg, muscles)
    nb_phases = ocp.nb_phases

    # --- time vector ---
    if (nb_phases>1):
        phase_time = []
        number_shooting_points = []
        for p in range(nb_phases):
            phase_time.append(ocp.nlp[p].tf)
            number_shooting_points.append(ocp.nlp[p].ns)
        t = get_time_vector(phase_time, number_shooting_points)
    else:
        phase_time = ocp.nlp[0].tf
        t = get_time_vector(ocp.nlp[0].tf, ocp.nlp[0].ns)

    # --- contact forces references ---
    coord_label = ['X', 'Y', 'Z']
    if (nb_phases>1):
        forces_ref = {}
        if two_leg:
            forces_ref["force_X_R"] = grf_ref[0][0][0, :]
            forces_ref["force_Y_R"] = grf_ref[0][0][1, :]
            forces_ref["force_Z_R"] = grf_ref[0][0][2, :]
            forces_ref["force_X_L"] = grf_ref[0][1][0, :]
            forces_ref["force_Y_L"] = grf_ref[0][1][1, :]
            forces_ref["force_Z_L"] = grf_ref[0][1][2, :]
        else:
            forces_ref["force_X_R"] = grf_ref[0][0, :]
            forces_ref["force_Y_R"] = grf_ref[0][1, :]
            forces_ref["force_Z_R"] = grf_ref[0][2, :]
        for p in range(1, nb_phases):
            for i, f in enumerate(forces_ref):
                forces_ref[f] = np.concatenate([forces_ref[f][:-1], grf_ref[p][i, :]])
                if two_leg:
                    if (i<3): #R
                        forces_ref[f] = np.concatenate([forces_ref[f][:-1], grf_ref[p][0][i, :]])
                    else:
                        forces_ref[f] = np.concatenate([forces_ref[f][:-1], grf_ref[p][1][i-3, :]])

    else:
        forces_ref = {}
        if two_leg:
            forces_ref["force_X_R"] = grf_ref[0][0, :]
            forces_ref["force_Y_R"] = grf_ref[0][1, :]
            forces_ref["force_Z_R"] = grf_ref[0][2, :]
            forces_ref["force_X_L"] = grf_ref[1][0, :]
            forces_ref["force_Y_L"] = grf_ref[1][1, :]
            forces_ref["force_Z_L"] = grf_ref[1][2, :]
        else:
            forces_ref["force_X_R"] = grf_ref[0, :]
            forces_ref["force_Y_R"] = grf_ref[1, :]
            forces_ref["force_Z_R"] = grf_ref[2, :]

    figure, axes = plt.subplots(1, 3)
    axes = axes.flatten()
    for i in range(3):
        axes[i].plot(t,
                     forces[f"Heel_r_{coord_label[i]}"]
                     + forces[f"Meta_1_r_{coord_label[i]}"]
                     + forces[f"Meta_5_r_{coord_label[i]}"],
                     'r-', alpha=0.5)
        axes[i].plot(t, forces_ref[f"force_{coord_label[i]}_R"], 'b-', alpha=0.5)
        axes[i].scatter(t,
                     forces[f"Heel_r_{coord_label[i]}"]
                     + forces[f"Meta_1_r_{coord_label[i]}"]
                     + forces[f"Meta_5_r_{coord_label[i]}"],
                     color='r', s=3)
        axes[i].scatter(t, forces_ref[f"force_{coord_label[i]}_R"], color='b', s=3)
        axes[i].set_title("Forces in " + coord_label[i] + " R")
        if (nb_phases > 1):
            pt = 0
            for p in range(nb_phases):
                pt += phase_time[p]
                axes[i].plot([pt, pt],
                             [np.min(forces_ref[f"force_{coord_label[i]}_R"]), np.max(forces_ref[f"force_{coord_label[i]}_R"])],
                             'k--')
    plt.legend(['simulated', 'reference'])

def plot_activation(biorbd_model, phase_time, number_shooting_points, activations, excitations_ref):
    # --- multiphase ---
    nb_phases = len(biorbd_model)
    if (nb_phases>1):
        model = biorbd_model[0]
    else:
        model = biorbd_model

    # --- time vector ---
    t = get_time_vector(phase_time, number_shooting_points)

    nb_mus = model.nbMuscleTotal()
    figure, axes = plt.subplots(3, 6)
    axes = axes.flatten()
    for i in range(nb_mus):
        axes[i].plot(t, activations[i, :], 'r-')
        if (nb_phases>1):
            a_plot = excitations_ref[0][i, :]
            for p in range(1, nb_phases):
                a_plot = np.concatenate([a_plot[:-1], excitations_ref[p][i, :]])
        else:
            a_plot = excitations_ref[i, :]
        axes[i].plot(t, a_plot, 'b--')
        if (nb_phases > 1):
            pt=0
            for p in range(nb_phases):
                pt += phase_time[p]
                axes[i].plot([pt, pt], [np.min(a_plot), np.max(a_plot)], 'k--')
        axes[i].set_title(model.muscle(i).name().to_string())