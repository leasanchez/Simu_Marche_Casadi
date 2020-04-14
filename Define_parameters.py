import biorbd

class Parameters():
    def __init__(self):
        # SET MODEL
        self.model_stance_dir = '/home/leasanchez/programmation/Simu_Marche_Casadi/ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact.bioMod'
        self.model_swing_dir  = '/home/leasanchez/programmation/Simu_Marche_Casadi/ModelesS2M/ANsWER_Rleg_6dof_17muscle_0contact.bioMod'

        self.model_stance = biorbd.Model(self.model_stance_dir)               # biorbd model with 1 contact --> stance
        self.model_swing  = biorbd.Model(self.model_stance_dir)               # biorbd model with 0 contact --> swing

        self.nbMus     = self.model_stance.nbMuscleTotal()                    # number of muscles
        self.nbQ       = self.model_stance.nbDof()                            # number of DoFs
        self.nbMarker  = self.model_stance.nbMarkers() - 1                    # number of markers (remove marker to check contact)
        self.nbBody    = self.model_stance.nbSegment()                        # number of body segments
        self.nbContact = self.model_stance.nbContacts()                       # number of contact (2 forces --> plan)

        # SET PROBLEME PARAMETERS
        self.nbNoeuds_stance = 25                                             # shooting points for stance phase
        self.nbNoeuds_swing  = 25                                             # shooting points for swing phase
        self.nbNoeuds        = self.nbNoeuds_stance + self.nbNoeuds_swing     # total shooting points

        self.nbU      = self.nbMus + self.nbQ                                 # number of controls : muscle activation + articular torque (residual torque)
        self.nbX      = 2 * self.nbQ                                          # number of states : generalized positions + velocities
        self.nP       = self.nbMus                                            # number of parameters : 1 per muscle

        self.nkutta   = 4                                                     # number of iteration for integration

        # SET WEIGHTING FACTORS
        self.wL  = 1                                                          # activation
        self.wMa = 30                                                         # anatomical marker
        self.wMt = 50                                                         # technical marker
        self.wU  = 1                                                          # excitation
        self.wR  = 0.05                                                       # ground reaction
        self.wt  = 100                                                        # residual torques

        # EXP DATAS
        self.name_subject = 'equincocont01'
        self.file         = '/home/leasanchez/programmation/Simu_Marche_Casadi/DonneesMouvement/' + self.name_subject + '_out.c3d'
        self.kalman_file  = '/home/leasanchez/programmation/Simu_Marche_Casadi/DonneesMouvement/' + self.name_subject + '_out_MOD5000_leftHanded_GenderF_Florent_.Q2'
        self.save_dir     = '/home/leasanchez/programmation/Simu_Marche_Casadi/Resultats/' + self.name_subject + '/'

        # SET TIME FROM EXP DATAS (init)
        self.T_stance = 0                                                     # stance phase duration
        self.T_swing  = 0                                                     # swing phase duration
        self.T        = self.T_stance + self.T_swing                          # gait cycle duration