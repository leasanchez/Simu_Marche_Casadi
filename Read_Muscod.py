import numpy as np

class Muscod():
    def __init__(self, name_subject):
        self.name_subject = name_subject
        self.file = '/home/leasanchez/programmation/Marche_Florent/ResultatsSimulation/' + self.name_subject + '_out/RES/ANsWER_gaitCycle_works4.txt'
        self.nbQ = 6
        self.nbX = 12 + 17
        self.nbU = 20
        self.nP = 17 + 1
        self.nbNoeuds = 51
        self.nbNoeuds_stance = 25
        self.nbNoeuds_swing = 25
        self.T_stance = 0.605
        self.T_swing = 0.37

        self.t_stance = np.linspace(0, self.T_stance, self.nbNoeuds_stance + 1)
        self.t_swing = np.linspace(0, self.T_swing, self.nbNoeuds_swing + 1)
        self.t = np.hstack([self.t_stance[:-1], self.T_stance + self.t_swing])

        self.U_muscod, self.X_muscod, self.P_muscod = self.Read_MUSCOD()

    def Read_MUSCOD(self):
        f = open(self.file, 'r')
        content = f.read()
        content_divide = content.split('\n')

        # FIND STATE -- sd
        x0 = np.zeros((self.nbX, (self.nbNoeuds + 1)))
        # phase 1 -- stance
        for n in range(self.nbNoeuds_stance):
            state = 'sd(0,' + str(n) + ')  ! ' + str(n)
            idx = content_divide.index(state)
            for x in range(self.nbX):
                a = content_divide[idx + x + 1].split(':')
                x0[x, n] = float(a[1])
        # phase 2 -- swing
        for n in range(self.nbNoeuds_swing):
            state = 'sd(1,' + str(n) + ')  ! ' + str(self.nbNoeuds_stance + n)
            idx = content_divide.index(state)
            for x in range(self.nbX):
                a = content_divide[idx + x + 1].split(':')
                x0[x, (self.nbNoeuds_stance + n)] = float(a[1])

        # phase 3 -- impact
        state = 'sd(2,0)  ! 50'
        idx = content_divide.index(state)
        for x in range(self.nbX):
            a = content_divide[idx + x + 1].split(':')
            x0[x, (self.nbNoeuds_stance + self.nbNoeuds_swing)] = float(a[1])
        state = 'sd(2,1)  ! 51'
        idx = content_divide.index(state)
        for x in range(self.nbX):
            a = content_divide[idx + x + 1].split(':')
            x0[x, self.nbNoeuds] = float(a[1])


        # FIND CONTROL -- u
        u0 = np.zeros((self.nbU, self.nbNoeuds))
        for n in range(self.nbNoeuds_stance):
            control = 'u(0,' + str(n) + ')  ! ' + str(n)
            idx = content_divide.index(control)
            for u in range(self.nbU):
                a = content_divide[idx + u + 1].split(':')
                u0[u, n] = float(a[1])
        # phase 2 -- swing
        for n in range(self.nbNoeuds_swing):
            control = 'u(1,' + str(n) + ')  ! ' + str(self.nbNoeuds_stance + n)
            idx = content_divide.index(control)
            for u in range(self.nbU):
                a = content_divide[idx + u + 1].split(':')
                u0[u, (self.nbNoeuds_stance + n)] = float(a[1])
        # phase 3 -- impact
        control = 'u(2,0)  ! 50'
        idx = content_divide.index(control)
        for u in range(self.nbU):
            a = content_divide[idx + u + 1].split(':')
            u0[u, (self.nbNoeuds - 1)] = float(a[1])


        # FIND PARAMETERS FOR ISOMETRIC FORCE
        p0    = np.zeros(self.nP)
        param = 'p'
        idx   = content_divide.index(param)
        for p in range(self.nP):
            a = content_divide[idx + p + 1].split(':')
            p0[p] = float(a[1])

        f.close()
        return u0, x0, p0

