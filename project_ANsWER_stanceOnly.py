import biorbd
from ezc3d import c3d
from casadi import *
from pylab import *
import numpy as np
import BiorbdViz
from Marche_Fcn_Integration import *
from Fcn_Objective import *
from LoadData import *
from Fcn_Affichage import *

SetForceISO                  = external('libforce_iso_max', 'libforce_iso_max.so',{'enable_fd':True})

# ----------------------------- Model ----------------------------------------------------------------------------------
model = biorbd.Model('/home/leasanchez/programmation/Marche_Florent/ModelesS2M/ANsWER_Rleg_6dof_17muscle_1contact.bioMod')
GaitPhase = 'stance'

nbNoeuds  = 25                             # nodes
nbMus     = model.nbMuscleTotal()          # muscles
nbQ       = model.nbDof()                  # DoFs : q and dq
nbMarker  = model.nbMarkers()              # markers
nbBody    = model.nbSegment()              # body segments
nbContact = model.nbContacts()             # contacts
nbU       = nbMus + 3
nbX       = 2*nbQ + nbMus
nP        = nbMus + 1

nkutta    = 5                 # iteration for the integration