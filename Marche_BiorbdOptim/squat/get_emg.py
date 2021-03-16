import matplotlib.pyplot as plt
import numpy as np
from ezc3d import c3d
from pyomeca import Analogs

def emg_processed(file_path, name):
    emg = Analogs.from_c3d(file_path, usecols=[name])
    emg_process = (
        emg.meca.band_pass(order=2, cutoff=[10, 425])
            .meca.center()
            .meca.abs()
            .meca.low_pass(order=4, cutoff=5, freq=emg.rate)
    )
    return emg_process

def get_mvc_value(data):
    max = 0
    for d in data:
        max+=np.max(d)
    return max/len(data)

path = 'Data_test/MVC/'
emg_mvc_path = path + 'MVC_GlutMax.c3d'
c=c3d(emg_mvc_path)
label=c["parameters"]["ANALOG"]["LABELS"]["value"]
MVC = []
LABEL_MUSCLE = []
EMG_NAME = ["GF.IM EMG1", "MF.IM EMG2", "DF.IM EMG3", "VM.IM EMG4", "ST.IM EMG5", "LA.IM EMG6", "So.IM EMG7", "GM.IM EMG8", "TA.IM EMG9", "LF.IM EMG10"]

# --- Grand fessier --- #
LABEL_MUSCLE.append("Grand fessier")
glut_max = emg_processed(file_path=path + 'MVC_GlutMax.c3d', name="GF.IM EMG1")
glut_max01 = emg_processed(file_path=path + 'MVC_GlutMax01.c3d', name="GF.IM EMG1")
mvc_glut_max = get_mvc_value((glut_max.data, glut_max01.data))
MVC.append(mvc_glut_max)

# --- Moyen fessier --- #
LABEL_MUSCLE.append("Moyen fessier")
glut_med = emg_processed(file_path=path + 'MVC_GlutMed.c3d', name="MF.IM EMG2")
glut_med01 = emg_processed(file_path=path + 'MVC_GlutMed01.c3d', name="MF.IM EMG2")
mvc_glut_med = get_mvc_value((glut_med.data, glut_med01.data))
MVC.append(mvc_glut_med)

# --- Droit anterieur --- #
LABEL_MUSCLE.append("Droit anterieur")
rect_fem = emg_processed(file_path=path + 'MVC_RectFem.c3d', name="DF.IM EMG3")
rect_fem01 = emg_processed(file_path=path + 'MVC_RectFem01.c3d', name="DF.IM EMG3")
mvc_rect_fem = get_mvc_value((rect_fem.data, rect_fem01.data))
MVC.append(mvc_rect_fem)

# --- Vaste medial --- #
LABEL_MUSCLE.append("Vaste medial")
vaste_med = emg_processed(file_path=path + 'MVC_VastMed.c3d', name="VM.IM EMG4")
vaste_med01 = emg_processed(file_path=path + 'MVC_VastMed01.c3d', name="VM.IM EMG4")
mvc_vaste_med = get_mvc_value((vaste_med.data, vaste_med01.data))
MVC.append(mvc_vaste_med)

# --- Semi tendineux --- #
LABEL_MUSCLE.append("Semi tendineux")
semi_ten = emg_processed(file_path=path + 'MVC_SemiTen.c3d', name="ST.IM EMG5")
semi_ten01 = emg_processed(file_path=path + 'MVC_SemiTen01.c3d', name="ST.IM EMG5")
mvc_semi_ten = get_mvc_value((semi_ten.data, semi_ten01.data))
MVC.append(mvc_semi_ten)

# --- Long adducteur --- #
LABEL_MUSCLE.append("Long adducteur")
add = emg_processed(file_path=path + 'MVC_Add.c3d', name="LA.IM EMG6")
add01 = emg_processed(file_path=path + 'MVC_Add01.c3d', name="LA.IM EMG6")
mvc_add = get_mvc_value((add.data, add01.data))
MVC.append(mvc_add)

# --- Soleaire --- #
LABEL_MUSCLE.append("Soleaire")
sol = emg_processed(file_path=path + 'MVC_Soleaire.c3d', name="So.IM EMG7")
sol01 = emg_processed(file_path=path + 'MVC_Soleaire01.c3d', name="So.IM EMG7")
mvc_sol = get_mvc_value((sol.data, sol01.data))
MVC.append(mvc_sol)

# --- Gastrocnemien medial --- #
LABEL_MUSCLE.append("Gastrocnemien medial")
gastroc_med = emg_processed(file_path=path + 'MVC_GastrocMed.c3d', name="GM.IM EMG8")
gastroc_med01 = emg_processed(file_path=path + 'MVC_GastrocMed01.c3d', name="GM.IM EMG8")
mvc_gastroc_med = get_mvc_value((gastroc_med.data, gastroc_med01.data))
MVC.append(mvc_gastroc_med)

# --- Tibial anterieur --- #
LABEL_MUSCLE.append("Tibial anterieur")
tib_ant = emg_processed(file_path=path + 'MVC_TibAnt.c3d', name="TA.IM EMG9")
tib_ant01 = emg_processed(file_path=path + 'MVC_TibAnt01.c3d', name="TA.IM EMG9")
tib_ant02 = emg_processed(file_path=path + 'MVC_TibAnt02.c3d', name="TA.IM EMG9")
mvc_tib_ant = get_mvc_value((tib_ant.data, tib_ant01.data, tib_ant02.data))
MVC.append(mvc_tib_ant)

# --- Long fibulaire --- #
LABEL_MUSCLE.append("Long fibulaire")
long_fib = emg_processed(file_path=path + 'MVC_LongFib.c3d', name="LF.IM EMG10")
long_fib01 = emg_processed(file_path=path + 'MVC_LongFib01.c3d', name="LF.IM EMG10")
mvc_long_fib = get_mvc_value((long_fib.data, long_fib01.data))
MVC.append(mvc_long_fib)

for (i,m) in enumerate(MVC):
    print(f"MVC {LABEL_MUSCLE[i]} : {m}")