# Disini ada nge-train lalu meng-klasifikasi

import os
import modulKlasifikasiCitra as MD

dirpath = MD.os.getcwd()
sDir = dirpath+'/../'+'datasets/'

print("using dir: ", sDir)

LabelKelas = []
LabelKelas.append("Belok_Kiri")
LabelKelas.append("Dilarang_Masuk")
LabelKelas.append("Hati_Hati")

TargetKelas = ([1, 0, 0],
               [0, 1, 0],
               [0, 0, 1])

md, history = MD.TrainingCNN(10, sDir, LabelKelas, 'weight.h5', 1)

DirKelas = os.path.join(sDir, LabelKelas[0])

ret = MD.Prediksia(DirKelas, md)
