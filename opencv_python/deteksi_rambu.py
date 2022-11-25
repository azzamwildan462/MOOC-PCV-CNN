import cv2
import os
from modulEkstraksiRambu import *
from modulKlasifikasiCitra import *


def Klasifikasi(frame, modelCNN, LabelKelas, TargetKelas):
    Fr = copy.deepcopy(frame)
    Im, ls = EkstraksiRambu(Fr)
    for p in ls:
        p1 = p[0]
        p2 = p[1]
        print(p1)
        CropFrame = p[2]

        hs = PrediksiCitra2(CropFrame, md)
        idx = LabelAnotasi2(TargetKelas, hs)
        if idx > -1:
            new_label = LabelKelas[idx].removesuffix('_banyak')
            print(new_label)
            color = (255, 255, 255)
            thickness = 1
            fontScale = 0.8
            font = cv2.FONT_HERSHEY_SIMPLEX
            Fr = cv2.putText(Fr, new_label, p1, font,
                             fontScale, color, thickness,
                             cv2.LINE_AA)
            Fr = cv2.rectangle(Fr, p1, p2, color, thickness)
    return Fr


dirpath = os.getcwd()
sDir_datasets = dirpath+'/../'+'datasets/'
sDir_test_img = dirpath+'/../'+'tests/imgs/'

LabelKelas = []
LabelKelas.append("Belok_Kiri_banyak")
LabelKelas.append("Dilarang_Masuk_banyak")
LabelKelas.append("Hati_Hati_banyak")

# Ketika belum training, maka uncomment dibawah ini
# md, history = TrainingCNN(10, sDir_datasets, LabelKelas, 'weight_final.h5', 1)

# Ketika sudah melakukan training
img_test = sDir_test_img+'image0.jpg'
TargetKelas = ([1, 0, 0],
               [0, 1, 0],
               [0, 0, 1])

md = load_model('weight_final.h5')
frame = cv2.imread(img_test)


frame = Klasifikasi(frame, md, LabelKelas, TargetKelas)

cv2.imshow('result', frame)
cv2.waitKey()
cv2.destroyAllWindows()
