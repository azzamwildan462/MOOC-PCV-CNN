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
        CropFrame = p[2]

        hs = PrediksiCitra2(CropFrame, md)
        idx = LabelAnotasi2(TargetKelas, hs)
        if idx > -1:
            new_label = LabelKelas[idx].removesuffix('_banyak')
            color = (0, 0, 255)
            thickness = 2
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
sDir_test_vid = dirpath+'/../'+'tests/vids/'

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

# Load model yang sudah di-train
md = load_model('weight_final.h5')

# Load video
video_file = sDir_test_vid+'video0.mp4'
cap = cv2.VideoCapture(video_file)
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame = Klasifikasi(frame, md, LabelKelas, TargetKelas)
        frame = cv2.resize(frame, (640, 360))
        # Menampilkan Hasil Klasifikasi
        cv2.imshow('Hasil', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
