import os
from keras.models import load_model
import cv2
import numpy as np
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, InputLayer
from keras.models import Model


def LoadCitraTraining(sDir, LabelKelas):
    n = len(LabelKelas)
    TargetKelas = np.eye(n)
    X = []
    D = []
    for i in range(len(LabelKelas)):
        print("asu: ", sDir, " jmbbt: ", LabelKelas[i])
        DirKelas = os.path.join(sDir, LabelKelas[i])

        files = os.listdir(DirKelas)
        for f in files:
            ff = f.lower()
            # print(f)
            if (ff.endswith('.jpg') | ff.endswith('.jpeg') | ff.endswith('.png')):
                NmFile = os.path.join(DirKelas, f)
                img = np.double(cv2.imread(NmFile, 1))
                img = cv2.resize(img, (128, 128))
                img = np.asarray(img)
                img = np.asarray(img)/255
                img = img.astype('float32')
                X.append(img)
                D.append(TargetKelas[i])
    X = np.array(X)
    D = np.array(D)
    X = X.astype('float32')
    D = D.astype('float32')
    return X, D


def Prediksia(DirKelas, ModelCNN):
    X = []
    files = os.listdir(DirKelas)
    n = 0
    for f in files:
        ff = f.lower()
        print(f)
        if (ff.endswith('.jpg') | ff.endswith('.jpeg') | ff.endswith('.png')):
            n = n+1
            NmFile = os.path.join(DirKelas, f)
            img = cv2.imread(NmFile, 1)
            img = cv2.resize(img, (128, 128))
            img = np.asarray(img)
            img = np.asarray(img)/255
            img = img.astype('float32')
            X.append(img)
    X = np.array(X)
    X = X.astype('float32')
    hs = ModelCNN.predict(X)
    lr = []
    print(n)
    for i in range(n):
        v = hs[i, :]
        if v.max() > 0.8:
            idx = np.max(np.where(v == v.max()))
        else:
            idx = -1
        lr.append(idx)

    return lr


def ModelDeepLearningCNN(JumlahKelas):
    input_img = Input(shape=(128, 128, 3))
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    Output = Dense(JumlahKelas, activation='softmax')(x)
    ModelCNN = Model(input_img, Output)
   # ModelCNN.compile(loss='categorical_crossentropy', optimizer='adam',                      metrics=['accuracy'])
    ModelCNN.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return ModelCNN


def TrainingCNN(JumlahEpoh, DirektoriDataSet, LabelKelas,  NamaFileModel, bTrainingBaru):
    X, D = LoadCitraTraining(DirektoriDataSet, LabelKelas)
    JumlahKelas = len(LabelKelas)
    if bTrainingBaru == 1:
        ModelCNN = ModelDeepLearningCNN(JumlahKelas)
    else:
        ModelCNN = load_model(NamaFileModel)
    history = ModelCNN.fit(X, D, epochs=JumlahEpoh, shuffle=True)
    ModelCNN.save(NamaFileModel)
    return ModelCNN, history


def PrediksiCitra(img, ModelCNN):
    im = cv2.resize(img, (128, 128))
    v = np.array(im)
    v = v.astype('float32')
    v = v/255
    hs = ModelCNN.predict(v.reshape(1, 128, 128, 3))
    return hs


def LabelAnotasi(hs, Label):
    l = []
    for i in range(len(hs)):
        if hs[i] >= 0:
            l.append(Label[hs[i]])
        else:
            l.append('Tidak Di temukan')

    return l


def PrediksiCitra2(img, ModelCNN):
    im = cv2.resize(img, (128, 128))
    v = np.array(im)
    v = v.astype('float32')
    v = v/255
    hs = ModelCNN.predict(v.reshape(1, 128, 128, 3))
    return hs


def LabelAnotasi2(TargetKelas, hs):
    if hs.max() > 0.8:
        idx = np.max(np.where(hs == hs.max()))
    else:
        idx = -1
    return idx
