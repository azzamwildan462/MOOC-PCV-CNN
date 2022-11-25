import cv2
import copy
import numpy as np


def SegmentasiWarna(frame, HueRange, SaturationRange, ValueRange):
    (HMin, HMax) = HueRange
    (SMin, SMax) = SaturationRange
    (VMin, VMax) = ValueRange
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if HMin < 0:
        LColor = np.array([180+HMin//2, SMin, VMin])
        UColor = np.array([180, SMax, VMax])
        mask1 = cv2.inRange(hsv, LColor, UColor)
        LColor = np.array([0, SMin, VMin])
        UColor = np.array([HMax//2, SMax, VMax])
        mask2 = cv2.inRange(hsv, LColor, UColor)
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        LColor = np.array([HMin//2, SMin, VMin])
        UColor = np.array([HMax//2, SMax, VMax])
        mask = cv2.inRange(hsv, LColor, UColor)
    Im = cv2.bitwise_and(frame, frame, mask=mask)
    return Im, mask


def DeteksiROI(frame, mask, skala, ListDT):
    nLabels, Labels = cv2.connectedComponents(mask)
    for i in range(1, nLabels):
        [bb, cc] = np.where(Labels == i)
        bmax = (np.max(bb))
        bmin = (np.min(bb))
        # print("jmbbt: ", bmax, " asd: ", bmin)
        b = bmax-bmin
        cmax = (np.max(cc))
        cmin = (np.min(cc))
        c = cmax-cmin
        if b == 0:
            b = 1
        rasio = c/b
        r = 0.6
        if (c > 20) & (b > 20) & (rasio >= r) & (rasio <= 1/r):
            # Mengembalikan koordinat ke ukuran se mula
            cmin = int(cmin/skala)
            cmax = int(cmax/skala)
            bmin = int(bmin/skala)
            bmax = int(bmax/skala)
            CropFrame = frame[bmin:bmax, cmin:cmax]
            ListDT.append(((cmin, bmin), (cmax, bmax), CropFrame))
    return ListDT


def DeteksiRambu(frame, HueRange, SaturationRange, ValueRange, ListDT):
    FrameNorm = copy.deepcopy(frame)
    skala = 426/frame.shape[0]
    FrameNorm = cv2.resize(FrameNorm, None, fx=skala, fy=skala)
    (height, width, w) = FrameNorm.shape
    res, mask = SegmentasiWarna(
        FrameNorm, HueRange, SaturationRange, ValueRange)
    DeteksiROI(frame, mask, skala, ListDT)
    return ListDT, mask


def EkstraksiRambu(frame):
    ListDT = []
    # Deteksi Rambu lalulintas Warna Kuning
    ListDT, mask1 = DeteksiRambu(
        frame, (20, 60), (60, 255), (200, 255), ListDT)
    # Deteksi Rambu lalulintas Warna Merah
    ListDT, mask2 = DeteksiRambu(
        frame, (-20, 20), (100, 255), (20, 255), ListDT)
    # Deteksi Rambu lalulintas Warna Biru
    ListDT, mask3 = DeteksiRambu(
        frame, (200, 280), (150, 255), (60, 250), ListDT)

    color = (255, 0, 0)
    thickness = 2
    fr = copy.deepcopy(frame)
    for p in ListDT:
        p1 = p[0]
        p2 = p[1]
        fr = cv2.rectangle(fr, p1, p2, color, thickness)
    return fr, ListDT
