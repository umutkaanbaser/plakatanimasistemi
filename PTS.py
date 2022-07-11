import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from alg1_plaka_tespiti import plaka_konum_don
from alg2_plaka_tanima import plakaTani
veriler = os.listdir("veriseti")

isim = veriler[3]
for isim in veriler:
    print("resim:","veriseti/"+isim)
    img = cv2.imread("veriseti/"+isim)
    img = cv2.resize(img,(500,500))

    plaka = plaka_konum_don(img)
    plakaImg,plakaKarakter = plakaTani(img,plaka)
    print("resimdeki plaka:",plakaKarakter)
    plt.imshow(plakaImg)
    plt.show()




























