# S5 APP4
# Problematique
# Simon Pelletier (PELS1201)
# Zachary Thuotte (THUZ3401)

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import math
import sys
from zplane import zplane

# **********************************************************************************************************************
# 1. Trouver et appliquer la fonction de transfert inverse
# **********************************************************************************************************************

def correction_abberations():

    plt.gray()
    # img_abberations = np.load("goldhill_aberrations.npy")
    img_abberations = np.load("image_complete.npy")

    plt.figure()
    plt.imshow(img_abberations)
    plt.title('img_abberations')
    # plt.show()

    # H = ((z - 0.9*math.exp(1j*np.pi/2)) * (z - 0.9*math.exp(-1j*np.pi/2)) * (z - 0.95*math.exp(1j*np.pi/8)) * (z - 0.95*math.exp(-1j*np.pi/8)) / (z * math.pow(z + 0.99, 2) * (z - 0.8))

    denominateurs = np.poly([0.9*np.exp(1j*np.pi/2), 0.9*np.exp(-1j*np.pi/2), 0.95*np.exp(1j*np.pi/8), 0.95*np.exp(-1j*np.pi/8)])
    numerateurs = np.poly([0, -0.99, -0.99, 0.8])

    plt.figure()
    zeros, poles, k = zplane(numerateurs, denominateurs, 'fonction_transfert.jpg')

    print('zeros = ', zeros)
    print('poles = ', poles)

    # H_z_inverse = numerateurs / denominateurs
    # plt.figure()
    # plt.plot(H_z_inverse)
    # plt.savefig('Fonction de transfert inverse')
    # plt.show()

    # print(img_abberations)

    # img_sortie = []
    # for colonne in range(img_abberations.shape[1]):
    #     img_sortie.append(signal.lfilter(numerateurs, denominateurs, img_abberations[:,colonne]))
    #     # img_sortie.append((img_abberations[:,colonne]))
    #
    # img_sortie = np.transpose(img_sortie)

    img_sortie = signal.lfilter(numerateurs, denominateurs, img_abberations)

    # print(np.shape(img_abberations))
    # print(np.shape(img_sortie))

    # print(img_abberations)
    # # print(img_abberations[:,0])
    # print(img_sortie)

    plt.figure()
    plt.imshow(img_sortie)
    plt.title('img_sortie')
    plt.show()

    return img_sortie

# **********************************************************************************************************************
# 2. Faire une rotation de l image
# **********************************************************************************************************************
def rotation(img_rotation):
    plt.gray()
    # img_rotation = mpimg.imread('goldhill_rotate.png')
    # img_rotation = np.mean(img_rotation, -1)

    M_tranfo = [[0,1],[-1, 0]]

    rows, columns = np.shape(img_rotation)
    new_img = np.zeros([int(columns), int(rows)])
    for i in range(0, rows):
        for j in range(0, columns):
            x = j * M_tranfo[0][0] + i * M_tranfo[1][0]
            y = j * M_tranfo[0][1] + i * M_tranfo[1][1]
            new_img[int(y)][int(x)] = img_rotation[i][j]

    plt.imshow(new_img)
    plt.show()

    return new_img

# **********************************************************************************************************************
# 3. Enlever le bruit de l image
# **********************************************************************************************************************

def enlever_bruit(img_bruit):

    plt.gray()
    # img_bruit = np.load("goldhill_bruit.npy")
    #
    # plt.figure()
    # plt.imshow(img_bruit)
    # plt.title('img_bruit')

    fs = 1600  # Hz
    fc = 500 # Hz
    fc_max = 750 # Hz
    gain = 0.2 # 0 +- 0.2 dB de 0 a 500 Hz
    gain_max = 60 # -60 dB a 750 Hz

    order, wn = signal.buttord(wp=fc / (fs / 2), ws= (fc_max / (fs / 2)), gpass=gain, gstop=gain_max)

    print('Butterworth')
    print('order = ', order)
    print('wn = ', wn)

    # b, a = signal.butter(N=order, Wn=wn)

    # Chebyshev I
    order, wn = signal.cheb1ord(wp=fc / (fs / 2), ws=fc_max / (fs / 2), gpass=gain, gstop=gain_max)

    print('Chevyshev I')
    print('order = ', order)
    print('wn = ', wn)

    # rp est 0.2 pour 0.2 dB
    # b, a = signal.cheby1(N=order, Wn=wn, rp=0.2)

    # Chebyshev II
    order, wn = signal.cheb2ord(wp=fc / (fs / 2), ws=fc_max / (fs / 2), gpass=gain, gstop=gain_max)

    print('Chevyshev II')
    print('order = ', order)
    print('wn = ', wn)

    # Elliptique
    order, wn = signal.ellipord(wp=fc / (fs / 2), ws=fc_max / (fs / 2), gpass=gain, gstop=gain_max)

    print('Elliptique')
    print('order = ', order)
    print('wn = ', wn)

    b, a = signal.ellip(N=order, Wn=wn, rp=gain, rs=gain_max)

    img_sortie = signal.lfilter(b, a, img_bruit)

    plt.figure()
    plt.imshow(img_sortie)
    plt.title('img_sortie filtree avec elliptique')
    plt.show()

    return img_sortie

# **********************************************************************************************************************
# 4. Compression
# **********************************************************************************************************************
# Retirer 50 pourcent des lignes
def retirer_50(img):
    new_img = np.zeros([img.shape[0], img.shape[1]])
    for i in range(new_img.shape[0]):
        if (i % 2 == 0):
            for j in range(new_img.shape[1]):
                new_img[i][j] = img[i][j]
    return new_img

# Retirer 70 pourcent des lignes
def retirer_70(img):
    new_img = np.zeros([img.shape[0], img.shape[1]])
    for i in range(new_img.shape[0]):
        if (i % 10 < 3):
            for j in range(new_img.shape[1]):
                new_img[i][j] = img[i][j]
    return new_img

def compression(img_complete):

    plt.gray()

    # plt.figure()
    # plt.imshow(img_complete)
    # plt.title('img_complete')
    # plt.show()

    print(img_complete.shape)

    matrice_covariance = np.cov(img_complete)

    w, v = np.linalg.eig(matrice_covariance)

    # w, v = np.linalg.eig(np.array([[4, 1], [2, 5]]))
    # w est valeurs propres
    # v est vecteur propre

    # print(w)
    # print(v)

    # La matrice de passage est directement v
    m_p = v;
    m_p_inv = np.linalg.inv(m_p)

    # matrice_passage = []
    #
    # for vecteur_propre in v:
    #     matrice_passage.append(vecteur_propre)
    # print(matrice_passage)

    img_compresse = np.matmul(img_complete, m_p)

    img_compresse = retirer_70(img_compresse)

    plt.figure()
    plt.imshow(img_compresse)
    plt.title('img_compresse')
    # plt.show()

    img_decompresse = np.matmul(img_compresse, m_p_inv)

    plt.figure()
    plt.imshow(img_decompresse)
    plt.title('img_decompresse')
    plt.show()



# **********************************************************************************************************************
# Main
# **********************************************************************************************************************
if __name__ == '__main__':
    img = correction_abberations()
    img = rotation(img)
    img = enlever_bruit(img)
    # img = np.load("image_complete.npy")
    # img = mpimg.imread('imagecouleur.png')
    # img = np.mean(img, -1)
    # print(img)
    compression(img)
