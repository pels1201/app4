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
    img_abberations = np.load("goldhill_aberrations.npy")

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

# **********************************************************************************************************************
# 2. Faire une rotation de l image
# **********************************************************************************************************************



# **********************************************************************************************************************
# 3. Enlever le bruit de l image
# **********************************************************************************************************************

def enlever_bruit():
    img_bruit = np.load("goldhill_bruit.npy")
    plt.figure()
    plt.imshow(img_bruit)
    plt.title('img_bruit')

# **********************************************************************************************************************
# 4. Compression
# **********************************************************************************************************************

# img_complete = np.load("image_complete.npy")
#
# plt.figure()
# plt.imshow(img_complete)
# plt.title('img_complete')
# plt.show()

# **********************************************************************************************************************
# Main
# **********************************************************************************************************************
if __name__ == '__main__':
    correction_abberations()

    plt.show()

