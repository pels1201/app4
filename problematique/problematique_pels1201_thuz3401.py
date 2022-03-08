# Universite de Sherbrooke
# S5 H22 GI APP4 Problematique
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

def corriger_aberrations(img_aberrations):
    # img_aberrations = np.load("goldhill_aberrations.npy")

    # H = ((z - 0.9*math.exp(1j*np.pi/2)) * (z - 0.9*math.exp(-1j*np.pi/2)) * (z - 0.95*math.exp(1j*np.pi/8)) * (z - 0.95*math.exp(-1j*np.pi/8)) / (z * math.pow(z + 0.99, 2) * (z - 0.8))
    numerateurs = np.poly([0, -0.99, -0.99, 0.8])
    denominateurs = np.poly([0.9*np.exp(1j*np.pi/2), 0.9*np.exp(-1j*np.pi/2), 0.95*np.exp(1j*np.pi/8), 0.95*np.exp(-1j*np.pi/8)])

    # plt.figure()
    plt.title('Pôles et zéros de la fonction de transfert inverse pour retirer aberrations')
    zeros, poles, k = zplane(numerateurs, denominateurs, 'pz_aberrations_inv.jpg')
    # print('zeros = ', zeros)
    # print('poles = ', poles)

    # H_z_inverse = numerateurs / denominateurs
    # plt.figure()
    # plt.plot(H_z_inverse)

    img_sortie = signal.lfilter(numerateurs, denominateurs, img_aberrations)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img_aberrations)
    ax1.set_title('Image avec aberrations')
    ax2.imshow(img_sortie)
    ax2.set_title('Image sans aberrations')

    return img_sortie

# **********************************************************************************************************************
# 2. Rotation de l image
# **********************************************************************************************************************
def appliquer_rotation(img_rotation):
    # img_rotation = np.mean(mpimg.imread('goldhill_rotate.png', -1))

    M_tranfo = [[0, 1], [-1, 0]]

    rows, columns = np.shape(img_rotation)
    new_img = np.zeros([int(columns), int(rows)])
    for i in range(0, rows):
        for j in range(0, columns):
            x = j * M_tranfo[0][0] + i * M_tranfo[1][0]
            y = j * M_tranfo[0][1] + i * M_tranfo[1][1]
            new_img[int(y)][int(x)] = img_rotation[i][j]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img_rotation)
    ax1.set_title('Image avant rotation')
    ax2.imshow(new_img)
    ax2.set_title('Image après rotation')

    return new_img

# **********************************************************************************************************************
# 3. Enlever le bruit de l image
# **********************************************************************************************************************

def calculer_bilineraire(img_bruit):
    '''
    Enlever le bruit d une image en la filtrant par un filtre trouve manuellement
    :param img_bruit: Image dont on veut retirer le bruit
    :return: Image filtree avec moins de bruit
    '''

    fs = 1600  # Hz

    # Numerateurs et denominateurs trouves manuellement
    numerateurs = np.array([0.418, 0.836, 0.418])
    denominateurs = np.array([1, 0.462, 0.21])

    plt.figure()
    plt.title('Pôles et zéros du filtre Butterworth d\'ordre 2 déterminé avec transformation bilinéaire')
    zeros, poles, k = zplane(numerateurs, denominateurs, 'pz_bilineaire.jpg')

    # print('zeros = ', zeros)
    # print('poles = ', poles)

    w, h = signal.freqz(numerateurs, denominateurs)

    plt.figure()
    plt.title('Module de la réponse en fréquence du filtre Butterworth d\'ordre 2 déterminé avec transformation bilinéaire')
    plt.plot(w * fs / (2 * np.pi), 20 * np.log10(abs(h)))
    plt.ylabel('Amplitude (dB)')
    plt.xlabel('Fréquence (Hz)')

    img_sortie = signal.lfilter(numerateurs, denominateurs, img_bruit)

    plt.figure()
    plt.imshow(img_sortie)
    plt.title('Image debruitée avec filtre Butterworth ordre 2 déterminé avec transformation bilinéaire')

def enlever_bruit(img_bruit):
    '''
    Enlever le bruit d une image en la filtrant par un filtre le meilleur possible
    :param img_bruit: Image dont on veut retirer le bruit
    :return: Image filtree avec moins de bruit
    '''

    # img_bruit = np.load("goldhill_bruit.npy")

    # plt.figure()
    # plt.imshow(img_bruit)
    # plt.title('img_bruit')

    fs = 1600  # Hz
    fc = 500 # Hz
    fc_max = 750 # Hz
    gain = 0.2 # 0 +- 0.2 dB de 0 a 500 Hz
    gain_max = 60 # -60 dB a 750 Hz

    # Butterworth ****************************************************************
    order, wn = signal.buttord(wp=fc / (fs / 2), ws= (fc_max / (fs / 2)), gpass=gain, gstop=gain_max)

    print('Butterworth')
    print('order = ', order)
    print('wn = ', wn)

    # b, a = signal.butter(N=order, Wn=wn)

    # Chebyshev I ****************************************************************
    order, wn = signal.cheb1ord(wp=fc / (fs / 2), ws=fc_max / (fs / 2), gpass=gain, gstop=gain_max)

    print('Chevyshev I')
    print('order = ', order)
    print('wn = ', wn)

    # rp est 0.2 dB
    # b, a = signal.cheby1(N=order, Wn=wn, rp=gain)

    # Chebyshev II ****************************************************************
    order, wn = signal.cheb2ord(wp=fc / (fs / 2), ws=fc_max / (fs / 2), gpass=gain, gstop=gain_max)

    print('Chevyshev II')
    print('order = ', order)
    print('wn = ', wn)

    # b, a = signal.cheby2(N=order, Wn=wn, rs=gain_max)

    # Elliptique ****************************************************************
    order, wn = signal.ellipord(wp=fc / (fs / 2), ws=fc_max / (fs / 2), gpass=gain, gstop=gain_max)

    print('Elliptique')
    print('order = ', order)
    print('wn = ', wn)

    b, a = signal.ellip(N=order, Wn=wn, rp=gain, rs=gain_max)

    plt.figure()
    plt.title('Pôles et zéros du filtre elliptique d\'ordre ' + str(order))
    zeros, poles, k = zplane(b, a, 'pz_elliptique.jpg')

    w, h = signal.freqz(b, a)

    plt.figure()
    plt.title('Module de la réponse en fréquence du filtre elliptique d\'ordre ' + str(order))
    plt.plot(w * fs / (2 * np.pi), 20 * np.log10(abs(h)))
    plt.ylabel('Amplitude (dB)')
    plt.xlabel('Fréquence (Hz)')

    img_sortie = signal.lfilter(b, a, img_bruit)

    plt.figure()
    plt.imshow(img_sortie)
    plt.title('Image débruitée avec filtre numérique elliptique d\'ordre ' + str(order))

    return img_sortie

# **********************************************************************************************************************
# 4. Compression
# **********************************************************************************************************************
# Retirer 50 pourcent des lignes
# def retirer_50(img):
#     new_img = np.zeros([img.shape[0], img.shape[1]])
#     for i in range(new_img.shape[0]):
#         if (i % 2 == 0):
#             for j in range(new_img.shape[1]):
#                 new_img[i][j] = img[i][j]
#     return new_img

# Retirer 70 pourcent des lignes
# def retirer_70(img):
#     new_img = np.zeros([img.shape[0], img.shape[1]])
#     for i in range(new_img.shape[0]):
#         if (i % 10 < 3):
#             for j in range(new_img.shape[1]):
#                 new_img[i][j] = img[i][j]
#     return new_img

def retirer_pourcentage(img, pourcentage):
    '''
    Remplacer les dernier pourcentage de lignes de l image par des lignes noires
    :param img: l image originale
    :param pourcentage: le pourcentage de lignes de l image a retirer a la fin
    :return: l image avec lignes remplacees par du noir
    '''
    new_img = np.zeros([img.shape[0], img.shape[1]])
    for i in range(int(img.shape[0] * (1 - pourcentage))):
            for j in range(new_img.shape[1]):
                new_img[i][j] = img[i][j]
    return new_img

def compresser_decompresser(img_complete):

    matrice_covariance = np.cov(img_complete)

    w, v = np.linalg.eig(matrice_covariance)
    # w est valeurs propres
    # v est vecteurs propre

    # Verification matrice A
    # w, v = np.linalg.eig(np.array([[4, 1], [2, 5]]))
    # print(w) # w est valeurs propres
    # print(v) # v est vecteur propre

    # La matrice de passage est directement v
    m_p = v

    # m_p = []
    # for vecteur_propre in v:
    #     m_p.append(vecteur_propre)
    # print(m_p)

    img_compresse = np.matmul(m_p.T, img_complete)
    img_compresse_50 = retirer_pourcentage(img_compresse, 0.5)
    img_compresse_70 = retirer_pourcentage(img_compresse, 0.7)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img_compresse_50)
    ax1.set_title('Image après compression (50%)')
    ax2.imshow(img_compresse_70)
    ax2.set_title('Image après compression (70%)')

    img_decompresse_50 = np.matmul(np.linalg.inv(m_p.T), img_compresse_50)
    img_decompresse_70 = np.matmul(np.linalg.inv(m_p.T), img_compresse_70)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img_decompresse_50)
    ax1.set_title('Image après compression et decompression (50%)')
    ax2.imshow(img_decompresse_70)
    ax2.set_title('Image après compression et decompression (70%)')

# **********************************************************************************************************************
# Main
# **********************************************************************************************************************
if __name__ == '__main__':
    plt.gray()
    img = np.load("image_complete.npy")

    # 1.
    img = corriger_aberrations(img)

    # 2.
    img = appliquer_rotation(img)

    # 3.
    calculer_bilineraire(img) # Pour verifier le resultat manuel sans conserver pour la prochaine etape
    img = enlever_bruit(img) # Pour conserver le resultat du filtre numerique elliptique pour la prochaine etape

    # 4.
    compresser_decompresser(img)

    plt.show()
