# S5 APP4
# Problematique
# Simon Pelletier (PELS1201)
# Zachary Thuotte (THUZ3401)

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import math

from zplane import zplane

# **********************************************************************************************************************
# 1. Trouver et appliquer la fonction de transfert inverse
# **********************************************************************************************************************

img_abberations = np.load("goldhill_aberrations.npy")

# H = ((z - 0.9*math.exp(1j*np.pi/2)) * (z - 0.9*math.exp(-1j*np.pi/2)) * (z - 0.95*math.exp(1j*np.pi/8)) * (z - 0.95*math.exp(-1j*np.pi/8)) / (z * math.pow(z + 0.99, 2) * (z - 0.8))


# numerateurs = np.poly([0.9*np.exp(1j*np.pi/2), 0.9*np.exp(-1j*np.pi/2), 0.95*np.exp(1j*np.pi/8), 0.95*np.exp(-1j*np.pi/8)])
# denums = np.poly([0.95*np.exp(1j*np.pi/8), 0.95*np.exp(-1j*np.pi/8)])
# zeros, poles, k = zplane(numerateurs, denums)

plt.figure()
plt.imshow(img_abberations)
# plt.title('img_abberations')
# plt.show()

# **********************************************************************************************************************
# 2. Faire une rotation de l image
# **********************************************************************************************************************

# **********************************************************************************************************************
# 3. Enlever le bruit de l image
# **********************************************************************************************************************

img_bruit = np.load("goldhill_bruit.npy")
plt.figure()
plt.imshow(img_bruit)
# plt.title('img_abberations')
# plt.show()


# **********************************************************************************************************************
# 4. Compression
# **********************************************************************************************************************

img_complete = np.load("image_complete.npy")

plt.figure()
plt.imshow(img_complete)
# plt.title('img_abberations')
plt.show()
