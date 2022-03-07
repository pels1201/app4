# S5 APP4
# Labo1

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

import math
from zplane import zplane

# Probleme 2

omega_barre = np.pi / 16

# Il faut mettre les zeros exactement ou on veut couper (a pi / 16)
# Les poles il faut mettre tres proches a 0.95 de notre zero. Sur le meme angle

numerateurs = np.poly([np.exp(1j*np.pi/16), np.exp(-1j*np.pi/16)])
denums = np.poly([0.98*np.exp(1j*np.pi/16), 0.98*np.exp(-1j*np.pi/16)])

zeros, poles, k = zplane(numerateurs, denums)

print('zeros = ', zeros)
print('poles = ', poles)
print('k = ', k)

w, H_omega = signal.freqz(numerateurs, denums)
plt.figure()
plt.plot(20 * np.log10(H_omega))
plt.title('H_z en dB')

n = np.arange(0, 1000, 1)
signal_x = np.sin(n * np.pi /16) + np.sin(n * np.pi /32)

plt.figure()
plt.plot(signal_x)
plt.title('signal_x')

sortie = signal.lfilter(numerateurs, denums, signal_x)

plt.figure()
plt.plot(sortie)
plt.title('sortie')

plt.show()
