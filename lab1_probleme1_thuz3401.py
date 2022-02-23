# S5 APP4
# Labo1

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

import math
from zplane import zplane

# Probleme 1

# H = ((z - 0.8j) * (z+0.8j)) / ((z - 0.95*math.exp(1j*np.py/8)) * (z - 0.95*math.exp(-1j*np.py/8)))

# a)

numerateurs = np.poly([0.8j, -0.8j])
denums = np.poly([0.95*np.exp(1j*np.pi/8), 0.95*np.exp(-1j*np.pi/8)])
zeros, poles, k = zplane(numerateurs, denums)

print('zeros = ', zeros)
print('poles = ', poles)
print('k = ', k)

# H_z = numerateurs / denums
# plt.figure()
# plt.plot(H_z)
# plt.savefig('probleme1c')
# plt.show()

# b)
# Reponse prof: Oui parceque les poles sont dans le cercle

# c)
w, H_omega = signal.freqz(numerateurs, denums)

plt.figure()
plt.plot(20 * np.log10(H_omega))
plt.title('H_omega en dB')
plt.savefig('probleme1c')
# plt.show()

# d)
# Utiliser signal.lfilter()
# Prend en parametres les coefficients et le signal a filtrer (impulsion). Retourne en sortie la reponse

signal_impulsion = np.zeros(1000)
signal_impulsion = np.append(signal_impulsion, 1)
signal_impulsion = np.append(signal_impulsion, np.zeros(1000))

# plt.figure()
# plt.plot(impulsion)
# plt.title('Impulsion')

h_n = signal.lfilter(numerateurs, denums, signal_impulsion)

plt.figure()
plt.stem(h_n)
plt.title('h_n')

# e)
# On inverse les poles et les zeros du filtre

# f)
sortie = signal.lfilter(denums, numerateurs, h_n)

plt.figure()
plt.plot(sortie)
plt.title('sortie')

plt.show()

