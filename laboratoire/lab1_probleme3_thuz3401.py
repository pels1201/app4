# S5 APP4
# Labo1

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

import math
from zplane import zplane

# Probleme 3

#a)

fs = 48000 # Hz

# On divise par fs / 2 pour rammener entre 0 et 1
order, wn = signal.buttord(wp = 2500 /  (fs / 2), ws = 3500 /  (fs / 2), gpass = 0.2, gstop = 40)

print('Butterworth')
print('order = ', order)
print('wn = ', wn)

b, a = signal.butter(N = order, Wn = wn)

# print('b (numerateurs) = ', b)
# print('a (denominateurs) = ', a)

# L'ordre est 18

w, H_omega = signal.freqz(b, a)
plt.figure()
plt.plot(20 * np.log10(H_omega))
plt.title('H_z de Butterworth en dB')
plt.xlabel('numero dechantillon n')
plt.ylabel('Gain (dB)')

plt.figure()

zeros, poles, k = zplane(b, a)

# b)

order, wn = signal.cheb1ord(wp = 2500 /  (fs / 2), ws = 3500 /  (fs / 2), gpass = 0.2, gstop = 40)

print('Chevyshev I')
print('order = ', order)
print('wn = ', wn)

# rp est 0.2 pour 0.2 dB
b, a = signal.cheby1(N = order, Wn = wn, rp = 0.2)

w, H_omega = signal.freqz(b, a)
plt.figure()
plt.plot(20 * np.log10(H_omega))
plt.title('H_z de Chebyshev II en dB')
plt.xlabel('numero dechantillon n')
plt.ylabel('Gain (dB)')

# c)

order, wn = signal.cheb2ord(wp = 2500 /  (fs / 2), ws = 3500 /  (fs / 2), gpass = 0.2, gstop = 40)

print('Chevyshev II')
print('order = ', order)
print('wn = ', wn)

# rs est 0.2 pour -40 dB
b, a = signal.cheby2(N = order, Wn = wn, rs = 40)

w, H_omega = signal.freqz(b, a)
plt.figure()
plt.plot(20 * np.log10(H_omega))
plt.title('H_z de Chebyshev II en dB')
plt.xlabel('numero dechantillon n')
plt.ylabel('Gain (dB)')

# d)

order, wn = signal.ellipord(wp = 2500 /  (fs / 2), ws = 3500 /  (fs / 2), gpass = 0.2, gstop = 40)

print('Elliptique')
print('order = ', order)
print('wn = ', wn)

b, a = signal.ellip(N = order, Wn = wn, rp = 0.2 ,rs = 40)

w, H_omega = signal.freqz(b, a)
plt.figure()
plt.plot(20 * np.log10(H_omega))
plt.title('H_z de Elliptique en dB')
plt.xlabel('numero dechantillon n')
plt.ylabel('Gain (dB)')

# Ordres attendus
# 18    butter
# 8     cheby1
# 8     cheby2
# 5     elliptique

plt.show()
