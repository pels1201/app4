import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from zplane import zplane

z1 = 0.8j
z2 = -0.8j
p1 = 0.95*np.exp(1j*np.pi/8)
p2 = 0.95*np.exp(-1j*np.pi/8)

# --------------------------------------------------------------------------------------------------------
# A
num = np.poly([z1, z2])
denum = np.poly([p1, p2])
zeroes, poles, k = zplane(num, denum)

print("zeroes are : {0} ".format(zeroes))   # zeroes are : [-0.+0.8j  0.-0.8j] 
print("poles are : {0} ".format(poles))     # poles are : [0.87768556+0.36354926j 0.87768556-0.36354926j] 

# --------------------------------------------------------------------------------------------------------
# B
# oui le filtre est stable car les poles sont dans le cercle unitaire

# --------------------------------------------------------------------------------------------------------
# C
H_w = num/denum

w, H_n = signal.freqz(num, denum)

plt.figure()
plt.plot(20*np.log(H_n))
# plt.savefig("Rep_en_freq.png")
plt.title("Rep_en_freq")
plt.show()


# --------------------------------------------------------------------------------------------------------
# D
s = np.zeros(500)
s[250] = 1

h_n = signal.lfilter(num, denum, s)

plt.figure()
plt.stem(h_n)
plt.title("Rep_a impl")
plt.show()

# --------------------------------------------------------------------------------------------------------
# E
# on fait 1/h_n
# --------------------------------------------------------------------------------------------------------
# F
h_n_inv = signal.lfilter(denum, num, h_n)

plt.figure()
plt.stem(h_n_inv)
plt.title("signal")
plt.show()

