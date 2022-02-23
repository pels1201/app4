from turtle import clear
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from zplane import zplane




# 1
#########################################################################################################
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
w, H_n = signal.freqz(num, denum)

plt.figure()
plt.plot(20*np.log(H_n))
# plt.savefig("Rep_en_freq.png")
plt.title("Rep_en_freq")



# --------------------------------------------------------------------------------------------------------
# D
s = np.zeros(500)
s[250] = 1

h_n = signal.lfilter(num, denum, s)

plt.figure()
plt.stem(h_n)
plt.title("Rep_a impl")


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

# 2
#########################################################################################################

w_barre = np.pi/16
n = np.arange(0, 1000, 1)
x_n = np.sin(n*np.pi/16)+np.sin(n*np.pi/32)
# --------------------------------------------------------------------------------------------------------
# A
plt.figure()
plt.plot(x_n)
plt.title("signal")

# enlever le signal pi/16
z1 = np.exp(1j*np.pi/16)
z2 = np.exp(-1j*np.pi/16)
p1 = 0.95*np.exp(1j*np.pi/16)
p2 = 0.95*np.exp(-1j*np.pi/16)

num = np.poly([z1, z2])
denum = np.poly([p1, p2])
h_n = signal.lfilter(num, denum, x_n)

plt.figure()
plt.plot(h_n)
plt.title("signal filtree")
plt.show()

# trouver poles, zeros
zeroes, poles, k = zplane(num, denum)

#  trouver rep en frequence
w, H_n = signal.freqz(num, denum)

plt.figure()
plt.plot(20*np.log(H_n))
plt.title("Rep_en_freq")
plt.show()


# 3
#########################################################################################################
Fe = 48000
Fc = 2500
# --------------------------------------------------------------------------------------------------------
# A
# butterworth

order, wn = signal.buttord(wp=Fc/(Fe/2), ws=3500/(Fe/2), gpass=0.2, gstop=40)
b, a = signal.butter(order, wn)

print("order is {0}".format(order))

w, H_w = signal.freqz(b, a)
plt.figure()
plt.plot(20*np.log(H_w))
plt.title("Rep_en_freq de butterworth")



# --------------------------------------------------------------------------------------------------------
# B
# Chebyshev 1


order, wn = signal.cheb1ord(wp=Fc/(Fe/2), ws=3500/(Fe/2), gpass=0.2, gstop=40)
b, a = signal.cheby1(order,0.2 , wn)

print("order is {0}".format(order))

w, H_w = signal.freqz(b, a)
plt.figure()
plt.plot(20*np.log(H_w))
plt.title("Rep_en_freq de Chebyshev 1")


# --------------------------------------------------------------------------------------------------------
# C
# Chebyshev 2


order, wn = signal.cheb2ord(wp=Fc/(Fe/2), ws=3500/(Fe/2), gpass=0.2, gstop=40)
b, a = signal.cheby2(order,40 , wn)

print("order is {0}".format(order))

w, H_w = signal.freqz(b, a)
plt.figure()
plt.plot(20*np.log(H_w))
plt.title("Rep_en_freq de Chebyshev 2")

# --------------------------------------------------------------------------------------------------------
# D
# Elliptique


order, wn = signal.ellipord(wp=Fc/(Fe/2), ws=3500/(Fe/2), gpass=0.2, gstop=40)
b, a = signal.ellip(order, 0.2, 40, wn)

print("order is {0}".format(order))

w, H_w = signal.freqz(b, a)
plt.figure()
plt.plot(20*np.log(H_w))
plt.title("Rep_en_freq de l'Elliptique")
plt.show()


# 4
#########################################################################################################
plt.gray()
img_couleur = mpimg.imread('imagecouleur.png')
img_gris= np.mean(img_couleur, -1)

# --------------------------------------------------------------------------------------------------------
# A 
# transformation lineaire T(x,y)

M_tranfo = [[2,0],[0, 0.5]]

rows, columns = np.shape(img_couleur)
new_img = np.zeros([int(columns/2), int(rows*2)])
for i in range(0, rows):
    for j in range(0, columns):
        x = j*M_tranfo[0][0] + i*M_tranfo[1][0]
        y = j*M_tranfo[0][1] + i*M_tranfo[1][1]
        new_img[int(y)][int(x)] = img_couleur[i][j]

plt.imshow(new_img)
plt.show()
